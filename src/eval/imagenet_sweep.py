import os
import json
import argparse
import pickle

import numpy as np
from matplotlib import pyplot as plt
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from scipy.stats import kendalltau

import torch
from transformers import AutoTokenizer, ViTConfig
from datasets import load_from_disk

import sys

from src.modeling.clip_model import ClipDisentangleModel
from src.utils import ImageNetDataset

sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/modeling/"
    )
)   # Very hacky but the imports are annoying otherwise

class bcolors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def info(text):
    print(f"{bcolors.OKBLUE}{text}{bcolors.ENDC}")

def good(text):
    print(f"{bcolors.OKGREEN}{text}{bcolors.ENDC}")

def bad(text):
    print(f"{bcolors.FAIL}{text}{bcolors.ENDC}")

@torch.no_grad()
def load_avg_activations(model, path, device):
    avg_activations = pickle.load(open(path, "rb"))
    model.load_captured_activations(avg_activations.to(device))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", "-m", default="data/runs/example/")
    parser.add_argument("--with-embedding-nodes", "-w", action="store_true") # TRUE if the run allowed removing embedding nodes
                                                                             # Here WITH means that masks were modeled over embedding nodes
    
    parser.add_argument("--sparsity-edge", "-se", default=None, type=float) # If you want to override the sparsity of the model
    parser.add_argument("--sparsity-node", "-sn", default=None, type=float) # If you want to override the sparsity of the model
    parser.add_argument("--split", "-s", default="test")
    parser.add_argument("--data-path", "-d", default="./data/datasets/ioi/")
    parser.add_argument("--num-examples", "-n", default=1000000, type=int)

    parser.add_argument("--device", "-D", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", "-b", default=16, type=int)
    parser.add_argument("--out-path", "-o", default=None)

    args = parser.parse_args()
    
    if args.out_path == "None":
        args.out_path = None

    return args

def try_fit_template(string, template):
    pieces_s, pieces_t = string.strip(), template.strip()
    
    mapping = {}
    
    for s, t in zip(pieces_s.split(), pieces_t.split()):
        if s == t:
            continue
        if s[-1] == t[-1] and s[-1] in [',', '.']:
            s, t = s[:-1], t[:-1]
        if t not in ['{A}', '{B}', '{PLACE}', '{OBJECT}']:
            return None
        elif t[1:-1].lower() in mapping:
            if mapping[t[1:-1].lower()] != s:
                return None
        else:
            mapping[t[1:-1].lower()] = s
    
    if 'place' not in mapping:
        mapping['place'] = None
    if 'object' not in mapping:
        mapping['object'] = None
    
    return mapping

def find_template(string):
    for template in baba_templates:
        mapping = try_fit_template(string, template)
        if mapping is not None:
            mapping.update({
                'template': template,
                'order': 'baba'
            })
            return mapping
    
    for template in abba_templates:
        mapping = try_fit_template(string, template)
        if mapping is not None:
            mapping.update({
                'template': template,
                'order': 'abba'
            })
            return mapping
    return None

def collect_accuracies(args, model_path, sweep_center, circuit_class=None):
    info("[i] Loading model and tokenizer...")
    vit_config = ViTConfig()
    vit_config.embedding_bias = False
    vit_config.layernorm_pre = True
    vit_config.layer_norm_eps = 1e-5
    vit_config.proj = True
    vit_config.hidden_act = 'quick_gelu'
    clip_checkpoint = "/data/nvme1/yxpeng/PycharmProjects/transfer_learning/logs/full_ft_imagenet_clip_vit_b16/optimizer.args.lr-0.001_with_augment_seed-0_run0/checkpoints/ckp_best_val"
    circuit_checkpoint = os.path.join(model_path, "model.safetensors")
    if 'qkv_False' in circuit_checkpoint:
        model = ClipDisentangleModel('ViT-B/16', vit_config, include_qkv=False, checkpoint=circuit_checkpoint).to(
            args.device)
        model.eval()
        control_model = ClipDisentangleModel('ViT-B/16', vit_config, include_qkv=False, checkpoint=clip_checkpoint).to(
            args.device).eval()
    else:
        model = ClipDisentangleModel('ViT-B/16', vit_config, checkpoint=circuit_checkpoint).to(args.device)
        model.eval()
        control_model = ClipDisentangleModel('ViT-B/16', vit_config, checkpoint=clip_checkpoint).to(args.device).eval()

    if args.sparsity_edge is None:
        args.sparsity_edge = model._model.visual.get_edge_sparsity()
    if args.sparsity_node is None:
        args.sparsity_node = model._model.visual.get_node_sparsity()

    info("[i] Loading data...")
    transform = transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=224),
        transforms.ToTensor()
    ])
    imagenet_val = ImageNetDataset(root_dir='/data/nvme1/yxpeng/imagenet/val',
                                   transform=transform, select_class=1)
    val_loader = DataLoader(imagenet_val, batch_size=args.batch_size, shuffle=True)

    small_range = np.linspace(sweep_center - 0.001, sweep_center + 0.01, num=20)
    accuracys = []
    logit_differences = []
    reverse_accuracys = []
    circuit_class_accuracys = []
    for se in small_range:
        info("[i] Searching for threshold...")
        l = 0
        r = 1
        while r - l > 1e-5:
            threshold = (l + r) / 2
            model._model.visual.set_edge_threshold_for_deterministic(threshold)
            sparsity = model._model.visual.get_edge_sparsity()
            if sparsity > se:
                r = threshold
            else:
                l = threshold
        if abs(sparsity - se) > 0.01:
            info(f"[i] Edge Threshold not found")
            accuracys.append(0.5)
            logit_differences.append(0.5)
            reverse_accuracys.append(0.5)
            circuit_class_accuracys.append(0.5)
            continue
        else:
            info(f"[i] Edge Threshold found: {threshold}")
            info(f"[i] Edge Sparsity: {sparsity}")

        if args.sparsity_node is not None:
            # Binary search for the threshold
            info("[i] Searching for threshold...")
            l = 0
            r = 1
            while r - l > 1e-5:
                threshold = (l + r) / 2
                model._model.visual.set_node_threshold_for_deterministic(threshold)
                sparsity = model._model.visual.get_node_sparsity()
                if sparsity > args.sparsity_node:
                    r = threshold
                else:
                    l = threshold
            info(f"[i] Node Threshold found: {threshold}")
            info(f"[i] Node Sparsity: {sparsity}")
        else:
            threshold = args.threshold_node
            if threshold is None:
                info("[i] No node threshold specified")
            else:
                info(f"[i] Using node threshold: {threshold}")
                model._model.visual.set_node_threshold_for_deterministic(threshold)

        overall_edge_sparsity = model._model.visual.get_effective_edge_sparsity()
        info(f"[i] Overall Edge Sparsity: {overall_edge_sparsity}")

        accuracy = 0
        logit_difference = 0
        exact_match = 0
        control_accuracy = 0
        reverse_accuracy = 0
        close_all_accuracy = 0
        circuit_class_accuracy = 0
        outputs_ = []
        i = 0
        with open("/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/activations/mean_clip_imagenet_val_class_0",
                  "rb") as f:  # "rb" = read binary
            loaded_data = pickle.load(f)
        # for zero-ablate found circuit!!
        # loaded_data = torch.zeros_like(loaded_data)
        info("[i] Producing outputs...")
        bar = tqdm(val_loader)
        for images, labels in bar:
            images = images.to(args.device)
            labels = labels.to(args.device)

            control_outputs = control_model(images, labels=labels)
            outputs = model(images, labels=labels,
                            corr_x=loaded_data.unsqueeze(1).repeat(1, len(images), 1, 1).to(args.device))
            # reverse masks, check completeness
            reverse_outputs = model(images, labels=labels,
                                    corr_x=loaded_data.unsqueeze(1).repeat(1, len(images), 1, 1).to(args.device),
                                    reverse=True)
            close_all_outputs = model(images, labels=labels,
                                      corr_x=loaded_data.unsqueeze(1).repeat(1, len(images), 1, 1).to(args.device),
                                      close_all=True)
            logits = outputs.logits
            control_logits = control_outputs.logits
            reverse_logits = reverse_outputs.logits
            close_all_logits = close_all_outputs.logits

            for j in range(images.shape[0]):
                logit_target = logits[j, labels[j]].detach().cpu().item()
                masked_logits = logits[j].clone()
                masked_logits[labels[j]] = float('-inf')  # Replace target logit with -inf

                # Get the highest value after masking
                max_except_target = torch.max(masked_logits)
                distractor_class = torch.argmax(masked_logits)

                logit_difference += logit_target - max_except_target
                chosen_class = torch.argmax(logits[j])

                if circuit_class is not None:
                    circuit_class_accuracy += (chosen_class == circuit_class).int().detach().cpu().item()

                accuracy += (chosen_class == labels[j]).int().detach().cpu().item()
                control_choice = torch.argmax(control_logits[j])
                reverse_choice = torch.argmax(reverse_logits[j])
                close_all_choice = torch.argmax(close_all_logits[j])
                control_accuracy += (control_choice == labels[j]).int().detach().cpu().item()
                reverse_accuracy += (reverse_choice == labels[j]).int().detach().cpu().item()
                close_all_accuracy += (close_all_choice == labels[j]).int().detach().cpu().item()
                exact_match += (chosen_class == control_choice).int().detach().cpu().item()

                outputs_.append({
                    "image": images[j],
                    "target": labels[j],
                    "distractor": distractor_class,
                    "chosen_class": chosen_class,
                    "logit_target": logit_target,
                    "max_except_target": max_except_target,
                    "logit_difference": logit_target - max_except_target,
                    "choice": chosen_class.item(),
                })
                i += 1
            bar.set_description(
                f"Acc: {accuracy / i:.5f}, LD: {logit_difference / i:.5f}, CACC: {control_accuracy / i:.5f}, RACC:{reverse_accuracy / i:.5f}, CAACC: {close_all_accuracy / i:.5f} EM: {exact_match / i:.3f}")

        accuracy /= len(val_loader.dataset)
        logit_difference /= len(val_loader.dataset)
        reverse_accuracy /= len(val_loader.dataset)
        circuit_class_accuracy /= len(val_loader.dataset)
        accuracys.append(accuracy)
        logit_differences.append(logit_difference)
        reverse_accuracys.append(reverse_accuracy)
        circuit_class_accuracys.append(circuit_class_accuracy)

    return small_range, accuracys, reverse_accuracys, circuit_class_accuracys

@torch.no_grad()
def main():
    args = parse_args()

    # ft_model_path = "/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/data/runs/mean_ablate-qkv_False-FT-class-0-target_loss-clip-imagenet-train-wo_node_loss-elr0.9-llr0.9-relr0.9-rllr0.9-es1.1-ns0.72-t1000"
    lp_model_path_1 = "/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/data/runs/mean_ablate-qkv_False-LP-class-0-target_loss-clip-imagenet-train-wo_node_loss-elr0.8-llr0.8-relr0.8-rllr0.8-es1.05-ns0.72-t3000"
    lp_model_path_2 = "/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/data/runs/mean_ablate-qkv_False-LP-class-0-target_loss-clip-imagenet-train-wo_node_loss-elr0.9-llr0.9-relr0.9-rllr0.9-es1.1-ns0.72-t1000"
    lp_model_path_3 = "/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/data/runs/mean_ablate-qkv_False-LP-class-0-target_loss-clip-imagenet-train-wo_node_loss-elr1.0-llr1.0-relr1.0-rllr1.0-es1.1-ns0.72-t500"
    lp_model_path_4 = "/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/data/runs/mean_ablate-qkv_False-LP-class-0-target_loss-clip-imagenet-train-wo_node_loss-elr1.0-llr1.0-relr1.0-rllr1.0-lrw35-sw250-tbs8-es1.1-ns0.72-t500"
    lp_model_path_5 = "/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/data/runs/mean_ablate-qkv_False-LP-class-0-target_loss-clip-imagenet-train-wo_node_loss-elr1.0-llr1.0-relr1.0-rllr1.0-lrw35-sw300-tbs8-es1.1-ns0.72-t500"
    lp_model_path_6 = "/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/data/runs/mean_ablate-qkv_False-LP-class-0-target_loss-clip-imagenet-train-wo_node_loss-elr1.0-llr1.0-relr1.0-rllr1.0-lrw35-sw400-tbs8-es1.2-ns0.72-t500"
    lp_model_path_7 = "/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/data/runs/mean_ablate-qkv_False-LP-class-0-target_loss-clip-imagenet-train-wo_node_loss-elr1.0-llr1.0-relr1.0-rllr1.0-lrw35-sw400-tbs8-es1.3-ns0.72-t500"
    lp_model_path_8 = "/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/data/runs/mean_ablate-qkv_False-LP-class-0-target_loss-clip-imagenet-train-wo_node_loss-elr1.1-llr1.1-relr1.1-rllr1.1-lrw35-sw400-tbs8-es1.1-ns0.72-t500"
    lp_model_path_9 = "/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/data/runs/mean_ablate-qkv_False-LP-class-0-target_loss-clip-imagenet-train-wo_node_loss-elr1.2-llr1.2-relr1.2-rllr1.2-lrw35-sw400-tbs8-es1.1-ns0.72-t500"
    lp_model_path_10 = "/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/data/runs//data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/data/runs/mean_ablate-qkv_False-LP-class-127-kl_loss-clip-imagenet-train-wo_node_loss-elr0.9-llr0.9-relr0.9-rllr0.9-lrw70-sw800-tbs8-es1.1-ns0.72-t1000/edges.pdf"

    # small_range, accuracys_1, reverse_accuracys_1 = collect_accuracies(args, lp_model_path_1, 0.985)
    # small_range, accuracys_2, reverse_accuracys_2, circuit_class_accuracys_2 = collect_accuracies(args, lp_model_path_2, 0.985, circuit_class=0)
    # small_range, accuracys_3, reverse_accuracys_3 = collect_accuracies(args, lp_model_path_3, 0.985)
    # small_range, accuracys_4, reverse_accuracys_4 = collect_accuracies(args, lp_model_path_4, 0.985)
    # small_range, accuracys_5, reverse_accuracys_5 = collect_accuracies(args, lp_model_path_5, 0.985)
    # small_range, accuracys_6, reverse_accuracys_6 = collect_accuracies(args, lp_model_path_6, 0.985)
    # small_range, accuracys_7, reverse_accuracys_7 = collect_accuracies(args, lp_model_path_7, 0.985)
    # small_range, accuracys_8, reverse_accuracys_8 = collect_accuracies(args, lp_model_path_8, 0.985)
    # small_range, accuracys_9, reverse_accuracys_9 = collect_accuracies(args, lp_model_path_9, 0.985)
    small_range, accuracys_10, reverse_accuracys_10, circuit_class_accuracys_10 = collect_accuracies(args, lp_model_path_10, 0.970, circuit_class=127)

    plt.figure(figsize=(8, 5))
    # plt.plot(small_range, accuracys_1, marker='o', label='Accuracy 1')
    # plt.plot(small_range, accuracys_2, marker='^', label='Accuracy')
    # plt.plot(small_range, circuit_class_accuracys_2, marker='+', label='Class 0 Accuracy')
    # plt.plot(small_range, accuracys_3, marker='.', label='Accuracy 3')
    # plt.plot(small_range, accuracys_4, marker='+', label='Accuracy 4')
    # plt.plot(small_range, accuracys_5, marker='*', label='Accuracy 5')
    # plt.plot(small_range, accuracys_6, marker='D', label='Accuracy 6')
    # plt.plot(small_range, accuracys_7, marker='s', label='Accuracy 7')
    # plt.plot(small_range, accuracys_8, marker='v', label='Accuracy 8')
    # plt.plot(small_range, accuracys_9, marker='<', label='Accuracy 9')
    plt.plot(small_range, accuracys_10, marker='<', label='Accuracy')
    plt.plot(small_range, circuit_class_accuracys_10, marker='.', label='class 0 Accuracy')


    # Labels and legend
    plt.xlabel('edge sparsity')
    plt.ylabel('Accuracy')
    plt.title('Acc curve')
    plt.legend()
    plt.grid(True)
    # Save the figure locally
    plt.savefig(os.path.join("/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/assets", 'lp_acc_circuit_class0_test_class1_kl_loss.png'), dpi=300, bbox_inches='tight')

    info(f"[i]     Accuracy: {accuracy}")
    info(f"[i]     Logit difference: {logit_difference}")   
    info(f"[i]     Control accuracy: {control_accuracy}")
    info(f"[i]     Reverse accuracy: {reverse_accuracy}")
    info(f"[i]     Exact Match: {exact_match}")
    
    if args.out_path is not None:
        info(f"[i] Saving outputs to {args.out_path}...")
        json.dump(outputs_, open(args.out_path, "w+"), indent=4)     

if __name__ == '__main__':
    main()