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

@torch.no_grad()
def main():
    args = parse_args()

    from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
    in_path = '/data/nvme1/yxpeng/imagenet/'
    in_info_path = '/data/nvme1/yxpeng/imagenet'
    in_hier = ImageNetHierarchy(in_path, in_info_path)
    superclass_wnid = common_superclass_wnid('mixed_10')
    class_ranges, label_map = in_hier.get_subclasses(['n02958343'],
                                                                        balanced=False)
    info("[i] Loading model and tokenizer...")
    vit_config = ViTConfig()
    vit_config.embedding_bias = False
    vit_config.layernorm_pre = True
    vit_config.layer_norm_eps = 1e-5
    vit_config.proj = True
    vit_config.hidden_act = 'quick_gelu'
    clip_checkpoint = "/data/nvme1/yxpeng/PycharmProjects/transfer_learning/logs/linprobe_imagenet_clip_vit_b16/weights_0.pkl"
    circuit_checkpoint = "/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/data/runs/mean_ablate-qkv_False-LP-class-n02958343-kl_loss-clip-imagenet-train-wo_node_loss-elr0.9-llr0.9-relr0.9-rllr0.9-lrw70-sw800-tbs8-es1.1-ns0.72-t1000/model.safetensors"
    if 'qkv_False' in circuit_checkpoint:
        model = ClipDisentangleModel('ViT-B/16', vit_config, include_qkv=False, checkpoint=circuit_checkpoint).to(args.device)
        model.eval()
        control_model = ClipDisentangleModel('ViT-B/16', vit_config, include_qkv=False, checkpoint=clip_checkpoint).to(args.device).eval()
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
                                   transform=transform, select_class=0)
    val_loader = DataLoader(imagenet_val, batch_size=args.batch_size, shuffle=True)

    if args.sparsity_edge is not None:
        # Binary search for the threshold
        info("[i] Searching for threshold...")
        l = 0
        r = 1
        while r-l > 1e-5:
            threshold = (l+r)/2
            model._model.visual.set_edge_threshold_for_deterministic(threshold)
            sparsity = model._model.visual.get_edge_sparsity()
            if sparsity > args.sparsity_edge:
                r = threshold
            else:
                l = threshold
        info(f"[i] Edge Threshold found: {threshold}")
        info(f"[i] Edge Sparsity: {sparsity}")
    else:
        threshold = args.threshold_edge
        if threshold is None:
            info("[i] No edge threshold specified")
        else:
            info(f"[i] Using edge threshold: {threshold}")
            model._model.visual.set_edge_threshold_for_deterministic(threshold)

    if args.sparsity_node is not None:
        # Binary search for the threshold
        info("[i] Searching for threshold...")
        l = 0
        r = 1
        while r-l > 1e-5:
            threshold = (l+r)/2
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
    logit_sums = torch.zeros((1000)).to(args.device)
    control_logit_sums = torch.zeros((1000)).to(args.device)
    outputs_ = []
    i = 0
    with open("/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/activations/mean_LP_clip_imagenet_train.pkl",
              "rb") as f:  # "rb" = read binary
        loaded_data = pickle.load(f)
    # for zero-ablate found circuit!!
    # loaded_data = torch.zeros_like(loaded_data)
    info("[i] Producing outputs...")
    bar = tqdm(val_loader)
    for images, labels in bar:
        images = images.to(args.device)
        # images = torch.zeros_like(images).to(args.device)
        labels = labels.to(args.device)

        control_outputs = control_model(images, labels=labels)
        outputs = model(images, labels=labels, corr_x=loaded_data.unsqueeze(1).repeat(1, len(images), 1, 1).to(args.device))
        # reverse masks, check completeness
        reverse_outputs = model(images, labels=labels, corr_x=loaded_data.unsqueeze(1).repeat(1, len(images), 1, 1).to(args.device), reverse=True)
        close_all_outputs = model(images, labels=labels, corr_x=loaded_data.unsqueeze(1).repeat(1, len(images), 1, 1).to(args.device), close_all=True)
        logits = outputs.logits
        logit_sum = logits.sum(dim=0)
        control_logits = control_outputs.logits
        control_logit_sum = control_logits.sum(dim=0)
        reverse_logits = reverse_outputs.logits
        close_all_logits = close_all_outputs.logits
        logit_sums += logit_sum
        control_logit_sums += control_logit_sum

        for j in range(images.shape[0]):
            logit_target = logits[j, labels[j]].detach().cpu().item()
            masked_logits = logits[j].clone()
            masked_logits[labels[j]] = float('-inf')  # Replace target logit with -inf

            # Get the highest value after masking
            max_except_target = torch.max(masked_logits)
            distractor_class = torch.argmax(masked_logits)

            logit_difference += logit_target - max_except_target
            chosen_class = torch.argmax(logits[j])

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
        bar.set_description(f"Acc: {accuracy/i:.5f}, LD: {logit_difference/i:.5f}, CACC: {control_accuracy/i:.5f}, RACC:{reverse_accuracy/i:.5f}, CAACC: {close_all_accuracy/i:.5f} EM: {exact_match/i:.3f}")

    accuracy /= len(val_loader.dataset)
    logit_difference /= len(val_loader.dataset)
    control_accuracy /= len(val_loader.dataset)
    reverse_accuracy /= len(val_loader.dataset)
    exact_match /= len(val_loader.dataset)
    logit_sums /= len(val_loader.dataset)
    logit_sums = logit_sums.detach().cpu().numpy()
    control_logit_sums /= len(val_loader.dataset)
    control_logit_sums = control_logit_sums.detach().cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(logit_sums, marker='o', linestyle='-', markersize=2, alpha=0.7)
    plt.xlabel("Class Index")
    plt.ylabel("Logit")
    plt.title("Visualization of circuit 1000 Logits")
    plt.grid(True)
    top_5_indices = np.argsort(logit_sums)[-5:]
    top_5_values = logit_sums[top_5_indices]
    # Highlight top 5 logits
    plt.scatter(top_5_indices, top_5_values, color='red', s=50, label="Top 5 Logits", edgecolors='black')

    # Annotate the top 5 points
    for i in range(5):
        plt.annotate(f"({top_5_indices[i]}, {top_5_values[i]:.2f})", (top_5_indices[i], top_5_values[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='red')

    plt.savefig(f'/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/assets/logits/LP_superclass_car_circuit_logits_on_class_0_images')

    plt.figure(figsize=(10, 5))
    plt.plot(control_logit_sums, marker='o', linestyle='-', markersize=2, alpha=0.7)
    plt.xlabel("Class Index")
    plt.ylabel("Logit")
    plt.title("Visualization of full model 1000 Logits")
    plt.grid(True)
    top_5_indices = np.argsort(control_logit_sums)[-5:]
    top_5_values = control_logit_sums[top_5_indices]
    # Highlight top 5 logits
    plt.scatter(top_5_indices, top_5_values, color='red', s=50, label="Top 5 Logits", edgecolors='black')

    # Annotate the top 5 points
    for i in range(5):
        plt.annotate(f"({top_5_indices[i]}, {top_5_values[i]:.2f})", (top_5_indices[i], top_5_values[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='red')

    plt.savefig(
        f'/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/assets/logits/LP_full_model_logits_on_class_0_images')

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