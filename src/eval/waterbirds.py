import sys, os
import random

from src.custom_datasets import ColoredMNIST, WaterbirdDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import json
import argparse
import pickle

import numpy as np
from matplotlib import pyplot as plt
from safetensors.torch import load_file
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from tqdm import tqdm
from scipy.stats import kendalltau

import torch
from transformers import AutoTokenizer, ViTConfig
from datasets import load_from_disk

import sys

from src.modeling.clip_model import ClipDisentangleModel
from src.modeling.vit import ViTHeadModel
from src.utils import ImageNetDataset

sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/modeling/"
    )
)  # Very hacky but the imports are annoying otherwise


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
    parser.add_argument("--control_model_name_or_path", "-cm", default="data/runs/example/")
    parser.add_argument("--with-embedding-nodes", "-w",
                        action="store_true")  # TRUE if the run allowed removing embedding nodes
    # Here WITH means that masks were modeled over embedding nodes

    parser.add_argument("--sparsity-edge", "-se", default=None,
                        type=float)  # If you want to override the sparsity of the model
    parser.add_argument("--sparsity-node", "-sn", default=None,
                        type=float)  # If you want to override the sparsity of the model
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

def compute_total_effect(logit_bias, logit_antibias, label):
    if logit_bias.shape[-1] > 1:
        y_anti = torch.softmax(logit_antibias, dim=-1)[label]
        y = torch.softmax(logit_bias, dim=-1)[label]
    else:
        y_anti = torch.sigmoid(logit_antibias)
        y = torch.sigmoid(logit_bias)
        if label == 0:
            y = 1 - y
            y_anti = 1 - y_anti
    TE = (y - y_anti)
    return TE

def compute_indirect_effect(logit_bias_intervene, logit_antibias, label):
    if logit_bias_intervene.shape[-1] > 1:
        y_anti = torch.softmax(logit_antibias, dim=-1)[label]
        y = torch.softmax(logit_bias_intervene, dim=-1)[label]
    else:
        y_anti = torch.sigmoid(logit_antibias)
        y = torch.sigmoid(logit_bias_intervene)
        if label == 0:
            y = 1 - y
            y_anti = 1 - y_anti
    TE = (y - y_anti)
    return TE

def build_class_index(dataset):
    class_index = {}
    for idx, label in enumerate(dataset.y_array):
        label = str(label)
        bg = dataset.place_array[idx]
        bg = str(bg)  # Ensure consistent key type
        if f'({label}, {bg})' not in class_index.keys():
            class_index[f'({label}, {bg})'] = []
        class_index[f'({label}, {bg})'].append(idx)
    return class_index


@torch.no_grad()
def main():
    args = parse_args()

    embedding = False
    ablate_type = 'bg'
    select_class = 0
    info("[i] Loading model and tokenizer...")
    ckpt = torch.load(args.control_model_name_or_path)
    new_state_dict = {}
    for old_key, value in ckpt.items():
        if "vit.encoder.layer" in old_key:
            new_key = old_key.replace("vit.encoder.layer", "vit.encoder")
            new_state_dict[new_key] = value
        else:
            new_state_dict[old_key] = value
    control_model = ViTHeadModel.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        state_dict=new_state_dict,
        include_qkv=False,
        with_embedding_nodes=embedding,
        disable_linear_regularization_term=False,
    ).eval().to(args.device)
    circuit_checkpoint = args.model_name_or_path
    circuit_state_dict = load_file(circuit_checkpoint)
    model = ViTHeadModel.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        state_dict=circuit_state_dict,
        include_qkv=False,
        with_embedding_nodes=embedding,
        disable_linear_regularization_term=False,
    ).eval().to(args.device)
    reverse_model = ViTHeadModel.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        state_dict=circuit_state_dict,
        include_qkv=False,
        with_embedding_nodes=embedding,
        disable_linear_regularization_term=False,
    ).eval().to(args.device)

    if args.sparsity_edge is None:
        args.sparsity_edge = model.get_edge_sparsity()
    if args.sparsity_node is None:
        args.sparsity_node = model.get_node_sparsity()

    info("[i] Loading data...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = WaterbirdDataset(data_correlation=0.95, split="train",
                               root_dir="/data/nvme1/yxpeng/PycharmProjects/vit-spurious-robustness/datasets",
                               transform=transform)
    ood_dataset = WaterbirdDataset(data_correlation=0.95, split="val",
                               root_dir="/data/nvme1/yxpeng/PycharmProjects/vit-spurious-robustness/datasets",
                               transform=transform)

    if ablate_type == 'bg' or (ablate_type == 'prediction' and select_class == 'all'):
        dataset_index = build_class_index(dataset)
    elif ablate_type == 'prediction':
        opposite_dataset = ColoredMNIST(root='/data/nvme1/yxpeng/PycharmProjects/pyvenv-experiments/vision-grokking/new_data', env='all_train', select_class=1 - select_class,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                           ]))

    # ood_dataset = ColoredMNIST(root='/data/nvme1/yxpeng/PycharmProjects/pyvenv-experiments/vision-grokking/new_data', env='test',
    #                        transform=transforms.Compose([
    #                            transforms.ToTensor(),
    #                            transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
    #                        ]))
    # ood_class_index = build_class_index(ood_dataset)
    train_size = int(0.8 * len(dataset))
    print(f'tarin_size: {train_size}')
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if ablate_type == 'mean':
        with open(
                f"/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/activations/mean_DRO_92.125_new_colored_mnist_all_train_unbiased.pkl",
                "rb") as f:  # "rb" = read binary
            loaded_data = pickle.load(f).to(args.device)
        loaded_data = loaded_data.unsqueeze(1).repeat(1, args.batch_size, 1, 1)

    if args.sparsity_edge is not None:
        # Binary search for the threshold
        info("[i] Searching for threshold...")
        l = 0
        r = 1
        while r - l > 1e-5:
            threshold = (l + r) / 2
            model.set_edge_threshold_for_deterministic(threshold)
            sparsity = model.get_edge_sparsity()
            if abs(sparsity - args.sparsity_edge) < 1e-3:
                break
            if sparsity > args.sparsity_edge:
                r = threshold
            else:
                l = threshold
        info(f"[i] Edge Threshold found: {threshold}")
        info(f"[i] Edge Sparsity: {sparsity}")
        reverse_model.set_edge_threshold_for_deterministic(threshold)
    else:
        threshold = args.threshold_edge
        if threshold is None:
            info("[i] No edge threshold specified")
        else:
            info(f"[i] Using edge threshold: {threshold}")
            model.set_edge_threshold_for_deterministic(threshold)
            reverse_model.set_edge_threshold_for_deterministic(threshold)


    if args.sparsity_node is not None:
        # Binary search for the threshold
        info("[i] Searching for threshold...")
        l = 0
        r = 1
        while r - l > 1e-5:
            threshold = (l + r) / 2
            model.set_node_threshold_for_deterministic(threshold)
            sparsity = model.get_node_sparsity()
            if abs(sparsity - args.sparsity_node) < 1e-3:
                break
            if sparsity > args.sparsity_node:
                r = threshold
            else:
                l = threshold
        info(f"[i] Node Threshold found: {threshold}")
        info(f"[i] Node Sparsity: {sparsity}")
        reverse_model.set_node_threshold_for_deterministic(threshold)
    else:
        threshold = args.threshold_node
        if threshold is None:
            info("[i] No node threshold specified")
        else:
            info(f"[i] Using node threshold: {threshold}")
            model.set_node_threshold_for_deterministic(threshold)
            reverse_model.set_node_threshold_for_deterministic(threshold)

    overall_edge_sparsity = model.get_effective_edge_sparsity()
    info(f"[i] Overall Edge Sparsity: {overall_edge_sparsity}")
    reverse_model.set_effective_edge_mask(reverse=True)

    accuracy = 0
    logit_difference = 0
    exact_match = 0
    control_accuracy = 0
    reverse_accuracy = 0
    close_all_accuracy = 0
    reverse_match = 0
    close_all_match = 0
    control_ctft_match = 0
    TE = 0
    IE = 0
    outputs_ = []
    i = 0

    # for zero-ablate found circuit!!
    # loaded_data = torch.zeros_like(loaded_data)
    info("[i] Producing outputs...")
    bar = tqdm(val_loader)
    for images, labels, bgs in bar:
        images = images.to(args.device)
        # images = torch.zeros_like(images).to(args.device)
        labels = labels.to(args.device)

        if ablate_type == 'bg':
            ctft_images_list = [dataset[random.choice(dataset_index[f'({label.item()}, {1 - bg.item()})'])][0] for label, bg
                                in zip(labels, bgs)]  # get the opposite class image
            ctft_images = torch.stack(ctft_images_list).to(args.device)
        elif ablate_type == 'prediction' and select_class == 'all':
            ctft_images_list = [dataset[random.choice(dataset_index[f'({label.item()}, {bg.item()})'])][0] for label, bg in zip(labels, bgs)] # get the opposite class image
            ctft_images = torch.stack(ctft_images_list).to(args.device)
        elif ablate_type == 'prediction':
            ctft_images_list = [opposite_dataset[random.choice(range(len(opposite_dataset)))][0] for i in range(len(labels))] # get the opposite class image
            ctft_images = torch.stack(ctft_images_list).to(args.device)
        elif ablate_type == 'mean':
            ctft_images = None

        control_outputs = control_model(images, labels=labels)
        if ablate_type == 'mean':
            corr_x = loaded_data
            control_ctft_output = control_model(images, corr_x=corr_x, close_all=True)
        else:
            control_ctft_output = control_model(ctft_images, labels=labels, output_writer_states=True)
            corr_x = control_ctft_output.writer_states
        outputs = model(images, labels=labels,
                        corr_x=corr_x)
        # reverse masks, check completeness
        reverse_outputs = reverse_model(images, labels=labels,
                                corr_x=corr_x)
        close_all_outputs = model(images, labels=labels,
                                  corr_x=corr_x,
                                  close_all=True)
        logits = outputs.logits
        control_logits = control_outputs.logits
        control_ctft_logits = control_ctft_output.logits
        reverse_logits = reverse_outputs.logits
        close_all_logits = close_all_outputs.logits

        for j in range(images.shape[0]):
            logit_target = logits[j].detach().cpu()
            TE += compute_total_effect(control_logits[j], control_ctft_logits[j], labels[j]).detach().cpu().item()
            IE += compute_indirect_effect(control_logits[j], reverse_logits[j], labels[j]).detach().cpu().item()
            chosen_class = torch.argmax(torch.softmax(logits[j], dim=-1)).int()

            accuracy += (chosen_class == labels[j]).int().detach().cpu().item()
            control_choice =  torch.argmax(torch.softmax(control_logits[j], dim=-1)).int()
            reverse_choice =  torch.argmax(torch.softmax(reverse_logits[j], dim=-1)).int()
            close_all_choice =  torch.argmax(torch.softmax(close_all_logits[j], dim=-1)).int()
            control_ctft_choice =  torch.argmax(torch.softmax(control_ctft_logits[j], dim=-1)).int()
            control_accuracy += (control_choice == labels[j]).int().detach().cpu().item()
            reverse_accuracy += (reverse_choice == labels[j]).int().detach().cpu().item()
            close_all_accuracy += (close_all_choice == labels[j]).int().detach().cpu().item()
            exact_match += (chosen_class == control_choice).int().detach().cpu().item()
            reverse_match += (reverse_choice == control_choice).int().detach().cpu().item()
            close_all_match += (close_all_choice == control_choice).int().detach().cpu().item()
            control_ctft_match += (control_ctft_choice == control_choice).int().detach().cpu().item()

            outputs_.append({
                "image": images[j],
                "target": labels[j],
                "chosen_class": chosen_class,
                "logit_target": logit_target,
                "choice": chosen_class.item(),
            })
            i += 1
        bar.set_description(
            f"Acc: {accuracy / i:.5f}, CACC: {control_accuracy / i:.5f}, RM:{reverse_match / i:.5f}, EM: {exact_match / i:.3f}, CAM: {close_all_match / i:.3f}, CCFTM: {control_ctft_match / i:.3f}, TE: {TE / i:.3f}, IE: {IE / i:.3f}")

    accuracy /= len(val_loader.dataset)
    logit_difference /= len(val_loader.dataset)
    control_accuracy /= len(val_loader.dataset)
    reverse_accuracy /= len(val_loader.dataset)
    exact_match /= len(val_loader.dataset)
    reverse_match /= len(val_loader.dataset)

    info(f"[i]     Accuracy: {accuracy}")
    info(f"[i]     Control accuracy: {control_accuracy}")
    info(f"[i]     Reverse match: {reverse_match}")
    info(f"[i]     Exact Match: {exact_match}")

    if args.out_path is not None:
        info(f"[i] Saving outputs to {args.out_path}...")
        json.dump(outputs_, open(args.out_path, "w+"), indent=4)


if __name__ == '__main__':
    main()