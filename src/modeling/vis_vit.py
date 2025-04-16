import os
import json
import argparse

import sys

from safetensors.torch import load_file
from transformers import ViTConfig

from src.modeling.vit import ViTHeadModel

sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/modeling/"
    )
)
from clip_model import ClipDisentangleModel

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_path", "-i", type=str, required=True)
    parser.add_argument("--out_path", "-o", type=str, default=None)
    parser.add_argument("--with_embedding_nodes", "-w", action="store_true")
    parser.add_argument("--edge_sparsity", "-e", type=float, default=None)
    parser.add_argument("--node_sparsity", "-n", type=float, default=None)

    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.in_path), "edges.json")
        print(f"Output path not specified. Saving to {args.out_path}.")

    return args

def main():
    args = parse_args()

    model_name = 'google-pretrain'
    with_embedding = False
    state_dict = load_file(args.in_path)
    if model_name == 'tiny':
        vit_config = ViTConfig(image_size=28, patch_size=7, num_hidden_layers=2, num_attention_heads=4,
                               intermediate_size=256 * 3,
                               num_channels=3, num_labels=1)
        model = ViTHeadModel(
            config=vit_config,
            include_qkv=False,
            with_embedding_nodes=with_embedding,
            disable_linear_regularization_term=False,
        ).eval()
        model.load_state_dict(state_dict)
    elif model_name == 'google':
        model = ViTHeadModel.from_pretrained(
            'google/vit-base-patch16-224',
            state_dict=state_dict,
            include_qkv=False,
            with_embedding_nodes=with_embedding,
            disable_linear_regularization_term=False,
        ).eval()
    elif model_name == 'google-pretrain':
        model = ViTHeadModel.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            state_dict=state_dict,
            include_qkv=False,
            with_embedding_nodes=with_embedding,
            disable_linear_regularization_term=False,
        ).eval()
    if args.edge_sparsity is None:
        args.edge_sparsity = model.get_edge_sparsity()
    if args.node_sparsity is None:
        args.node_sparsity = model.get_node_sparsity()
                
    l = 0
    r = 1
    while r-l > 1e-5:
        threshold = (l+r)/2
        model.set_edge_threshold_for_deterministic(threshold)
        sparsity = model.get_edge_sparsity()
        if abs(sparsity - args.edge_sparsity) < 1e-3:
            break
        if sparsity > args.edge_sparsity:
            r = threshold
        else:
            l = threshold        

    l = 0
    r = 1
    while r-l > 1e-5:
        threshold = (l+r)/2
        model.set_node_threshold_for_deterministic(threshold)
        sparsity = model.get_node_sparsity()
        if abs(sparsity - args.node_sparsity) < 1e-3:
            break
        if sparsity > args.node_sparsity:
            r = threshold
        else:
            l = threshold

    overall_edge_sparsity = model.get_effective_edge_sparsity()
    print("Overall edge sparsity:", overall_edge_sparsity.item())

    edges = model.get_edges()
                
    json.dump(edges, open(args.out_path, "w+"), indent=4)

if __name__ == '__main__':
    main()