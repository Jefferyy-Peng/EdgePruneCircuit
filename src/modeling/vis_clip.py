import os
import json
import argparse

import sys

from transformers import ViTConfig


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

    vit_config = ViTConfig()
    vit_config.embedding_bias = False
    vit_config.layernorm_pre = True
    vit_config.layer_norm_eps = 1e-5
    vit_config.proj = True
    vit_config.hidden_act = 'quick_gelu'
    model = ClipDisentangleModel('ViT-B/16', vit_config, include_qkv=False, checkpoint=args.in_path).eval()
    if args.edge_sparsity is None:
        args.edge_sparsity = model._model.visual.get_edge_sparsity()
    if args.node_sparsity is None:
        args.node_sparsity = model._model.visual.get_node_sparsity()
                
    l = 0
    r = 1
    while r-l > 1e-5:
        threshold = (l+r)/2
        model._model.visual.set_edge_threshold_for_deterministic(threshold)
        sparsity = model._model.visual.get_edge_sparsity()
        if sparsity > args.edge_sparsity:
            r = threshold
        else:
            l = threshold        

    l = 0
    r = 1
    while r-l > 1e-5:
        threshold = (l+r)/2
        model._model.visual.set_node_threshold_for_deterministic(threshold)
        sparsity = model._model.visual.get_node_sparsity()
        if sparsity > args.node_sparsity:
            r = threshold
        else:
            l = threshold

    overall_edge_sparsity = model._model.visual.get_effective_edge_sparsity()
    print("Overall edge sparsity:", overall_edge_sparsity.item())

    edges = model._model.visual.get_edges()
                
    json.dump(edges, open(args.out_path, "w+"), indent=4)

if __name__ == '__main__':
    main()