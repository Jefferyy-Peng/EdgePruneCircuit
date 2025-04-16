# Note: this model does normalization.
import json
from typing import Optional, Tuple

from accelerate import Accelerator
from safetensors.torch import load_file
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTConfig, set_seed, AutoImageProcessor
from transformers.utils import ModelOutput
from dataclasses import dataclass
from collections import OrderedDict
import torchvision.models as models
from torchvision.models import resnet50
import torch
from torch import nn
import os
import pickle

# from . import model_utils

from torchvision.transforms import Normalize, transforms
import sys


sys.path.append(
    os.path.join(
        os.getcwd(),
        ""
    )
)
import CLIP.clip as clip
sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/"
    )
)   # Very hacky but the imports are annoying otherwise

from src.utils import ImageNetDataset
from src.modeling.vit_simple import ViTModelSimple
from src.modeling import model_utils
from src.modeling.vit import ViTModel

MODELS = {'RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16',
          'ViT-L/14', 'ViT-L/14@336px'}

normalize_transform = Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711))


def build_model_scratch(model_name, device):
    model_path = clip.clip._download(clip.clip._MODELS[model_name], os.path.expanduser("~/.cache/clip"))
    model = torch.jit.load(model_path, map_location="cpu").eval()
    state_dict = model.state_dict()

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = clip.model.CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    clip.model.convert_weights(model)
    return model.eval().to(device)


def set_requires_grad(component, val):
    for param in component.parameters():
        param.requires_grad = val

@dataclass
class CLIPHeadModelOutput(ModelOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    writer_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    target_edge_sparsity: Optional[torch.FloatTensor] = None
    target_node_sparsity: Optional[torch.FloatTensor] = None
    model_edge_sparsity: Optional[torch.FloatTensor] = None
    model_node_sparsity: Optional[torch.FloatTensor] = None
    edge_loss: Optional[torch.FloatTensor] = None
    node_loss: Optional[torch.FloatTensor] = None

class ClipModel(nn.Module):

    def __init__(self, model_name, scratch=False):
        # If scratch is True, then we randomly initialize weights.
        super().__init__()
        if model_name not in MODELS:
            raise ValueError(f'model_name must be in {MODELS} but was {model_name}')
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Note that model has both a language and vision part.
        if scratch:
            model = build_model_scratch(model_name, device=self._device)
        else:
            model, _ = clip.load(model_name, device=self._device)
        self._model_name = model_name
        self._model = model
        self._model.visual.float()
        self._classifier = None

    def forward(self, x, output_feature=False):
        features = self.get_features(x)
        if self._classifier is None:
            return features
        if output_feature:
            return self._classifier(features), features
        return self._classifier(features)

    def set_requires_grad(self, val):
        for param in self._model.parameters():
            param.requires_grad = val
        if self._classifier is not None:
            for param in self._classifier.parameters():
                param.requires_grad = val

    def get_layers(self):
        visual = self._model.visual
        if self._model_name in {'ViT-B/32', 'ViT-B/16',
                                'ViT-L/14', 'ViT-L/14@336px'}:
            layers = [
                ('patch_embed', visual.conv1),
                ('ln_pre', visual.ln_pre),  # To streamline number of layers with CLIP.
                ('pos_embed', model_utils.ParamWrapperModule(visual.positional_embedding)),
                ('cls_token', model_utils.ParamWrapperModule(visual.class_embedding)),
            ]
            blocks = visual.transformer.resblocks
            for i, block in zip(range(len(blocks)), blocks):
                layers += [
                    ('trans' + str(i) + '_norm1', block.ln_1),
                    ('trans' + str(i) + '_attn', block.attn),
                    ('trans' + str(i) + '_norm2', block.ln_2),
                    ('trans' + str(i) + '_mlp', block.mlp),
                ]
            layers += [('post_norm', visual.ln_post)]
            layers += [('head', self.get_last_layer())]
        elif self._model_name in {'RN50'}:
            layers = [
                ('conv1', visual.conv1),
                ('bn1', visual.bn1),
                ('conv2', visual.conv2),
                ('bn2', visual.bn2),
                ('conv3', visual.conv3),
                ('bn3', visual.bn3),
                ('layer1', visual.layer1),
                ('layer2', visual.layer2),
                ('layer3', visual.layer3),
                ('attnpool', visual.attnpool)]
            if self._classifier is not None:
                layers = layers + [('head', self._classifier)]
        else:
            raise NotImplementedError
        return layers

    def get_num_trans_layers(self):
        visual = self._model.visual
        return len(visual.transformer.resblocks)

    def freeze_bottom_trans(self, num_trans_freeze, freeze_embed):
        num_trans_layers = self.get_num_trans_layers()
        if num_trans_freeze == num_trans_layers:
            # If we are tuning the head, don't tune the post layer norm.
            num_layers_freeze = num_trans_freeze * 4 + 5
        elif num_trans_freeze == 0:
            if freeze_embed:
                num_layers_freeze = 2
            else:
                num_layers_freeze = 0
        else:
            num_layers_freeze = num_trans_freeze * 4 + 4
        # Set all grads to True.
        set_requires_grad(self, True)
        # Call freeze.
        # TODO: Q: should we freeze pos embedding and cls token?
        self.freeze_bottom_k(num_layers_freeze)

    def tune_bottom_k(self, k):
        layers = [layer for name, layer in self.get_layers()]
        if k > len(layers):
            raise ValueError(f"k {k} should be less than number of layers {len(layers)}")
        set_requires_grad(self._model, False)
        for i in range(k):
            set_requires_grad(layers[i], True)
        # Also need to tune the prediction head because the fine-tuning task is different
        # from pretraining.
        set_requires_grad(layers[-1], True)

    def freeze_bottom_k(self, k):
        layers = [layer for name, layer in self.get_layers()]
        if k > len(layers):
            raise ValueError(f"k {k} should be less than number of layers {len(layers)}")
        for i in range(k):
            set_requires_grad(layers[i], False)

    def new_last_layer(self, num_classes):
        num_in_features = self._model.visual.output_dim
        self._classifier = nn.Linear(num_in_features, num_classes)
        self._classifier.to(self._device)

    def add_probe(self, probe):
        self._classifier = probe

    def get_last_layer(self):
        return self._classifier

    def set_last_layer(self, coef, intercept):
        model_utils.set_linear_layer(self._classifier, coef, intercept)

    def get_feature_extractor(self):
        raise NotImplementedError('Be careful, we need to normalize image first before encoding it.')

    def zero_shot_predict(self, images):
        if not os.path.exists('./imagenet-simple-labels.json'):
            os.system("wget -O imagenet-simple-labels.json https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json")

        with open("imagenet-simple-labels.json") as f:
            imagenet_labels = json.load(f)

        text_inputs = torch.cat([clip.tokenize(f"a photo of a {label}") for label in imagenet_labels]).to(images.device)
        image_logits, _ = self._model(images, text_inputs)
        return image_logits

    def get_features(self, x):
        return self._model.encode_image(normalize_transform(x))

def generate_new_state_dict_for_vit(state_dict, target_state_dict):
    new_state_dict = {}
    new_state_dict['embeddings.patch_embeddings.projection.weight'] = state_dict['conv1.weight']
    new_state_dict['embeddings.cls_token'] = state_dict['class_embedding'].unsqueeze(0).unsqueeze(0)
    new_state_dict['embeddings.position_embeddings'] = state_dict['positional_embedding'].unsqueeze(0)
    new_state_dict['layernorm_pre.weight'] = state_dict['ln_pre.weight']
    new_state_dict['layernorm_pre.bias'] = state_dict['ln_pre.bias']
    new_state_dict['layernorm.weight'] = state_dict['ln_post.weight']
    new_state_dict['layernorm.bias'] = state_dict['ln_post.bias']
    new_state_dict['proj'] = state_dict['proj']
    for old_key, value in state_dict.items():
        if "transformer.resblocks" in old_key:
            new_key = old_key.replace("transformer.resblocks", "encoder")
            if "ln_1" in new_key:
                new_key = new_key.replace("ln_1", "layernorm_before")
            if "ln_2" in new_key:
                new_key = new_key.replace("ln_2", "layernorm_after")
            if "attn" in new_key:
                if "in_proj_weight" in new_key:
                    hidden_size = int(value.shape[0] / 3)
                    value = value.view(3, hidden_size, -1)
                    q, k, v = value[0], value[1], value[2]
                    new_state_dict[new_key.replace('attn.in_proj_weight', 'attention.attention.query.weight')] = q
                    new_state_dict[new_key.replace('attn.in_proj_weight', 'attention.attention.key.weight')] = k
                    new_state_dict[new_key.replace('attn.in_proj_weight', 'attention.attention.value.weight')] = v
                    continue
                elif "in_proj_bias" in new_key:
                    hidden_size = int(value.shape[0] / 3)
                    value = value.view(3, hidden_size)
                    q, k, v = value[0], value[1], value[2]
                    new_state_dict[
                        new_key.replace('attn.in_proj_bias', 'attention.attention.query.bias')] = q
                    new_state_dict[new_key.replace('attn.in_proj_bias', 'attention.attention.key.bias')] = k
                    new_state_dict[
                        new_key.replace('attn.in_proj_bias', 'attention.attention.value.bias')] = v
                    continue
                elif "out_proj" in new_key:
                    new_key = new_key.replace("attn.out_proj", "attention.output.dense")
            elif "mlp" in new_key:
                if "c_fc" in new_key:
                    new_key = new_key.replace("mlp.c_fc", "intermediate.dense")
                else:
                    new_key = new_key.replace("mlp.c_proj", "output.dense")
            new_state_dict[new_key] = value
        else:
            # new_state_dict[old_key] = value
            continue
    for key, value in target_state_dict.items():
        if key in new_state_dict.keys():
            target_state_dict[key] = new_state_dict[key]
    return target_state_dict

class ClipDisentangleModel(nn.Module):

    def __init__(self, model_name, vit_config, include_qkv=True, scratch=False, checkpoint=None):
        # If scratch is True, then we randomly initialize weights.
        super().__init__()
        if model_name not in MODELS:
            raise ValueError(f'model_name must be in {MODELS} but was {model_name}')
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Note that model has both a language and vision part.
        if scratch:
            model = build_model_scratch(model_name, device=self._device)
        else:
            model, _ = clip.load(model_name, device=self._device)
        if checkpoint is not None:
            self._model_name = model_name
            self._model = model
            self.new_last_layer(1000)
            if checkpoint.endswith('.safetensors'):
                if include_qkv:
                    vit_disentangled = ViTModel(vit_config, with_embedding_nodes=False)
                else:
                    vit_disentangled = ViTModelSimple(vit_config, with_embedding_nodes=False)
                # state_dict = self._model.visual.state_dict()
                # target_state_dict = vit_disentangled.state_dict()
                # target_state_dict = generate_new_state_dict_for_vit(state_dict, target_state_dict)
                # vit_disentangled.load_state_dict(target_state_dict)
                self._model.visual = vit_disentangled
                self._model.visual.float()
                state_dict = load_file(checkpoint)
                self.load_state_dict(state_dict)
            elif 'linprobe_imagenet_clip_vit_b16' in checkpoint:
                with open(checkpoint, 'rb') as file:
                    (coef, intercept, best_c, best_i) = pickle.load(file)
                self.set_last_layer(coef, intercept)
                if include_qkv:
                    vit_disentangled = ViTModel(vit_config, with_embedding_nodes=False)
                else:
                    vit_disentangled = ViTModelSimple(vit_config, with_embedding_nodes=False)
                state_dict = self._model.visual.state_dict()
                target_state_dict = vit_disentangled.state_dict()
                target_state_dict = generate_new_state_dict_for_vit(state_dict, target_state_dict)
                vit_disentangled.load_state_dict(target_state_dict)
                self._model.visual = vit_disentangled
                self._model.visual.float()
            else:
                checkpoint = torch.load(checkpoint, map_location="cuda")
                self.load_state_dict(checkpoint['state_dict'])
                if include_qkv:
                    vit_disentangled = ViTModel(vit_config, with_embedding_nodes=False)
                else:
                    vit_disentangled = ViTModelSimple(vit_config, with_embedding_nodes=False)
                state_dict = self._model.visual.state_dict()
                target_state_dict = vit_disentangled.state_dict()
                target_state_dict = generate_new_state_dict_for_vit(state_dict, target_state_dict)
                vit_disentangled.load_state_dict(target_state_dict)
                self._model.visual = vit_disentangled
                self._model.visual.float()
        else:
            if include_qkv:
                vit_disentangled = ViTModel(vit_config, with_embedding_nodes=False)
            else:
                vit_disentangled = ViTModelSimple(vit_config, with_embedding_nodes=False)
            state_dict = model.visual.state_dict()
            target_state_dict = vit_disentangled.state_dict()
            target_state_dict = generate_new_state_dict_for_vit(state_dict, target_state_dict)
            vit_disentangled.load_state_dict(target_state_dict)
            model.visual = vit_disentangled
            self._model_name = model_name
            self._model = model
            self._model.visual.float()
            self._classifier = None

    def forward(self, input_images, labels=None, output_last_state=False, **kwargs):
        model_output = self.get_features(input_images, **kwargs)
        features = model_output.last_hidden_state
        loss = None
        if self._classifier is None:
            logits = None
        else:
            logits = self._classifier(features[:, 0, :])
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(logits.device)
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
        return CLIPHeadModelOutput(
            lm_loss=loss,
            logits=logits,
            past_key_values=model_output.past_key_values,
            hidden_states=model_output.hidden_states,
            last_hidden_state=model_output.last_hidden_state if output_last_state else None,
            attentions=model_output.attentions,
            writer_states=model_output.writer_states,
            target_edge_sparsity=model_output.target_edge_sparsity,
            target_node_sparsity=model_output.target_node_sparsity,
            model_edge_sparsity=model_output.model_edge_sparsity,
            model_node_sparsity=model_output.model_node_sparsity,
            edge_loss=model_output.edge_loss,
            node_loss=model_output.node_loss,
        )

    def set_requires_grad(self, val):
        for param in self._model.parameters():
            param.requires_grad = val
        if self._classifier is not None:
            for param in self._classifier.parameters():
                param.requires_grad = val

    def get_layers(self):
        visual = self._model.visual
        if self._model_name in {'ViT-B/32', 'ViT-B/16',
                                'ViT-L/14', 'ViT-L/14@336px'}:
            layers = [
                ('patch_embed', visual.conv1),
                ('ln_pre', visual.ln_pre),  # To streamline number of layers with CLIP.
                ('pos_embed', model_utils.ParamWrapperModule(visual.positional_embedding)),
                ('cls_token', model_utils.ParamWrapperModule(visual.class_embedding)),
            ]
            blocks = visual.transformer.resblocks
            for i, block in zip(range(len(blocks)), blocks):
                layers += [
                    ('trans' + str(i) + '_norm1', block.ln_1),
                    ('trans' + str(i) + '_attn', block.attn),
                    ('trans' + str(i) + '_norm2', block.ln_2),
                    ('trans' + str(i) + '_mlp', block.mlp),
                ]
            layers += [('post_norm', visual.ln_post)]
            layers += [('head', self.get_last_layer())]
        elif self._model_name in {'RN50'}:
            layers = [
                ('conv1', visual.conv1),
                ('bn1', visual.bn1),
                ('conv2', visual.conv2),
                ('bn2', visual.bn2),
                ('conv3', visual.conv3),
                ('bn3', visual.bn3),
                ('layer1', visual.layer1),
                ('layer2', visual.layer2),
                ('layer3', visual.layer3),
                ('attnpool', visual.attnpool)]
            if self._classifier is not None:
                layers = layers + [('head', self._classifier)]
        else:
            raise NotImplementedError
        return layers

    def get_num_trans_layers(self):
        visual = self._model.visual
        return len(visual.transformer.resblocks)

    def freeze_bottom_trans(self, num_trans_freeze, freeze_embed):
        num_trans_layers = self.get_num_trans_layers()
        if num_trans_freeze == num_trans_layers:
            # If we are tuning the head, don't tune the post layer norm.
            num_layers_freeze = num_trans_freeze * 4 + 5
        elif num_trans_freeze == 0:
            if freeze_embed:
                num_layers_freeze = 2
            else:
                num_layers_freeze = 0
        else:
            num_layers_freeze = num_trans_freeze * 4 + 4
        # Set all grads to True.
        set_requires_grad(self, True)
        # Call freeze.
        # TODO: Q: should we freeze pos embedding and cls token?
        self.freeze_bottom_k(num_layers_freeze)

    def tune_bottom_k(self, k):
        layers = [layer for name, layer in self.get_layers()]
        if k > len(layers):
            raise ValueError(f"k {k} should be less than number of layers {len(layers)}")
        set_requires_grad(self._model, False)
        for i in range(k):
            set_requires_grad(layers[i], True)
        # Also need to tune the prediction head because the fine-tuning task is different
        # from pretraining.
        set_requires_grad(layers[-1], True)

    def freeze_bottom_k(self, k):
        layers = [layer for name, layer in self.get_layers()]
        if k > len(layers):
            raise ValueError(f"k {k} should be less than number of layers {len(layers)}")
        for i in range(k):
            set_requires_grad(layers[i], False)

    def new_last_layer(self, num_classes):
        num_in_features = self._model.visual.output_dim
        self._classifier = nn.Linear(num_in_features, num_classes)
        self._classifier.to(self._device)

    def add_probe(self, probe):
        self._classifier = probe

    def get_last_layer(self):
        return self._classifier

    def set_last_layer(self, coef, intercept):
        model_utils.set_linear_layer(self._classifier, coef, intercept)

    def get_feature_extractor(self):
        raise NotImplementedError('Be careful, we need to normalize image first before encoding it.')

    def get_features(self, x, **kwargs):
        model_output = self._model.visual(normalize_transform(x).type(torch.float32), **kwargs)
        return model_output

def test_equal():
    set_seed(1)
    imagenet_val = ImageNetDataset(root_dir='/data/nvme1/yxpeng/imagenet/val',
                                   processor=AutoImageProcessor.from_pretrained("google/vit-base-patch16-224"))

    # Create DataLoader
    device = 'cuda:1'
    batch_size = 16  # Adjust as needed
    val_loader = DataLoader(imagenet_val, batch_size=batch_size, shuffle=False)
    origin_model = ClipModel('ViT-B/16').to(device).eval()
    vit_config = ViTConfig()
    vit_config.embedding_bias = False
    vit_config.layernorm_pre = True
    vit_config.layer_norm_eps = 1e-5
    vit_config.proj = True
    vit_config.hidden_act = 'quick_gelu'
    disentangled_model = ClipDisentangleModel('ViT-B/16', vit_config).to(device).eval()
    disentangled_model._model.visual.set_edge_threshold_for_deterministic(0.5)
    disentangled_model._model.visual.set_node_threshold_for_deterministic(0.5)

    with torch.no_grad():
        i = 0
        for batch in tqdm(val_loader):
            images, labels = batch  # Assuming dataset returns (image, label)
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions from both models
            origin_outputs = origin_model(images)
            disentangled_outputs = disentangled_model(images, labels).last_hidden_state

            difference = origin_outputs - disentangled_outputs[:,0]
            mean = abs(difference).mean()
            if i % 20 == 0:
                print(f'align rate: {mean}')

def compute_class_mean():
    set_seed(1)
    transform = transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=224),
        transforms.ToTensor()
    ])
    vit_config = ViTConfig()
    vit_config.embedding_bias = False
    vit_config.layernorm_pre = True
    vit_config.layer_norm_eps = 1e-5
    vit_config.proj = True
    vit_config.hidden_act = 'quick_gelu'
    ft_method = 'LP'
    dataset_split = 'val'
    if ft_method == 'FT':
        clip_checkpoint = "/data/nvme1/yxpeng/PycharmProjects/transfer_learning/logs/full_ft_imagenet_clip_vit_b16/optimizer.args.lr-0.001_with_augment_seed-0_run0/checkpoints/ckp_best_val"
    elif ft_method == 'LP':
        clip_checkpoint = "/data/nvme1/yxpeng/PycharmProjects/transfer_learning/logs/linprobe_imagenet_clip_vit_b16/weights_0.pkl"
    gpt2_model = ClipDisentangleModel('ViT-B/16', vit_config, include_qkv=False, checkpoint=clip_checkpoint)
    gpt2_model.to('cuda:6').eval()
    gpt2_model._model.visual.set_edge_threshold_for_deterministic(0.5)
    gpt2_model._model.visual.set_node_threshold_for_deterministic(0.5)
    batch_size = 20
    with torch.no_grad():
        activation_dict = {}
        for i in range(1000):
            imagenet_val = ImageNetDataset(root_dir=f'/data/nvme1/yxpeng/imagenet/{dataset_split}',
                                           transform=transform, select_class=i)
            val_loader = DataLoader(imagenet_val, batch_size=batch_size, shuffle=False, num_workers=12)
            activation = None
            print(f'Generating activations for class {i}')
            for batch in tqdm(val_loader):
                images, labels = batch  # Assuming dataset returns (image, label)
                images = images.to("cuda:6")
                labels = labels.to("cuda:6")
                output_batch = gpt2_model(input_images=images, labels=labels, output_writer_states=True)
                activation_batch = output_batch.writer_states
                if activation == None:
                    activation = activation_batch.sum(dim=1)
                else:
                    activation += activation_batch.sum(dim=1)
            activation_dict[f'class_{i}'] = {'activations': activation.detach().cpu(), 'size': len(imagenet_val)}
    with open(f"activations/sum_{ft_method}_clip_imagenet_{dataset_split}_class_split.pkl", "wb") as f:  # "wb" = write binary
        pickle.dump(activation_dict, f)

def accelerate_compute_class_mean():
    accelerator = Accelerator()
    set_seed(1)

    transform = transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=224),
        transforms.ToTensor()
    ])

    vit_config = ViTConfig()
    vit_config.embedding_bias = False
    vit_config.layernorm_pre = True
    vit_config.layer_norm_eps = 1e-5
    vit_config.proj = True
    vit_config.hidden_act = 'quick_gelu'

    ft_method = 'FT'
    dataset_split = 'train'

    if ft_method == 'FT':
        clip_checkpoint = "/data/nvme1/yxpeng/PycharmProjects/transfer_learning/logs/full_ft_imagenet_clip_vit_b16/optimizer.args.lr-0.001_with_augment_seed-0_run0/checkpoints/ckp_best_val"
    elif ft_method == 'LP':
        clip_checkpoint = "/data/nvme1/yxpeng/PycharmProjects/transfer_learning/logs/linprobe_imagenet_clip_vit_b16/weights_0.pkl"

    gpt2_model = ClipDisentangleModel('ViT-B/16', vit_config, include_qkv=False, checkpoint=clip_checkpoint)
    gpt2_model.eval()

    gpt2_model._model.visual.set_edge_threshold_for_deterministic(0.5)
    gpt2_model._model.visual.set_node_threshold_for_deterministic(0.5)
    gpt2_model = accelerator.prepare(gpt2_model)

    batch_size = 32

    with torch.no_grad():
        activation_dict = {}

        for i in range(1000):
            imagenet_val = ImageNetDataset(root_dir=f'/data/nvme1/yxpeng/imagenet/{dataset_split}',
                                           transform=transform, select_class=i)
            val_loader = DataLoader(imagenet_val, batch_size=batch_size, shuffle=False)
            val_loader = accelerator.prepare(val_loader)

            activation = None
            if accelerator.is_main_process:
                print(f'Generating activations for class {i}, number of samples: {len(imagenet_val)}')

            val_loader = tqdm(val_loader) if accelerator.is_main_process else val_loader
            for batch in val_loader:
                images, labels = batch  # Assuming dataset returns (image, label)
                images, labels = images.to(accelerator.device), labels.to(accelerator.device)

                output_batch = gpt2_model(input_images=images, labels=labels, output_writer_states=True)
                activation_batch = output_batch.writer_states

                if activation is None:
                    activation = activation_batch.sum(dim=1)
                else:
                    activation += activation_batch.sum(dim=1)
            activation = activation.unsqueeze(0) # since gather concate the first dim
            activation = accelerator.gather(activation)
            activation = activation.sum(dim=0)
            activation_dict[f'class_{i}'] = {'activations': activation.detach().cpu(), 'size': len(imagenet_val)}
            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        with open(f"activations/sum_{ft_method}_clip_imagenet_{dataset_split}_class_split.pkl", "wb") as f:
            pickle.dump(activation_dict, f)

def combine_classes():
    activation_path = '/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/activations/sum_FT_clip_imagenet_train_class_split.pkl'
    with open(activation_path, "rb") as f:  # "rb" = read binary
        loaded_data = pickle.load(f)

    sum_activation = torch.zeros((156,197,768))
    num_samples = 0
    for key, value in loaded_data.items():
        sum_activation += value['activations']
        num_samples += value['size']

    mean_activation = sum_activation / num_samples
    with open(f"activations/mean_FT_clip_imagenet_train.pkl", "wb") as f:
        pickle.dump(mean_activation, f)


def compute_mean():
    set_seed(1)
    transform = transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=224),
        transforms.ToTensor()
    ])
    imagenet_val = ImageNetDataset(root_dir='/data/nvme1/yxpeng/imagenet/val',
                                   transform=transform)

    # Create DataLoader
    batch_size = 16  # Adjust as needed
    val_loader = DataLoader(imagenet_val, batch_size=batch_size, shuffle=False)

    vit_config = ViTConfig()
    vit_config.embedding_bias = False
    vit_config.layernorm_pre = True
    vit_config.layer_norm_eps = 1e-5
    vit_config.proj = True
    vit_config.hidden_act = 'quick_gelu'
    clip_checkpoint = "/data/nvme1/yxpeng/PycharmProjects/transfer_learning/logs/full_ft_imagenet_clip_vit_b16/optimizer.args.lr-0.001_with_augment_seed-0_run0/checkpoints/ckp_best_val"
    clip_state_dict = torch.load(clip_checkpoint, map_location="cuda")['state_dict']
    origin_model = ClipModel('ViT-B/16')
    origin_model.new_last_layer(1000)
    origin_model.load_state_dict(clip_state_dict)
    origin_model.to('cuda').eval()
    gpt2_model = ClipDisentangleModel('ViT-B/16', vit_config, checkpoint=clip_checkpoint)
    gpt2_model.to('cuda').eval()
    gpt2_model._model.visual.set_edge_threshold_for_deterministic(0.5)
    gpt2_model._model.visual.set_node_threshold_for_deterministic(0.5)
    activation = None
    match = 0
    with torch.no_grad():
        i = 0
        for batch in tqdm(val_loader):
            images, labels = batch  # Assuming dataset returns (image, label)
            images = images.to("cuda")
            labels = labels.to("cuda")
            origin_outputs, origin_features = origin_model(images, output_feature=True)
            output_batch = gpt2_model(input_images=images, labels=labels, output_writer_states=True)
            output_activation = output_batch.last_hidden_state
            activation_batch = output_batch.writer_states
            activation_first_token = output_activation[:,0,:]
            mean_diff = abs(activation_first_token - origin_features).mean()
            origin_preds = torch.argmax(origin_outputs, dim=-1)
            disentangle_preds = torch.argmax(output_batch.logits, dim=-1)
            match += (origin_preds == disentangle_preds).sum().item()
            if activation == None:
                activation = activation_batch.sum(dim=1)
            else:
                activation += activation_batch.sum(dim=1)
            i += batch_size
            match_rate = match / i
            print(f'match:{match_rate}')
            print(f'mean diff:{mean_diff}')
    activation = activation / i
    import pickle
    with open("mean_clip_imagenet_val.pkl", "wb") as f:  # "wb" = write binary
        pickle.dump(activation.detach().cpu(), f)

if __name__ == '__main__':
    # test_equal()
    # compute_mean()
    # compute_class_mean()
    # accelerate_compute_class_mean()
    combine_classes()