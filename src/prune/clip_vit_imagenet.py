#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import sys, os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import logging
import torch
import pickle
import random
import sys
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union
from torch.utils.data import random_split, DataLoader, TensorDataset
import datasets
import evaluate
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from torchvision.transforms import transforms
from transformers import AutoImageProcessor, ViTForImageClassification, ViTConfig

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import torch.nn as nn
from torch.optim import AdamW



sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/"
    )
)   # Very hacky but the imports are annoying otherwise

from utils import get_failure_list
from custom_datasets import WaterbirdDataset
from utils import ImageNet
from utils import bernoulli_kl
from custom_datasets import ColoredMNIST
from modeling.clip_model import ClipModel, ClipDisentangleModel
from modeling.vit import ViTHeadModel
from utils import ImageNetDataset

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
logger = logging.getLogger(__name__)

from transformers import Trainer
import torch
import torch.nn as nn

from transformers import TrainerCallback

class ImageClassifierTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.target_edge_sparsity = kwargs.pop('target_edge_sparsity', 0.0)
        self.start_edge_sparsity = kwargs.pop('start_edge_sparsity', 0.0)
        self.target_layer_sparsity = kwargs.pop('target_layer_sparsity', 0.0)
        self.start_layer_sparsity = kwargs.pop('start_layer_sparsity', 0.0)
        if "num_edge_sparsity_warmup_steps" in kwargs:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_edge_sparsity_warmup_steps')
        else:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', 0)
        if "num_layer_sparsity_warmup_steps" in kwargs:
            self.num_layer_sparsity_warmup_steps = kwargs.pop('num_layer_sparsity_warmup_steps')
        else:
            self.num_layer_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps',
                                                              self.num_edge_sparsity_warmup_steps)
        _ = kwargs.pop('num_sparsity_warmup_steps', None)
        self.warmup_type = kwargs.pop('warmup_type', 'linear')
        self.gpt2_model = kwargs.pop('gpt2_model', None)
        self.skip_layer_loss_if_higher_sparsity = kwargs.pop('skip_layer_loss_if_higher_sparsity', False)
        self.loss_type = kwargs.pop('loss_type', 'kl')
        self.target_class = kwargs.pop('target_class', None)
        self.alpha = kwargs.pop('alpha', 1.0)

        self.digits = None
        self.device_count = torch.cuda.device_count()

        super().__init__(*args, **kwargs)

    def get_current_edge_target_sparsity(self, global_step):
        if global_step < self.num_edge_sparsity_warmup_steps:
            if self.warmup_type == 'linear':
                return (
                        self.start_edge_sparsity + (self.target_edge_sparsity - self.start_edge_sparsity) *
                        global_step / self.num_edge_sparsity_warmup_steps
                )
            elif self.warmup_type == 'logarithmic':
                log_one_minus_sparsity = math.log(1 - self.start_edge_sparsity) + (
                            math.log(1 - self.target_edge_sparsity) -
                            math.log(1 - self.start_edge_sparsity)) * global_step / self.num_edge_sparsity_warmup_steps
                return 1 - math.exp(log_one_minus_sparsity)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_edge_sparsity

    def get_current_layer_target_sparsity(self, global_step):
        if global_step < self.num_layer_sparsity_warmup_steps:
            if self.warmup_type == 'linear':
                return (
                        self.start_layer_sparsity + (self.target_layer_sparsity - self.start_layer_sparsity) *
                        global_step / self.num_layer_sparsity_warmup_steps
                )
            elif self.warmup_type == 'logarithmic':
                log_one_minus_sparsity = math.log(1 - self.start_layer_sparsity) + (
                            math.log(1 - self.target_layer_sparsity) -
                            math.log(
                                1 - self.start_layer_sparsity)) * global_step / self.num_layer_sparsity_warmup_steps
                return 1 - math.exp(log_one_minus_sparsity)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_layer_sparsity

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if inputs['corr_inputs'] is not None:
            corr_images = inputs.pop("corr_inputs")
        else:
            inputs.pop("corr_inputs")
            corr_images = 'zero_ablate'
        clean_images = inputs.pop("inputs")

        bsz = clean_images.shape[0]

        with torch.no_grad():
            # First get the logits from the GPT-2 model
            if not self.loss_type == 'target':
                gpt2_output = self.gpt2_model(input_images=clean_images, **inputs)
                gpt2_logits = gpt2_output.logits
                gpt2_logits = torch.nn.functional.log_softmax(gpt2_logits, dim=-1) if gpt2_logits.shape[-1] > 1 else gpt2_logits
            # gpt2_activation = gpt2_output.writer_states

            # Now run the corrupted inputs through it, and retain the activations
            if corr_images == 'zero_ablate':
                #TODO this is hard coded
                corr_x = torch.zeros((156, bsz, 197, 768))
            elif len(corr_images.shape) == 3:
                corr_x = corr_images.unsqueeze(1).repeat(1, bsz, 1, 1)
            else:
                corr_x = self.gpt2_model(input_images=corr_images, **inputs, output_writer_states=True).writer_states

            # Reshape corr_x in case we have distributed training, comment in case of accelerate
            # tgt_shape = (-1, bsz // self.device_count, *corr_x.shape[2:])
            # corr_x = corr_x.reshape(tgt_shape)

        outputs = model(
            input_images=clean_images,
            **inputs,
            target_edge_sparsity=self.get_current_edge_target_sparsity(self.state.global_step),
            target_node_sparsity=self.get_current_layer_target_sparsity(self.state.global_step),
            corr_x=corr_x
        )

        reg_edge_loss = outputs["edge_loss"]
        if self.skip_layer_loss_if_higher_sparsity and outputs["model_node_sparsity"] > outputs["target_node_sparsity"]:
            reg_layer_loss = 0
        else:
            reg_layer_loss = outputs["node_loss"]
        reg_loss = reg_edge_loss + reg_layer_loss

        ## Restricting to 01-99 for now
        # Only the last position
        logits = outputs.logits
        logits = torch.nn.functional.log_softmax(logits, dim=-1) if logits.shape[-1] > 1 else logits

        if self.loss_type == 'partial_kl':
            assert logits.shape[-1] > 1, "the model only output one class, partial kl is not supported"
            if isinstance(self.target_class, list):
                target_classes = torch.tensor(self.target_class)
                circuit_loss = nn.functional.kl_div(logits[:, target_classes], gpt2_logits[:, target_classes], reduction="batchmean", log_target=True)
            else:
                circuit_loss = nn.functional.kl_div(logits[:, self.target_class], gpt2_logits[:, self.target_class], reduction="batchmean", log_target=True)
        elif self.loss_type == 'kl':
            if logits.shape[-1] > 1:
                circuit_loss = nn.functional.kl_div(logits, gpt2_logits, reduction="batchmean", log_target=True)
            else:
                circuit_loss = bernoulli_kl(logits, gpt2_logits, reduction="batchmean")
        elif self.loss_type == 'target':
            circuit_loss = outputs.lm_loss
        elif self.loss_type == 'logit_difference':
            chosen_class = torch.argmax(gpt2_logits, dim=-1)
            circuit_loss = ((gpt2_logits[:, chosen_class] - logits[:, chosen_class]) ** 2).mean()
        else:
            raise NotImplementedError
        loss = self.alpha * circuit_loss + reg_loss
        outputs["loss"] = loss
        outputs["circuit_loss"] = circuit_loss
        outputs["prob_digits"] = torch.nn.functional.softmax(logits, dim=-1) if logits.shape[-1] > 1 else torch.sigmoid(logits)
        self.log({
                "train/circuit_loss": circuit_loss.item(),
                "train/reg_edge_loss": reg_edge_loss.mean().item(),
            })

        return (loss, outputs) if return_outputs else loss


@dataclass
class DataTrainingArguments:
    dataset_path: Optional[str] = field(
        default="./data/datasets/gt/",
        metadata={"help": "The path to the directory with the JSON files of the task."},
    )
    train_split: Optional[str] = field(
        default="train",
        metadata={"help": "The split to use for training."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=64,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    start_edge_sparsity: Optional[float] = field(
        default=0.0,
        metadata={"help": "The initial edge sparsity of the model."}
    )
    target_edge_sparsity: Optional[float] = field(
        default=0.98,
        metadata={"help": "The target edge sparsity of the model."}
    )
    start_layer_sparsity: Optional[float] = field(
        default=0.0,
        metadata={"help": "The initial layer sparsity of the model."}
    )
    target_layer_sparsity: Optional[float] = field(
        default=0.68,
        metadata={"help": "The target layer sparsity of the model."}
    )
    stop_optimizing_layer_if_higher_sparsity: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to stop optimizing the layer sparsity if it is higher than the target."}
    )
    num_sparsity_warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": "The number of steps to reach the target sparsity."}
    )
    edge_learning_rate: Optional[float] = field(
        default=1e-2,
        metadata={"help": "The learning rate for the regularization term."}
    )
    layer_learning_rate: Optional[float] = field(
        default=1,
        metadata={"help": "The learning rate for the regularization term."}
    )
    reg_edge_learning_rate: Optional[float] = field(
        default=1e-2,
        metadata={"help": "The learning rate for the regularization term."}
    )
    reg_layer_learning_rate: Optional[float] = field(
        default=1,
        metadata={"help": "The learning rate for the regularization term."}
    )
    warmup_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The type of warmup to use for the regularization term."}
    )
    with_embedding_nodes: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to include the embedding nodes"}
    )
    disable_linear_reg_term: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable the linear regularization term."}
    )
    disable_node_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable node loss."}
    )
    corr_mode: Optional[str] = field(
        default="noise",
        metadata={"help": "The mode of corrupted input."}
    )
    loss_type: Optional[str] = field(
        default="kl",
        metadata={"help": "The type of loss"}
    )
    dataset: Optional[str] = field(
        default="imagenet",
        metadata={"help": "The type of loss"}
    )
    ood_dataset: Optional[str] = field(
        default="v2",
        metadata={"help": "The type of loss"}
    )
    alpha: Optional[float] = field(
        default=1.0,
        metadata={"help": "The type of loss"}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    initialize_from: str = field(
        default="gpt2",
        metadata={"help": "The model to initialize from."},
    )
    include_qkv: bool = field(
        default=True,
        metadata={"help": "Include qkv edges or not."},
    )
    ft_method: str = field(
        default="FT",
        metadata={"help": "The interpreted model"},
    )
    target_class: str = field(
        default='0',
        metadata={"help": "target class of the circuit"},
    )
    ckpt_id: str = field(
        default='0',
        metadata={"help": "target class of the circuit"},
    )

    def __post_init__(self):
        # Convert purely numeric string inputs to integers, keep WNIDs as strings
        if self.target_class.isdigit():
            self.target_class = int(self.target_class)

def format_instance(instance, split):
    if isinstance(instance, dict) and "min_steps" in instance:
        return {
            "tokens": instance["tokens"],
            "split": split,
            "min_steps": instance["min_steps"],
        }
    else:
        return {
            "tokens": instance,
            "split": split,
        }

def load_datasets(dataset_path, max_train_samples, max_eval_samples, train_split="train"):
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(dataset_path)
    if "validation" not in dataset:
        assert max_eval_samples is not None, "Validation set is missing! (val)"
        assert max_train_samples is not None, "Validation set is missing! (train)"
        dataset = DatasetDict({
            train_split: dataset[train_split].select(range(max_train_samples)),
            "validation": dataset[train_split].select(range(max_train_samples, max_train_samples+max_eval_samples)),
        })
    else:
        if max_train_samples is not None and max_train_samples < len(dataset[train_split]):
            dataset[train_split] = dataset[train_split].select(range(max_train_samples))
        if max_eval_samples is not None and max_eval_samples < len(dataset["validation"]):
            dataset["validation"] = dataset["validation"].select(range(max_eval_samples))
    return dataset



class DataCollator:
    def __init__(self, mode, loaded_activation=None, ood_dataset=None):
        self.mode = mode
        self.loaded_activation = loaded_activation
        self.ood_dataset = ood_dataset
        self.ood_class_index = self._build_class_index(ood_dataset) if ood_dataset else None

    def _build_class_index(self, dataset):
        class_index = {}
        if hasattr(dataset, 'target_name'):
            for idx, label in enumerate(getattr(dataset, dataset.target_name)):  # or dataset.labels
                label = int(label)
                if label not in class_index:
                    class_index[label] = []
                class_index[label].append(idx)
        else:
            for idx, data in enumerate(dataset):
                if len(data) == 2:
                    image, label = data
                    _ = None
                elif len(data) == 3:
                    image, label, _ = data
                else:
                    raise ValueError("Unexpected number of items returned by dataset")
                label = int(label)  # Ensure consistent key type
                if label not in class_index:
                    class_index[label] = []
                class_index[label].append(idx)
        return class_index

    def __call__(self, examples):
        inputs = []
        corr_inputs = []
        labels = []         # need to pass something otherwise compute_metrics will not be called

        if self.mode == 'noise':
            for i, example in enumerate(examples):
                input = example[0]
                corr_image = torch.clamp(input + torch.randn_like(input) * 0.3, -1, 1)
                inputs.append(input)
                corr_inputs.append(corr_image)
                labels.append(example[1])
            return {
                "inputs": torch.stack(inputs),
                "corr_inputs": torch.stack(corr_inputs),
                "labels": torch.tensor(labels),
            }
        elif self.mode == 'mean':
            for i, example in enumerate(examples):
                input = example[0]
                inputs.append(input)
                labels.append(example[1])
            return {
                "inputs": torch.stack(inputs),
                "corr_inputs": self.loaded_activation,
                "labels": torch.tensor(labels),
            }
        elif self.mode == 'zero':
            for i, example in enumerate(examples):
                input = example[0]
                inputs.append(input)
                labels.append(example[1])
            return {
                "inputs": torch.stack(inputs),
                "corr_inputs": None,
                "labels": torch.tensor(labels),
            }
        elif self.mode == 'color':
            for i, example in enumerate(examples):
                input = example[0]
                corr_image = input[[1, 0, 2]]
                inputs.append(input)
                corr_inputs.append(corr_image)
                labels.append(example[1])
            return {
                "inputs": torch.stack(inputs),
                "corr_inputs": torch.stack(corr_inputs),
                "labels": torch.tensor(labels),
            }
        elif self.mode == 'ood' or self.mode == 'ood_failure':
            for i, example in enumerate(examples):
                input = example[0]
                label = example[1]

                corr_image = self.ood_dataset[random.choice(self.ood_class_index[label])][0]
                corr_inputs.append(corr_image)
                inputs.append(input)
                labels.append(label)
            return {
                "inputs": torch.stack(inputs),
                "corr_inputs": torch.stack(corr_inputs),
                "labels": torch.tensor(labels),
            }

def eval_fn(eval_pred):         
    (
        _, logits, target_edge_sparsity, target_layer_sparsity, model_edge_sparsity, model_layer_sparsity, reg_edge_loss, reg_layer_loss,
        circuit_loss, prob_digits
    ) = eval_pred.predictions
    labels = eval_pred.label_ids
    
    if len(model_edge_sparsity.shape) > 0:
        model_edge_sparsity = model_edge_sparsity[0].item()
        model_layer_sparsity = model_layer_sparsity[0].item()
        target_edge_sparsity = target_edge_sparsity[0].item()
        target_layer_sparsity = target_layer_sparsity[0].item()
    else:
        model_edge_sparsity = model_edge_sparsity.item()
        model_layer_sparsity = model_layer_sparsity.item()
        target_edge_sparsity = target_edge_sparsity.item()
        target_layer_sparsity = target_layer_sparsity.item()
    
    predictions = np.argmax(logits, axis=-1) if logits.shape[-1] > 1 else (logits > 0).astype(int)

    correct = (predictions == labels).sum()
    accuracy = correct.item() / labels.shape[0]
    
    circuit_loss = circuit_loss.mean().item()
    reg_edge_loss = reg_edge_loss.mean().item()
    reg_layer_loss = reg_layer_loss.mean().item()
    
    return {
        "eval_accuracy": accuracy,
        "model_edge_sparsity": model_edge_sparsity,
        "model_layer_sparsity": model_layer_sparsity,
        "target_edge_sparsity": target_edge_sparsity,
        "target_layer_sparsity": target_layer_sparsity,
        "eval_circuit_loss": circuit_loss,
        "eval_reg_edge_loss": reg_edge_loss,
        "eval_reg_layer_loss": reg_layer_loss,
    }
    
def freeze_all_except_pruning_params(model):
    for n, p in model.named_parameters():
        if 'log_alpha' in n or 'sparsity_lambda' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

def get_optimizers(model, edges_lr, layers_lr, reg_edges_lr, reg_layers_lr, num_training_steps, warmup_steps=0, disable_node_loss=False):
    optimizer_1_group = []
    optimizer_2_group = []
    optimizer_3_group = []
    optimizer_4_group = []

    for n, p in model.named_parameters():
        if 'write_log_alpha' in n:
            optimizer_3_group.append(p)
        elif 'read_log_alpha' in n:
            optimizer_1_group.append(p)
        elif 'sparsity_lambda_edge' in n:
            optimizer_2_group.append(p)
        elif ('sparsity_lambda_node' in n) and (not disable_node_loss):
            optimizer_4_group.append(p)
    
    optimizer = AdamW(
        [
            {
                'params': optimizer_1_group,
                'lr': edges_lr,
            },
            {
                'params': optimizer_2_group,
                'maximize': True,
                'lr': reg_edges_lr,
            },
            {
                'params': optimizer_3_group,
                'lr': layers_lr,
            },
            {
                'params': optimizer_4_group,
                'maximize': True,
                'lr': reg_layers_lr,
            } 
        ],
        lr=edges_lr
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    return optimizer, scheduler

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
    in_path = '/data/nvme1/yxpeng/imagenet/'
    in_info_path = '/data/nvme1/yxpeng/imagenet'
    in_hier = ImageNetHierarchy(in_path, in_info_path)
    if isinstance(model_args.target_class, str) and data_args.dataset == 'imagenet':
        class_ranges, label_map = in_hier.get_subclasses([model_args.target_class],
                                                         balanced=False)
        class_ranges = list(class_ranges[0])
    else:
        class_ranges = model_args.target_class

    ood_dataset = None
    if data_args.dataset == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=224),
            transforms.ToTensor()
        ])
        if model_args.ft_method == 'google':
            dataset = ImageNetDataset(root_dir='/data/nvme1/yxpeng/imagenet/train',
                                               processor=AutoImageProcessor.from_pretrained("google/vit-base-patch16-224"), select_class=class_ranges)
            ood_dataset = ImageNet(root='/data/nvme1/yxpeng/imagenet', split=data_args.ood_dataset, processor=AutoImageProcessor.from_pretrained("google/vit-base-patch16-224"))
        else:
            dataset = ImageNetDataset(root_dir='/data/nvme1/yxpeng/imagenet/train',
                                               transform=transform, select_class=class_ranges)
            ood_dataset = ImageNet(root='/data/nvme1/yxpeng/imagenet', split=data_args.ood_dataset, transform=transform)
    elif data_args.dataset == 'waterbirds':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = WaterbirdDataset(data_correlation=0.95, split="train", root_dir="/data/nvme1/yxpeng/PycharmProjects/vit-spurious-robustness/datasets", transform=transform)
        ood_dataset = WaterbirdDataset(data_correlation=0.95, split="val", root_dir="/data/nvme1/yxpeng/PycharmProjects/vit-spurious-robustness/datasets", transform=transform)
    elif data_args.dataset == 'colored_mnist_all_train':
        dataset = ColoredMNIST(root='/data/nvme1/yxpeng/PycharmProjects/pyvenv-experiments/vision-grokking/new_data', env='all_train', select_class=model_args.target_class,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                                     ]))
        ood_dataset = ColoredMNIST(root='/data/nvme1/yxpeng/PycharmProjects/pyvenv-experiments/vision-grokking/new_data',
                               env='test', select_class='all',
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                               ]))
    elif data_args.dataset == 'colored_mnist_all_train_unbiased':
        dataset = ColoredMNIST(root='/data/nvme1/yxpeng/PycharmProjects/pyvenv-experiments/vision-grokking/new_data', env='all_train_unbiased', select_class=model_args.target_class,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                                     ]))
        ood_dataset = ColoredMNIST(root='/data/nvme1/yxpeng/PycharmProjects/pyvenv-experiments/vision-grokking/new_data',
                               env='test', select_class='all',
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                               ]))
    else:
        ood_dataset = None
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    raw_datasets = {'train': train_dataset, 'validation': eval_dataset}

    if model_args.ft_method == 'google':
        vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        state_dict = vit_model.state_dict()
        new_state_dict = {}
        for old_key, value in state_dict.items():
            if "vit.encoder.layer" in old_key:
                new_key = old_key.replace("vit.encoder.layer", "vit.encoder")
                new_state_dict[new_key] = value
            else:
                new_state_dict[old_key] = value
        model = ViTHeadModel.from_pretrained(
            'google/vit-base-patch16-224',
            state_dict=new_state_dict,
            include_qkv=model_args.include_qkv,
            with_embedding_nodes=data_args.with_embedding_nodes,
            disable_linear_regularization_term=data_args.disable_linear_reg_term,
        ).eval()
        gpt2_model = ViTHeadModel.from_pretrained(
            'google/vit-base-patch16-224',
            include_qkv=model_args.include_qkv,
            state_dict=new_state_dict,
            with_embedding_nodes=data_args.with_embedding_nodes,
        ).to("cuda").eval()
    elif model_args.ft_method == 'IN21k-ERM-WaterBird':
        ckpt = torch.load(f'/data/nvme1/yxpeng/PycharmProjects/vit-spurious-robustness/output/waterbirds_exp/waterbirds/ViT/ViT-B_16{model_args.ckpt_id}.bin')
        new_state_dict = {}
        for old_key, value in ckpt.items():
            if "vit.encoder.layer" in old_key:
                new_key = old_key.replace("vit.encoder.layer", "vit.encoder")
                new_state_dict[new_key] = value
            else:
                new_state_dict[old_key] = value
        model = ViTHeadModel.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            state_dict=new_state_dict,
            include_qkv=model_args.include_qkv,
            with_embedding_nodes=data_args.with_embedding_nodes,
            disable_linear_regularization_term=data_args.disable_linear_reg_term,
        ).eval()
        gpt2_model = ViTHeadModel.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            include_qkv=model_args.include_qkv,
            state_dict=new_state_dict,
            with_embedding_nodes=data_args.with_embedding_nodes,
        ).to("cuda").eval()
        if data_args.corr_mode == 'ood_failure':
            failure_list = get_failure_list(f'/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/failure/IN21k-ERM-WaterBird-{model_args.ckpt_id}.p', ood_dataset, gpt2_model)
            ood_dataset.filter_from_list(failure_list)
    elif model_args.ft_method == 'ERM' or model_args.ft_method == 'IRM' or model_args.ft_method == 'DRO':
        vit_config = ViTConfig(image_size=28, patch_size=7, num_hidden_layers=2, num_attention_heads=4,
                           intermediate_size=256 * 3,
                           num_channels=3, num_labels=1)
        if model_args.ft_method == 'ERM':
            vit_ckpt = f'/data/nvme1/yxpeng/PycharmProjects/pyvenv-experiments/vision-grokking/checkpoints/vit_erm/ViT_coloredmnist_erm_test_{model_args.ckpt_id}.pt'
        elif model_args.ft_method == 'IRM':
            vit_ckpt = f'/data/nvme1/yxpeng/PycharmProjects/pyvenv-experiments/vision-grokking/checkpoints/vit_irm_new/ViT_coloredmnist_irm_test_{model_args.ckpt_id}.pt'
        elif model_args.ft_method == 'DRO':
            vit_ckpt = f'/data/nvme1/yxpeng/PycharmProjects/pyvenv-experiments/vision-grokking/checkpoints/vit_dro_new/ViT_coloredmnist_dro_test_{model_args.ckpt_id}.pt'
        state_dict = torch.load(vit_ckpt)
        new_state_dict = {}
        for old_key, value in state_dict.items():
            if "vit.encoder.layer" in old_key:
                new_key = old_key.replace("vit.encoder.layer", "vit.encoder")
                new_state_dict[new_key] = value
            else:
                new_state_dict[old_key] = value
        model = ViTHeadModel(
            config=vit_config,
            state_dict=new_state_dict,
            include_qkv=model_args.include_qkv,
            with_embedding_nodes=data_args.with_embedding_nodes,
            disable_linear_regularization_term=data_args.disable_linear_reg_term,
        ).eval()
        gpt2_model = ViTHeadModel(
            config=vit_config,
            include_qkv=model_args.include_qkv,
            state_dict=new_state_dict,
            with_embedding_nodes=data_args.with_embedding_nodes,
        ).to("cuda").eval()
    else:
        vit_config = ViTConfig()
        vit_config.embedding_bias = False
        vit_config.layernorm_pre = True
        vit_config.layer_norm_eps = 1e-5
        vit_config.proj = True
        vit_config.hidden_act = 'quick_gelu'
        if model_args.ft_method == 'FT':
            clip_checkpoint = "/data/nvme1/yxpeng/PycharmProjects/transfer_learning/logs/full_ft_imagenet_clip_vit_b16/optimizer.args.lr-0.001_with_augment_seed-0_run0/checkpoints/ckp_best_val"
        elif model_args.ft_method == 'LP':
            clip_checkpoint = "/data/nvme1/yxpeng/PycharmProjects/transfer_learning/logs/linprobe_imagenet_clip_vit_b16/weights_0.pkl"
        model = ClipDisentangleModel('ViT-B/16', vit_config, include_qkv=model_args.include_qkv, checkpoint=clip_checkpoint).eval()
        gpt2_model = ClipDisentangleModel('ViT-B/16', vit_config, include_qkv=model_args.include_qkv, checkpoint=clip_checkpoint).to('cuda').eval()

    # test the loaded model's accuracy
    # from tqdm import tqdm
    # batch_size = 16  # Adjust as needed
    # val_loader = DataLoader(imagenet_val, batch_size=batch_size, shuffle=True)
    # origin_model = ClipModel('ViT-B/16').to(torch.float32).to('cuda').eval()
    # with torch.no_grad():
    #     i = 0
    #     align = 0
    #     total = 0
    #     correct = 0
    #     zero_shot_correct = 0
    #     for batch in tqdm(val_loader):
    #         images, labels = batch  # Assuming dataset returns (image, label)
    #         images = images.to("cuda")
    #         labels = labels.to("cuda")
    #
    #         # Get predictions from both models
    #         original_outputs = origin_model.zero_shot_predict(images)
    #         disentangled_output = gpt2_model(input_images=images, labels=labels).logits
    #
    #         # Convert logits to predicted class indices
    #         vit_preds = torch.argmax(original_outputs, dim=-1)
    #         gpt2_preds = torch.argmax(disentangled_output, dim=-1)
    #
    #         # Compare predictions with ground truth
    #         align += (vit_preds == gpt2_preds).sum().item()
    #         zero_shot_correct += (vit_preds == labels).sum().item()
    #         correct += (gpt2_preds == labels).sum().item()
    #         total += labels.size(0)
    #         i += 1
    #         if i % 20 == 0:
    #             print(f'align rate: {align/total}')
    #             print(f'acc: {correct/total}')
    #             print(f'zero_shot_acc: {zero_shot_correct/total}')

    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # tokenizer.pad_token = tokenizer.eos_token
    
    freeze_all_except_pruning_params(model)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

    if training_args.do_eval:
        # We don't have a validation dataset, so we'll just use the test dataset.
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]

    if data_args.corr_mode == 'mean':
        if data_args.dataset == 'imagenet':
            # with open("/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/activations/mean_clip_{model_args.ft_method}_imagenet_val.pkl",
            #           "rb") as f:  # "rb" = read binary
            #     loaded_data = pickle.load(f)
            with open(f"/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/activations/mean_{model_args.ft_method}_clip_imagenet_train.pkl",
                      "rb") as f:  # "rb" = read binary
                loaded_data = pickle.load(f)
        elif data_args.dataset == 'colored_mnist_all_train' or data_args.dataset == 'colored_mnist_all_train_unbiased':
            with open(
                    f"/data/nvme1/yxpeng/PycharmProjects/Edge-Pruning/activations/mean_{model_args.ft_method}_{model_args.ckpt_id}_new_{data_args.dataset}.pkl",
                    "rb") as f:  # "rb" = read binary
                loaded_data = pickle.load(f)
        if loaded_data.is_cuda:
            loaded_data = loaded_data.detach().cpu()
    else:
        loaded_data = None

    # Data collator
    collator = DataCollator(data_args.corr_mode, loaded_activation=loaded_data, ood_dataset=ood_dataset)

    optimizers = get_optimizers(
        model, 
        edges_lr=data_args.edge_learning_rate,
        layers_lr=data_args.layer_learning_rate,
        reg_edges_lr=data_args.reg_edge_learning_rate,
        reg_layers_lr=data_args.reg_layer_learning_rate,
        num_training_steps=training_args.max_steps,
        warmup_steps=training_args.warmup_steps,
        disable_node_loss=data_args.disable_node_loss
    )

    # Initialize our Trainer
    trainer = ImageClassifierTrainer(
        model=model,
        # tokenizer=tokenizer,
        gpt2_model=gpt2_model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=eval_fn,
        optimizers=optimizers,
        start_edge_sparsity=data_args.start_edge_sparsity,
        target_edge_sparsity=data_args.target_edge_sparsity,
        start_layer_sparsity=data_args.start_layer_sparsity,
        target_layer_sparsity=data_args.target_layer_sparsity,
        skip_layer_loss_if_higher_sparsity=data_args.stop_optimizing_layer_if_higher_sparsity,
        num_sparsity_warmup_steps=data_args.num_sparsity_warmup_steps,
        warmup_type=data_args.warmup_type,
        loss_type=data_args.loss_type,
        alpha=data_args.alpha,
        target_class=class_ranges,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(
            resume_from_checkpoint=checkpoint,
        )
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    kwargs = {"finetuned_from": "gpt-2"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()