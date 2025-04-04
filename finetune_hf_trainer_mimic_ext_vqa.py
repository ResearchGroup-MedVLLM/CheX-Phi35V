"""
example for finetuning Phi-3-V on the DocVQA dataset using the Hugging Face Trainer API
Modified from Idefics-2 finetuning notebook:
https://colab.research.google.com/drive/1rm3AGquGEYXfeeizE40bbDtcWh5S4Nlq?usp=sharing

Install dependencies:
    pip install transformers==4.38.1 \
        datasets \
        accelerate==0.30.1 \
        peft \
        Levenshtein \
        deepspeed==0.13.1
minimal run:
    torchrun --nproc_per_node=4 finetune_hf_trainer_docvqa.py
"""
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np

import Levenshtein
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import gather_object
from src.datasets.mimic import MiMiCEXTVQADataset, MiMiCEXTVQADataCollator
from peft import LoraConfig
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
import warnings
warnings.filterwarnings("ignore")

# suggested deepspeed config
DS_CONFIG_DICT = {
    'zero_optimization': {
        'stage': 2,
        'allgather_partitions': True,
        'allgather_bucket_size': 5e8,
        'overlap_comm': True,
        'reduce_scatter': True,
        'reduce_bucket_size': 5e8,
        'contiguous_gradients': True,
        'round_robin_gradients': True,
    },
    'fp16': {
        'enabled': 'auto',
        'loss_scale': 0,
        'loss_scale_window': 1000,
        'initial_scale_power': 16,
        'hysteresis': 2,
        'min_loss_scale': 1,
    },
    'bf16': {'enabled': 'auto'},
    'train_micro_batch_size_per_gpu': 'auto',
    'train_batch_size': 'auto',
    'gradient_accumulation_steps': 'auto',
    'gradient_clipping': 'auto',
}


def create_dataset(dataset_name):
    """
    Create dataset for different types of dataset
    Args:
        dataset_name: str, name of the dataset, e.g. 'diff', 'region', 'easy', 'concat'
    """
    if dataset_name == 'concat':
        train_datasets, eval_datasets, test_datasets = [], [], []
        for name in ['basic', 'region', 'compare']:
            train_file = f'./data/annotations/SFT/train_{name}_vqa.json'
            eval_file = f'./data/annotations/SFT/valid_{name}_vqa.json'
            test_file = f'./data/annotations/SFT/test_{name}_vqa.json'
            img_root = './data/images/MIMIC_CXR_JPG_region/' if name == 'region' else './data/images/MIMIC_CXR_JPG/'
            train_datasets.append(MiMiCEXTVQADataset(annotation_file=train_file, vis_root=img_root, name=name))
            eval_datasets.append(MiMiCEXTVQADataset(annotation_file=eval_file, vis_root=img_root, name=name))
            test_datasets.append(MiMiCEXTVQADataset(annotation_file=test_file, vis_root=img_root, name=name))
        return ConcatDataset(train_datasets), ConcatDataset(eval_datasets), ConcatDataset(test_datasets)
    else:
        train_file = f'./data/annotations/SFT/train_{dataset_name}_vqa.json'
        eval_file = f'./data/annotations/SFT/valid_{dataset_name}_vqa.json'
        test_file = f'./data/annotations/SFT/test_{dataset_name}_vqa.json'
        img_root = './data/images/MIMIC_CXR_JPG_region' if dataset_name == 'region' else './data/images/MIMIC_CXR_JPG/'
        train_dataset = MiMiCEXTVQADataset(annotation_file=train_file, vis_root=img_root, name=dataset_name)
        eval_dataset = MiMiCEXTVQADataset(annotation_file=eval_file, vis_root=img_root, name=dataset_name)
        test_dataset = MiMiCEXTVQADataset(annotation_file=test_file, vis_root=img_root, name=dataset_name)
        return train_dataset, eval_dataset, test_dataset

def create_lora_config(rank, alpha_to_rank_ratio=2.0, dropout=0.0, freeze_vision_model=False):
    linear_modules = [
        # Phi language modules
        'qkv_proj',  # attention
        'o_proj',
        'down_proj',  # MLP
        'gate_up_proj',
        'lm_head',
    ]
    if not freeze_vision_model:
        vision_linear_modules = [
            # CLIP modules
            'q_proj',  # attention
            'k_proj',
            'v_proj',
            'out_proj',
            'fc1',  # MLP
            'fc2',
            # image projection
            'img_projection.0',
            'img_projection.2',
        ]
        linear_modules.extend(vision_linear_modules)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=round(rank * alpha_to_rank_ratio),
        lora_dropout=dropout,
        target_modules=linear_modules,
        init_lora_weights='gaussian',
    )
    return lora_config


def create_model(model_name_or_path, use_flash_attention=False, use_qlora=False):
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16 if use_flash_attention else torch.float16,
        )
        if use_qlora
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        # Phi-3-V is originally trained in bf16 + flash attn
        # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'eager',
        quantization_config=bnb_config,
    )

    return model


def patch_clip_for_lora(model):
    # remove unused parameters and then monkey patch
    def get_img_features(self, img_embeds):
        clip_vision_model = self.img_processor.vision_model
        hidden_states = clip_vision_model.embeddings(img_embeds)
        hidden_states = clip_vision_model.pre_layrnorm(hidden_states)
        patch_feature = clip_vision_model.encoder(
            inputs_embeds=hidden_states, output_hidden_states=True
        ).hidden_states[-1][:, 1:]
        return patch_feature

    image_embedder = model.model.vision_embed_tokens
    layer_index = image_embedder.layer_idx
    clip_layers = image_embedder.img_processor.vision_model.encoder.layers
    if layer_index < 0:
        layer_index = len(clip_layers) + layer_index
    del clip_layers[layer_index + 1 :]
    del image_embedder.img_processor.vision_model.post_layernorm
    image_embedder.get_img_features = get_img_features.__get__(image_embedder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='microsoft/Phi-3.5-vision-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--bf16', action='store_true', help='Use BF16')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--use_qlora', action='store_true', help='Use QLora')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--save_steps', type=int, default=10, help='save_steps')
    parser.add_argument('--num_crops', type=int, default=16, help='Number of maximum image crops')
    parser.add_argument(
        '--num_train_epochs', type=int, default=1, help='Number of training epochs'
    )
    parser.add_argument('--dataset_type', type=str, default='easy', help='Dataset name')
    parser.add_argument('--learning_rate', type=float, default=4.0e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    parser.add_argument('--lora_rank', type=int, default=64, help='LoRA rank')
    parser.add_argument(
        '--lora_alpha_ratio', type=float, default=2, help='LoRA alpha to rank ratio'
    )
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--freeze_vision_model', action='store_true', help='Freeze vision model')
    args = parser.parse_args()

    assert args.num_crops <= 16, 'num_crops must be less than or equal to 16'
    if args.use_qlora:
        args.use_lora = True

    accelerator = Accelerator()

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path, trust_remote_code=True, num_crops=args.num_crops
        )
        # processor.chat_template = processor.tokenizer.chat_template
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
            use_qlora=args.use_qlora,
        )

    train_dataset, eval_dataset, test_dataset = create_dataset(args.dataset_type)   # diff, region, easy

    num_gpus = accelerator.num_processes
    print(f'training on {num_gpus} GPUs')
    assert args.batch_size % num_gpus == 0, 'Batch size must be divisible by the number of GPUs'
    gradient_accumulation_steps = args.batch_size // num_gpus
    if args.bf16:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    # hard coded training args
    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,  # NOTE currently only supports batch_size == 1
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},  # NOTE important for LoRA
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=1500,
        logging_steps=50,
        output_dir=args.output_dir,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=12,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        # generation_max_length=64,
        report_to='none',
        deepspeed=None if args.use_lora else DS_CONFIG_DICT,
        disable_tqdm=not args.tqdm,
        # default:4
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        ddp_find_unused_parameters=False,
    )

    data_collator = MiMiCEXTVQADataCollator(processor)

    if not args.use_qlora:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model = model.to(f'cuda:{local_rank}')

    if args.use_lora:
        patch_clip_for_lora(model)
        lora_config = create_lora_config(
            rank=args.lora_rank,
            alpha_to_rank_ratio=args.lora_alpha_ratio,
            dropout=args.lora_dropout,
            freeze_vision_model=args.freeze_vision_model,
        )
        model.add_adapter(lora_config)
        model.enable_adapters()

    if args.freeze_vision_model:
        model.model.vision_embed_tokens.requires_grad_(False)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    dataloader = trainer.get_train_dataloader()

    trainer.train()
    trainer.save_model()
    if accelerator.is_main_process:
        processor.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()