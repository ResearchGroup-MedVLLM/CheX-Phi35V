import argparse
import json
import os
import random
from pathlib import Path

import Levenshtein
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
import numpy as np
# from datasets import load_dataset
from src.datasets.mimic import MiMiCEXTVQADataset, MiMiCEXTVQADataCollator
from torch.utils.data import Dataset, ConcatDataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
import warnings
warnings.filterwarnings("ignore")


def create_test_dataset(dataset_name):
    """
    Create dataset for different types of dataset
    Args:
        dataset_name: str, name of the dataset, e.g. 'compare', 'region', 'basic', 'concat'
    """
    if dataset_name == 'concat':
        test_datasets = []
        for name in ['basic', 'region', 'compare']:
            test_file = f'./data/annotations/SFT/test_{name}_vqa.json'
            img_root = './data/images/MIMIC_CXR_JPG_region/' if name == 'region' else './data/images/MIMIC_CXR_JPG/'
            test_datasets.append(MiMiCEXTVQADataset(annotation_file=test_file, vis_root=img_root, name=name))
        return ConcatDataset(test_datasets)
    else:
        test_file = f'./data/annotations/SFT/test_{dataset_name}_vqa.json'
        img_root = './data/images/MIMIC_CXR_JPG_region/' if dataset_name == 'region' else './data/images/MIMIC_CXR_JPG/'
        return MiMiCEXTVQADataset(annotation_file=test_file, vis_root=img_root, name=dataset_name)




@torch.no_grad()
def evaluate_with_token_probs(model, processor, eval_dataset, save_path=None, disable_tqdm=False):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    model.eval()
    save_info_list = []
    acc = []

    print(f"evaluating on {len(eval_dataset)} examples")
    # eval_dataset_shard = eval_dataset.shard(num_shards=world_size, index=rank) 
    for i in tqdm(range(len(eval_dataset)), disable=(rank != 0) or disable_tqdm):
        # if i > 1999:
        #     break
        # Phi-3-V currently only supports batch_size == 1
        example = eval_dataset[i]
        if example['image1'] is None:
            image = [example['image']]
        else:
            image = [example['image'], example['image1']]
        question = example['question']
        prompt = example['prompt']
        prompt_message = {
            'role': 'user',
            'content': prompt,
        }
        prompt = processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )

        inputs = processor(prompt, image, return_tensors='pt').to(f'cuda:{local_rank}')
        generated_ids = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=128,
            output_scores=True,
            return_dict_in_generate=True
        )

        batch_index = 0
        generated_texts = processor.batch_decode(
            generated_ids.sequences[:, inputs['input_ids'].size(1) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # add the token probabilities
        token_probs = []
        generated_tokens = []
        for i, scores in enumerate(generated_ids.scores):
            probs = torch.softmax(scores, dim=-1)
            generated_token_id = generated_ids.sequences[batch_index, inputs['input_ids'].size(1) + len(token_probs)]
            token_prob = probs[batch_index, generated_token_id].item()
            token_probs.append(token_prob)
        
        # print("Generated text and token probabilities:")
        for idx, prob in enumerate(token_probs):
            token = processor.decode(generated_ids.sequences[batch_index, inputs['input_ids'].size(1) + idx])
            # print(f"{token} - Probability: {prob}")
            generated_tokens.append(token)

        answer = example['answer'][0] if type(example['answer']) is list else example['answer']
        generated_dict = {"image_id": example['image_id'],
                "image_id1": example['image_id1'],
                "question": question,
                "answer": answer,
                "explanation": example['explanation'],
                'token_probs': token_probs,
                'token_preds': generated_tokens
                }
        for i, text in enumerate(generated_texts):
            generated_dict[f"generated_report_{i}"] = text.strip()
            if answer.lower() in text.lower():
                generated_dict["acc"] = 1
                acc.append(1)
            else:
                generated_dict["acc"] = 0
                acc.append(0)

        save_info_list.append(generated_dict)

        
        if len(save_info_list) == 10000 or len(save_info_list) == 20000:
            if rank == 0 and save_path:
                partial_save_path = f"{save_path.replace('.json', '')}_partial_{len(save_info_list)}.json"
                
                with open(partial_save_path, 'w') as f:
                    save_dict = {
                        'sample info': save_info_list,
                        'acc': np.mean(acc),
                    }
                    json.dump(save_dict, f)
                print(f"Saved partial results after {len(save_info_list)} samples.")
    
    # gather outputs from all ranks
    save_info_list = gather_object(save_info_list)
    acc = gather_object(acc)

    if rank == 0:
        if save_path:
            with open(save_path, 'w') as f:
                save_dict = {
                    'sample info': save_info_list,
                    'acc': np.mean(acc),
                }
                json.dump(save_dict, f)


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
        default='./checkpoints/Phi-3.5-vision-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument('--dataset_type', type=str, default='easy', help='Dataset name')
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--output_dir', type=str, default=None, help='Output LoRA directory')
    parser.add_argument('--save_path', type=str, help='Output Json directory')
    parser.add_argument('--num_crops', type=int, default=16, help='Number of maximum image crops')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help='Disable tqdm')

    args = parser.parse_args()
    assert args.num_crops <= 16, 'num_crops must be less than or equal to 16'
    
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, num_crops=args.num_crops
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        # Phi-3-V is originally trained in bf16 + flash attn
        # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
        torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if args.use_flash_attention else 'eager',
    )

    print(f"base model:\n{model}")
    patch_clip_for_lora(model)
    if args.output_dir is not None:
        model.load_adapter(args.output_dir)
        print(f"lora model:\n{model}")

    print(f"dataset type: {args.dataset_type}")
    test_dataset = create_test_dataset(args.dataset_type)   # compare, region, easy

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    model = model.to(f'cuda:{local_rank}')
    print(local_rank)
    evaluate_with_token_probs(
        model,
        processor,
        test_dataset,
        save_path=args.save_path,
        disable_tqdm=not args.tqdm,
    )



if __name__ == '__main__':
    main()