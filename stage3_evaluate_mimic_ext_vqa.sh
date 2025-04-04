#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=64
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

##evaluate using lora
torchrun --master_port 22358 evaluate_hf_mimic_ext_vqa.py \
    --model_name_or_path "./checkpoints/CheX-Phi-3.5-vision-instruct-SFT" \
    --use_flash_attention \
    --use_lora \
    --output_dir './dpo_1k_beta_0.1/checkpoint-2' \
    --save_path  './dpo_1k_beta_0.1/jsons/checkpoint-2.json'

##evaluate using merged model
torchrun --master_port 22358 evaluate_hf_mimic_ext_vqa.py \
    --model_name_or_path "./checkpoints/CheX-Phi-3.5-vision-instruct-DPO" \
    --use_flash_attention \
    --save_path  './dpo_30k_beta_0.15/jsons/dpo_result.json'

