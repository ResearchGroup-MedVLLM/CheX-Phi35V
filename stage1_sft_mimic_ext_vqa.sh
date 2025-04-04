#!/bin/bash
export CUDA_VISIBLE_DEVICES=3,4
export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


accelerate launch --main_process_port 25191 finetune_hf_trainer_mimic_ext_vqa.py \
    --bf16 \
    --use_lora \
    --use_flash_attention \
    --model_name_or_path "./checkpoints/Phi-3.5-vision-instruct" \
    --batch_size 128 \
    --output_dir './concat_mimic_ext_vqa_ep2_rk128_lr8e-5' \
    --learning_rate 8.0e-5 \
    --num_train_epochs 2 \
    --save_steps 500 \
    --lora_rank 128 \
    --freeze_vision_model \
    --dataset_type concat \
    > concat_mimic_ext_vqa_ep1_rk128_lr6e-5.log
