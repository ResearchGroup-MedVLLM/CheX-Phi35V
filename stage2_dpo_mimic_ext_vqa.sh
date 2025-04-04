
# 单卡
export OMP_NUM_THREADS=64
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=1


torchrun --master_port 26886 dpofinetune_hf_trainer_mimic_ext_vqa.py \
   --bf16 \
   --use_lora \
   --lora_rank 128 \
   --model_name_or_path "./checkpoints/CheX-Phi-3.5-vision-instruct-SFT/" \
   --batch_size 256 \
   --output_dir './dpo_output_mimic_rk128_lr4e-6_beta_0.1_24k_hard_16' \
   --learning_rate 4.0e-6 \
   --num_train_epochs 1 \
   --use_flash_attention \
   --freeze_vision_model \
   --save_steps 4 \
   --num_crops 16 \
   --warmup_steps 9 \
   > dpo_output_mimic_rk128_lr4e-6_beta_0.1_24k_hard_16.log