export WANDB_PROJECT=TSPO
export WANDB_NAME=final_10k_16_12_5e-4

# 禁用WANDB
export WANDB_MODE=disabled


mkdir -p ./ckpt/$WANDB_PROJECT/$WANDB_NAME

SCRIPT_DIR=$(cd "$(dirname $(dirname "${BASH_SOURCE[0]}"))" &>/dev/null && pwd)
export PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH


deepspeed --hostfile hostfile \
    src/open_tspo/tspo.py \
    --deepspeed scripts/zero3.json \
    --output_dir ./ckpt/$WANDB_PROJECT/$WANDB_NAME \
    --model_name_or_path /your/path/lmms-lab/LLaVA-Video-7B-Qwen2 \
    --video_folder /your/path/datasets/lmms-lab/LLaVA-Video-178K \
    --num_generations 8 \
    --dataset_name xxx \
    --jsonl_path final_10k.jsonl \
    --clip_path /your/path/openai/clip-vit-large-patch14 \
    --window_size 12 \
    --training_sample_len 16 \
    --score_tau 0.025 \
    --max_prompt_length 32768 \
    --learning_rate 5e-4 \
    --beta 0.04 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --save_steps 100 \
    --save_total_limit 8 \
    --save_only_model true \
