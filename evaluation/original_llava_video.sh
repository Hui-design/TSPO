dataset_name=$1
experiment_name=$2

if [ "$dataset_name" = "LongVideoBench" ]; then
    task_name="longvideobench_val_v"
elif [ "$dataset_name" = "VideoMME" ]; then
    task_name="videomme"
elif [ "$dataset_name" = "MLVU" ]; then
    task_name="mlvu_dev"
else
    echo "Our code doesn't support this dataset now"
    exit 1
fi

save_name="${dataset_name}"
export HF_HOME="evaluation/data/videos/${dataset_name}"
export LMMSJSON="evaluation/jsons_idx/${experiment_name}_${dataset_name}_frameIdx.json"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llava_vid_tspo \
    --model_args pretrained=/your/path/lmms-lab/LLaVA-Video-7B-Qwen2,is_uniform=True,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=64,overwrite=False \
    --tasks "${task_name}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${experiment_name}_${dataset_name}" \
    --output_path "./results/${save_name}"