# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
import os, sys
sys.path.append(os.getcwd())
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
warnings.filterwarnings("ignore")
from src.open_tspo.trainer.utils import visualize_images
from model.temporal_agent import TSPOModel # tch
from transformers import AutoProcessor, CLIPProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import sys, os


if __name__ == "__main__":
    qwen_pretrained = "/your/path/lmms-lab/LLaVA-Video-7B-Qwen2"
    TSPO_model_path = "./TSPO-0.4B"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_pretrained, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(qwen_pretrained)
    
    tspo_model = TSPOModel.from_pretrained(TSPO_model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="auto")  # tch
    clip_processor = CLIPProcessor.from_pretrained(TSPO_model_path) # tch

    # video_path = "demo/ytb__8dMRKQ0Mjo.mp4" 
    # video_path = 'demo/7XWqI121-Q4.mp4' 
    video_path = 'demo/208.mp4'
    question = input("input your question: ")  # what is the scene in the beginning of the video?
    messages = [{
        'role': 'user',
        'content': [{ 'type': 'video', 'video': video_path, 'max_pixels': 480*640}, 
                                {'text': f"<image>\n{question}", 
                                'type': 'text'}],
            }
        ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    problem = question.split("\nOptions")[0]
    sample_num = 16
    # import pdb; pdb.set_trace()
    if video_inputs[0].shape[0] > sample_num:
        if video_inputs[0].shape[0] > 600*2:
            sample_num = 64
            print("The video is too long, we set the sample_num to 64")
        ts_ids, pred_score = tspo_model(clip_processor, video_inputs[0], problem, sample_num=sample_num, window_size=12, method='topk', processor_type='qwen25vl') # tch
        video_inputs[0] = video_inputs[0].cpu()[ts_ids.cpu()]
        visualize_images(video_inputs[0].permute(0,2,3,1).float().cpu().numpy().astype('uint8'), "demo/sampled_frames_TSPO.jpg", idx_list=ts_ids.cpu())

    inputs = processor(
        text=[text],
        images=None,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    generation_config = GenerationConfig(
                max_new_tokens=256,
                do_sample=False,
                temperature=0,
                num_return_sequences=1,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

    generated_ids = model.generate(
        **inputs,
        generation_config=generation_config,   
        ) 

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)  # 注意如果用inputs会被截断
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("==="*50)
    print(f'\033[32m The answer is: {output_text}\033[0m')


