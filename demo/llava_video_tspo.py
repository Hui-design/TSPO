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
from transformers import CLIPProcessor


def load_video_sampled(video_path, max_frames_num, fps, problem, temporal_agent, clip_processor, uniform=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]  # 1fps
    frame_time = [i / fps for i in frame_idx]

    if len(frame_idx) > max_frames_num and not uniform:  
        candidates = vr.get_batch(frame_idx).asnumpy()
        if len(candidates) > 600:
            max_frames_num = 64
            print("The video is too long, we set the sample_num to 64")
        ts_ids, pred_score = temporal_agent(clip_processor, candidates, problem, sample_num=max_frames_num, window_size=12, method='topk') # tch
        frame_idx = [frame_idx[i] for i in ts_ids]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        visualize_images(spare_frames, "demo/sampled_frames_TSPO.jpg", idx_list=[x//fps for x in frame_idx])
    else:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        visualize_images(spare_frames, "demo/sampled_frames_uniform.jpg", idx_list=[x//fps for x in frame_idx])
    # import pdb;pdb.set_trace()
    return spare_frames, frame_time, video_time


if __name__ == "__main__":
    llava_pretrained = "/your/path/lmms-lab/LLaVA-Video-7B-Qwen2"
    TSPO_model_path = "./TSPO-0.4B"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(llava_pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    model.eval()
    
    tspo_model = TSPOModel.from_pretrained(TSPO_model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="auto")  # tch
    clip_processor = CLIPProcessor.from_pretrained(TSPO_model_path) # tch
    
    import pdb; pdb.set_trace()
    # video_path = "demo/ytb__8dMRKQ0Mjo.mp4" 
    # video_path = 'demo/7XWqI121-Q4.mp4' 
    video_path = 'demo/208.mp4'
    question = input("input your question: ")  # what does the woman doing in the beginning?
    problem = question.split("\nOption")[0]
    max_frames_num = 16
    video, _, _ = load_video_sampled(video_path, max_frames_num, 1, problem, tspo_model, clip_processor)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(model.device).to(model.dtype)
    video = [video]
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + question
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    cont = model.generate(
        input_ids,
        images=video,
        modalities= ["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    print(f'\033[32m The answer is: {text_outputs}\033[0m')
