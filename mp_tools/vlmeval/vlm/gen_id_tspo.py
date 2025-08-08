from transformers import AutoTokenizer, AutoProcessor, GenerationConfig, CLIPModel, CLIPProcessor
import sys, os
import sys, os, json
import torch
from vlmeval.vlm.base import BaseModel

from vlmeval.dataset import build_video
from tqdm import tqdm
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt

from decord import VideoReader, cpu
import numpy as np
import copy
from transformers import CLIPProcessor


def load_video(video_path, max_frames_num=256,fps=1,force_sample=False):
    try:
        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps()/fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i/fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
    except Exception as e:
        print(f"{str(e)}")
        raise ValueError(f"{video_path} invalid")
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time, torch.tensor(frame_idx)


class GEN_Frame_ID_TSPO(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_path, root=None, save_root=None, sample_num=64,**kwargs):
        self.model_path = model_path
        self.root = root
        self.save_root = save_root
        self.sample_num = sample_num
        sys.path.insert(0, root)
        os.chdir(root)
        from model.temporal_agent import TSPOModel # tch
        self.tspo_model = TSPOModel.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="auto")  # tch
        self.clip_processor = CLIPProcessor.from_pretrained(model_path) # tch

    
    def generate_inner(self, message, index=None, dataset=None):
        # print(message)
        # import pdb; pdb.set_trace()
        video_path = message[0]['value']
        question = message[1]['value'] 
        assert "\nOptions" in question
        problem = question.replace("<image>\n","").replace("Question: ", "").split("\nOptions")[0]
        # print("problem is: ", f'\033[32m{problem}\033[0m')
        save_root = os.path.join(self.save_root, dataset)
        if not os.path.exists(f"{save_root}/{index}.pth"):
            video,_,_, sampled_idx = load_video(video_path, max_frames_num=50000, fps=1, force_sample=False) # 50000
            image_embeddings, text_features, clip_scores = self.tspo_model.extract_feature(self.clip_processor, video, problem)  # new, 一个样本只要过一次
            os.makedirs(save_root, exist_ok=True)
            torch.save({"image": image_embeddings.cpu(), "text": text_features.cpu(),
                        "clip_scores": clip_scores.cpu(), "sampled_idx": sampled_idx}, f"{save_root}/{index}.pth")
        else:
            stat = torch.load(f"{save_root}/{index}.pth")
            image_embeddings, text_features, clip_scores, sampled_idx = stat["image"], stat["text"], stat["clip_scores"], stat["sampled_idx"]
            image_embeddings = image_embeddings.to(self.tspo_model.device).to(self.tspo_model.dtype)
            text_features = text_features.to(self.tspo_model.device).to(self.tspo_model.dtype)
            clip_scores = clip_scores.to(self.tspo_model.device).to(self.tspo_model.dtype)

        sample_num = self.sample_num
        assert dataset in ["LongVideoBench", "MLVU", "VideoMME", "LVBench"]
        method = 'topk' if dataset != 'VideoMME' else 'bin-max'
        if len(image_embeddings) > sample_num:
            with torch.no_grad():
                window_size = 12
                ts_ids, pred_scores = self.tspo_model.temporal_sampling(image_embeddings, text_features, clip_scores, method, window_size, sample_num) 
                abs_ids = sampled_idx[ts_ids.cpu()]
        else:
            abs_ids = sampled_idx
        print(abs_ids, "len:", len(abs_ids))

        return abs_ids.float().tolist()