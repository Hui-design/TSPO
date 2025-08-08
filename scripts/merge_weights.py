from transformers import CLIPModel, CLIPProcessor
import torch
from safetensors import safe_open
import os, sys
sys.path.append(os.getcwd())
from model.temporal_agent import MultiModal_Align, TSPOModel

import torch
import torch.nn as nn


clip_path = "ckpt/openai/clip-vit-large-patch14"
selector_path = "ckpt/TSPO/final_10k_16_12_5e-4/model-00004-of-00004.safetensors"
clip_model = CLIPModel.from_pretrained(clip_path, 
                                            attn_implementation="flash_attention_2",
                                            torch_dtype=torch.bfloat16, device_map="auto")
clip_processor = CLIPProcessor.from_pretrained(clip_path)
clip_config = clip_model.config
with safe_open(selector_path, framework="pt", device="cpu") as f:
    module_prefix = "multiModal_align."
    module_state_dict = {
        k.replace(module_prefix, ""): f.get_tensor(k)
        for k in f.keys() if k.startswith(module_prefix)
    }
selector = MultiModal_Align().to(clip_model.device).to(clip_model.dtype)
selector.load_state_dict(module_state_dict)
total_params = sum(p.numel() for p in module_state_dict.values()) + sum(p.numel() for p in clip_model.parameters())
print(f"parameter_num: {total_params / 1e9} B")


def create_and_save_merged_model(clip_path: str, selector_path: str, output_path: str):
    clip_model = CLIPModel.from_pretrained(
        clip_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    with safe_open(selector_path, framework="pt", device="cpu") as f:
        module_prefix = "multiModal_align."
        module_state_dict = {
            k.replace(module_prefix, ""): f.get_tensor(k)
            for k in f.keys() if k.startswith(module_prefix)
        }
    
    merged_model = TSPOModel.from_merged_components(
        clip_model=clip_model,
        selector_state_dict=module_state_dict
    )
    
    merged_model.save_pretrained(output_path)
    
    processor = CLIPProcessor.from_pretrained(clip_path)
    processor.save_pretrained(output_path)
    
    print(f"Merged model saved to {output_path}")

create_and_save_merged_model(clip_path, selector_path, "TSPO-0.4B")