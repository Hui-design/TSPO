# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import torch

from datasets import load_dataset
from datasets import Dataset, DatasetDict
import json
from math_verify import parse, verify
from src.open_tspo.trainer import LLaVAVideoTSPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
os.environ['WANDB_DISABLED'] = 'true'
# import warnings
# warnings.filterwarnings("ignore", message="[h264 @ 0x31fe4fc0] mmco: unref short failure")

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    video_folder: Optional[str] = field(
        default=None,
        metadata={"help": "video root path"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "temporal"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "json file path"},
    )
    toy_jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "json file path"},
    )
    clip_path: Optional[str] = field(
        default=None,
        metadata={"help": "clip model path"},
    )
    window_size: Optional[int] = field(
        default=8,
        metadata={"help": "window size"},
    )
    score_tau: Optional[float] = field(
        default=0.025,
        metadata={"help": "score temperature"},
    )
    training_sample_len: Optional[int] = field(
        default=16,
        metadata={"help": "training_sample_len"},
    )
    # irrevalent_video: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "json file path"},
    # )

def map_prediction_to_option(pred):

    model_response = pred.strip().lower()
    
    # extract A), a, A:, (A), A., (A). etc.
    matches = re.findall(r'(?<![a-z])[a-e](?![a-z])', model_response)
    # if len(matches) != 1:
    #     return False
    if len(matches) < 1:
        return False
    
    pred_option = matches[0] # first matched item

    return pred_option

def accuracy_reward(completions, solution, sel_idxs, total_mask, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for sel_idx, content, sol in zip(sel_idxs, contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                student_answer = map_prediction_to_option(content)  
                # ground_truth = ground_truth[0].lower()  # "[0]"" denotes taking the first letter, a, b, c, d 
                ground_truth = map_prediction_to_option(ground_truth)
                # Compare the extracted answers
                if student_answer == ground_truth:
                    # import pdb; pdb.set_trace()
                    # correct = torch.sum(total_mask[sel_idx[1].detach().cpu()]).item()
                    # iou_reward = float(correct / len(sel_idx[1]) > 0.4)
                    # reward = 1.0 * iou_reward
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def temporal_localization_reward(completions, solution, sel_idxs, total_mask, **kwargs):
    rewards = []
    # import pdb; pdb.set_trace()
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for sel_idx in sel_idxs:
        reward = 0.0
        correct = torch.sum(total_mask[sel_idx[1].detach().cpu()]).item()
        reward = correct / len(sel_idx[1])
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Index reward: {reward} -------------\n")
    return rewards 

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "temporal": temporal_localization_reward,
}

def create_dataset_from_jsonl_simple(jsonl_path):
    base_dataset = Dataset.from_json(jsonl_path)
    return DatasetDict({
        "train": base_dataset
    })

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    
    is_toy_example = False
    if script_args.toy_jsonl_path:
        dataset = create_dataset_from_jsonl_simple(script_args.toy_jsonl_path)
        is_toy_example = True
    elif not script_args.toy_jsonl_path and script_args.jsonl_path:
        dataset = create_dataset_from_jsonl_simple(script_args.jsonl_path)
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    # QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    QUESTION_TEMPLATE = "{Question} Please answer with the option's letter from the given choices directly."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        # {"type": "video", "video": example["video"]},
                        # {"type": "video", "bytes": open(example["video"],"rb").read()},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
    }

    if "image" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    elif "video" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(
            make_conversation_video,
        )
    else:
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    def load_jsonl(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    # irrevalent_video = load_jsonl(script_args.irrevalent_video)
    irrevalent_video = load_jsonl(script_args.jsonl_path)
    
    trainer_cls = LLaVAVideoTSPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        video_folder=script_args.video_folder,
        irrevalent_video=irrevalent_video,
        clip_path=script_args.clip_path,
        window_size=script_args.window_size,
        is_toy_example=is_toy_example,
        score_tau=script_args.score_tau,
        training_sample_len=script_args.training_sample_len,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
