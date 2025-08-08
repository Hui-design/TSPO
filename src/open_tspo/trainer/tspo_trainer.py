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

import copy
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    # Qwen2VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    # is_wandb_available,
)
# from qwen2_vl import Qwen2VLForConditionalGeneration
# from qwen_vl_utils import process_vision_info
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from torch.distributions import Categorical, Bernoulli
from model.utils import *
from transformers import CLIPProcessor, CLIPModel
import PIL.Image as Image

from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names, has_length, is_sagemaker_mp_enabled, logger
import torch.distributed as dist

## LLaVA-Video imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model import *
from decord import VideoReader, cpu
import numpy as np
from .utils import *
import json, random

if is_peft_available():
    from peft import PeftConfig, get_peft_model

# if is_wandb_available():
#     import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class LLaVAVideoTSPOTrainer(Trainer):
    """
    Adaptation of GRPOTrainer

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        video_folder: path to videofolder
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        video_folder:  Optional[str] = None,
        irrevalent_video: Optional[str] = None,
        clip_path: Optional[str] = None,
        window_size: Optional[int] = 8,
        is_toy_example: Optional[bool] = False,
        score_tau: Optional[float] = 0.025,
        training_sample_len: Optional[int] = 16,
    ):
        self.video_folder = video_folder
        self.video_folder_frame = video_folder.replace("LLaVA-Video-178K",'LLaVA-Video-178K_frames')
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            
            # only support to LLaVA-Video now, yet easy to transfer to other Video-LLM
            model = LlavaQwenForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=False, attn_implementation=attn_implementation, torch_dtype="bfloat16")
            model.config.use_cache = False
            processing_class = AutoTokenizer.from_pretrained(model_id)
            vision_tower = model.get_vision_tower()
            vision_tower.requires_grad_(False)
            image_processor = vision_tower.image_processor
            self.image_processor = image_processor
            optimizer_parameters = {
                "params_LLM": [p for n, p in model.named_parameters() if ("multiModal_align" not in n) and (p.requires_grad)],
                "params_t_sampler": [p for n, p in model.named_parameters() if p.requires_grad if ("multiModal_align" in n) and (p.requires_grad)]
                }
            print('len_LLM_params', len(optimizer_parameters['params_LLM']))
            print('len_t_sampler_params', len(optimizer_parameters['params_t_sampler']))

        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        self.irrevalent_video = irrevalent_video
        self.window_size = window_size
        self.score_tau = score_tau
        self.training_sample_len = training_sample_len

        ## fixed irrevalent videos for toy example of Needle in a Haystack
        self.is_toy_example = is_toy_example
        # if self.is_toy_example:
        #     self.fixed_irrevalent_video = []
        #     for _ in range(12):
        #         self.fixed_irrevalent_video.append(sample_real_frames(self.irrevalent_video, root=self.video_folder,
        #                         sample_num=50, target_h=480, target_w=640))  # [50,50,50,50,50,50]

        # # Reference model
        # if is_deepspeed_zero3_enabled():
        #     if "Qwen2-VL" in model_id:
        #         self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        #     elif "Aria" in model_id:
        #         self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        #     else:
        #         self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        # elif peft_config is None:
        #     # If PEFT configuration is not provided, create a reference model based on the initial model.
        #     self.ref_model = create_reference_model(model)
        # else:
        #     # If PEFT is used, the reference model is not needed since the adapter can be disabled
        #     # to revert to the initial model.
        #     self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen2-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id
        else:
            pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            # do_sample=True,
            # temperature=1,  # HACK
            do_sample=False,
            temperature=0.0,  # remove the randomness of LLM
            # num_return_sequences=self.num_generations,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta
        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # import pdb; pdb.set_trace()
        self.clip_model = CLIPModel.from_pretrained(clip_path, 
                                                    attn_implementation="flash_attention_2",
                                                    torch_dtype=torch.bfloat16,
                                                    # device_map="auto",
                                                    device_map=None,
                                                    )
        self.clip_model.requires_grad_(False)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)

        if self.clip_model is not None:
            if self.is_deepspeed_enabled:
                self.clip_model = prepare_deepspeed(self.clip_model, self.accelerator)
            else:
                self.clip_model = self.accelerator.prepare_model(self.clip_model, evaluation_mode=True)

        # if self.ref_model is not None:
        #     if self.is_deepspeed_enabled:
        #         self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
        #     else:
        #         self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    
    def extract_clip_features(self, video, text):
        with unwrap_model_for_generation(self.clip_model, self.accelerator) as unwrapped_model:
            inputs_text = self.clip_processor(text=text, return_tensors="pt", padding=True,truncation=True).to('cuda')
            with torch.no_grad():
                text_features = unwrapped_model.get_text_features(**inputs_text)
            # print("text_features", text_features)
            image_list = []
            for j in range(len(video)):
                # raw_image = video[j].permute(1,2,0).numpy().astype(np.uint8)
                raw_image = np.array(video[j])
                raw_image = Image.fromarray(raw_image)
                image_list.append(raw_image)
            inputs_image = self.clip_processor(images=image_list, return_tensors="pt", padding=True).to(self.clip_model.device)
            with torch.no_grad():
                image_features = unwrapped_model.get_image_features(**inputs_image) # [bs, 768]
            # print("image_features", image_features)
            clip_scores = torch.nn.CosineSimilarity(dim=-1)(text_features, image_features)
            return image_features, text_features, clip_scores 
        
    def add_perturbation(self, x: int, N: int) -> int:
        perturbation = int(random.gauss(mu=0, sigma=1) * 15)
        x_new = x + perturbation
        # 约束边界
        if x_new < 0:
            return 0
        elif x_new >= N:
            return N-1
        else:
            return x_new

    def get_prompt_inputs(self, prompts_text, inputs=None, images=None, video_inputs=None):
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images if "image" in inputs[0] else None,
            videos=video_inputs if "video" in inputs[0] else None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        # import pdb; pdb.set_trace()
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]
        return prompt_inputs
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        str1 = 'Please provide your answer by stating the letter followed by the full option.'
        str2 = 'Please respond with only the letter of the correct answer.'
        if "\nA" in inputs[0]["original_question"]:
            problem = inputs[0]["original_question"].replace("<image>\n","").replace(str1,"").replace(str2,"").split("\nA")[0]
        elif "\n(A)" in inputs[0]["original_question"]:
            problem = inputs[0]["original_question"].replace("<image>\n","").replace(str1,"").replace(str2,"").split("\n(A)")[0]
        # print(problem)
        prompts = [x["prompt"] for x in inputs]
        
        if "image" in inputs[0]:
            images = [x["image"] for x in inputs]
        elif "video" in inputs[0]:
            videos = [x["video"] for x in inputs]
            video_inputs = []
            for (inp_idx, inp) in enumerate(inputs):
                new_inp = inp.copy()
                # import pdb; pdb.set_trace()
                new_inp['prompt'][0]['content'][0]['text'] = os.path.join(self.video_folder, inputs[inp_idx]["video"]) 
                video,_,_ = load_video(os.path.join(self.video_folder,inputs[0]["video"]), 
                                       max_frames_num=128, fps=1, force_sample=False)  
                # new_inp['prompt'][0]['content'][0]['text'] = inputs[inp_idx]["video"]  # for debugging, 0707
                # video,_,_ = load_video(os.path.join(self.video_folder,inputs[0]["video"]), 
                #                        max_frames_num=256, fps=1, force_sample=False)  # for debugging, 0707
                video_inputs.append(video)
            if inputs[0]["type"] == "specific":
                if self.is_toy_example:
                    video = resize_video(video)
                    true_videos = repeat_videos(video, repeat_times=1, sample_len=50)  # [128] -> [50,50,50,50]
                    wrong_videos = self.fixed_irrevalent_video
                    shuffle_video, shuffle_mask = shuffle_fixed_clips(true_videos, wrong_videos)
                else:
                    true_videos = repeat_videos(video, repeat_times=random.randint(1, 4), sample_len=50)  # [128] -> [50,50,50,50]
                    wrong_videos = []
                    wrong_num = 12
                    for _ in range(wrong_num):  # 50*10=500
                        wrong_video = sample_real_frames(self.irrevalent_video, root=self.video_folder,
                            sample_num=len(true_videos[0]), target_h=video.shape[1], target_w=video.shape[2])  # [50,50,50,50,50,50]
                        ## for faster training, you can pre-extract the frames of the irrelevant and load the frames instead of the videos
                        # wrong_video = sample_real_frames_pre(self.irrevalent_video, root=self.video_folder_frame,
                        #     sample_num=len(true_videos[0]), target_h=video.shape[1], target_w=video.shape[2])  # [50,50,50,50,50,50]
                        wrong_videos.append(wrong_video)
                    shuffle_video, shuffle_mask = shuffle_clips(true_videos, wrong_videos)
                video_inputs[0] = shuffle_video
            else:
                shuffle_mask = torch.ones(len(video[0])).bool()
            # import pdb; pdb.set_trace()
            # visualize_images(shuffle_video, save_path='image.jpg')

        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        question = inputs[0]["original_question"].replace("<image>\n","").replace(str1,"").replace(str2,"")
        question = DEFAULT_IMAGE_TOKEN + "\n" + question + "\nPlease answer with the option's letter from the given choices directly."
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.processing_class, 
                                                  IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
        ## temperature annealing
        score_tau = self.score_tau - (self.score_tau-0.01)/self.state.max_steps*self.state.global_step
        with torch.no_grad():
            image_embeddings, text_features, clip_scores = self.extract_clip_features(video_inputs[0], problem)  # new, 一个样本只要过一次
        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            num_generations = self.num_generations
            temp_generation_config = copy.deepcopy(self.generation_config)
            temp_generation_config.num_return_sequences = 1
            all_prompt_completions = []
            temp_prompt_inputs_batch = []
            all_completion_ids = []
            all_tsids = []
            for i in range(num_generations):  
                ## step1: sampling frames
                if inputs[0]["type"] == "specific":
                    sample_len = self.training_sample_len
                else:
                    sample_len = self.training_sample_len // 2
                ts_ids, _, _ = unwrapped_model.temporal_sampling(image_embeddings, text_features, clip_scores, 
                                                                 sample_len=sample_len, window_size=self.window_size, score_tau=score_tau) 
                ts_video_inputs = [video_inputs[0][ts_ids[1].cpu()]]

                if self.is_toy_example:
                    tmp_inputs = copy.deepcopy(ts_video_inputs)

                ## step2: transform inputs
                # None for LLaVA-Video, as it does not pre-fill input ids
                for kk in range(len(ts_video_inputs)):
                    ts_video_inputs[kk] = self.image_processor.preprocess(ts_video_inputs[kk], return_tensors="pt")["pixel_values"].to(image_embeddings.device).to(image_embeddings.dtype)  # .half()
                          
                ## step3: running mllms
                completion = unwrapped_model.generate(  # previous bug: model
                    input_ids,
                    images=ts_video_inputs,
                    modalities= ["video"],
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=256,
                )  # LLaVA only outputs completion 
                
                all_tsids.append(ts_ids)
                all_completion_ids.append(completion)

        per_token_kl_batch, completions_batch, ts_logps_batch = [], [], []
        for i, (ts_ids, completion_ids) in enumerate(zip(all_tsids, all_completion_ids)):
            ## ts_logps is the key to optimize the temporal sampler
            _, ts_logps_raw, pred_scores = model.temporal_sampling(image_embeddings, text_features, clip_scores, ts_ids=ts_ids, 
                                                                   sample_len=sample_len, window_size=self.window_size, score_tau=score_tau)  # new
            ts_logps = ts_logps_raw[ts_ids[1]]  
            ts_logps_batch.append(ts_logps) 

            assert clip_scores.shape == pred_scores.shape
            per_token_kl_batch.append(torch.tensor([0.0]).to(ts_logps.device))
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                completions = [[{"role": "assistant", "content": completion}] for completion in completions]
            completions_batch += completions
        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        device = self.accelerator.device

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        ts_length = [len(x[1]) for x in all_tsids] # [bs]
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions_batch, 
                                             sel_idxs=all_tsids, total_mask=shuffle_mask, clip_scores=clip_scores, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)  # [bs, 2]

        if inputs[0]["type"] == "specific":
            rewards = rewards_per_func.sum(dim=1)
        else:
            rewards = rewards_per_func[:,0:1].sum(dim=1) + 1
        # import pdb; pdb.set_trace()
        if self.is_toy_example:
            acc_reward = [rewards_per_func[x,0].item() for x in range(len(rewards_per_func))]
            temporal_reward = [rewards_per_func[x,1].item() for x in range(len(rewards_per_func))]
            print('accuracy reward', acc_reward)
            if inputs[0]["type"] == "specific":
                print('temporal reward', temporal_reward)
            training_step = f'{self.state.global_step}'
            os.makedirs(f'toy_example/save_image_{inputs[0]["type"]}', exist_ok=True)
            visualize_images(tmp_inputs[0], f'toy_example/save_image_{inputs[0]["type"]}/sampled_frames_{training_step}.jpg', idx_list=ts_ids[1].tolist())    
            plot_smooth_tensor(rewards_per_func, pred_scores.float().detach().cpu()*score_tau, 
                               clip_scores.float().detach().cpu(), f'{training_step}', data_type=inputs[0]["type"])
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        loss_list, mean_kl_list, completion_length_list = 0.0, 0.0, 0
        for batch_idx in range(self.num_generations):
            # token_probs_item = torch.exp(per_token_logps_batch[batch_idx] - per_token_logps_batch[batch_idx].detach()).mean()
            ts_probs_item = torch.exp(ts_logps_batch[batch_idx] - ts_logps_batch[batch_idx].detach()).mean()
            # probs_loss = token_probs_item * ts_probs_item * advantages[batch_idx]
            probs_loss = ts_probs_item * advantages[batch_idx]
            # mean_kl = per_token_kl_batch[batch_idx].mean() + ts_kl_batch[batch_idx].mean()
            mean_kl = per_token_kl_batch[batch_idx].mean()
            # loss = - (probs_loss - self.beta * mean_kl)  
            loss = - (probs_loss) 
            loss_list += loss
            mean_kl_list += mean_kl 
            completion_length_list += all_completion_ids[batch_idx].shape[1]
        loss_avg = loss_list / self.num_generations
        mean_kl_avg = mean_kl_list / self.num_generations
        completion_length_avg = completion_length_list / self.num_generations
        completion_length_avg = self.accelerator.gather_for_metrics(torch.tensor([completion_length_avg]).to(device)).float().mean().item()
        # import torch.distributed as dist
        # print(f"Rank {dist.get_rank()}: {loss_avg}")

        ts_length = torch.tensor([len(x[1]) for x in all_tsids]).float().mean().to(device)
        self._metrics["ts_length"].append(self.accelerator.gather_for_metrics(ts_length).float().mean().item())
        self._metrics["completion_length"].append(completion_length_avg)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # if 'format' in reward_func_name:
            #     reward_func_name = 'ts_length'
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        
        self._metrics["advantages"].append(self.accelerator.gather_for_metrics(advantages).mean().item())
        
        self._metrics["reward_mean"].append(self.accelerator.gather_for_metrics(mean_grouped_rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        # self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl_avg).mean().item())

        # self._metrics["entropy"].append(self.accelerator.gather_for_metrics(entropy_avg).mean().item())

        return loss_avg


    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            # wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

