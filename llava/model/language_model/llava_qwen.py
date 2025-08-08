#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig
from model.temporal_agent import MultiModal_Align
from torch.distributions import Categorical, Bernoulli
from model.utils import *
import PIL.Image as Image
import random



class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.model.requires_grad_(False)
        print('set LLM false')
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.requires_grad_(False)
        print('set lm_head false')
        self.multiModal_align = MultiModal_Align()

        # Initialize weights and apply final processing
        self.post_init()

    def get_confidence_score(self):
        return self.multiModal_align

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
    
    def temporal_sampling(self, image_embeddings, text_features, clip_scores=None, sample_len=64, ts_ids=None, 
                          window_size=None, score_tau=0.025, method=None):
        confidence, _ = self.get_confidence_score()(image_embeddings, text_features, clip_scores, window_size, score_tau)
        assert confidence.ndim == 1
        if method is None:
            if ts_ids is None:
                sel_idx, probs, logp_ts = gumbel_softmax(confidence.unsqueeze(1), sample_len=sample_len)
                sel_idx = (sel_idx.clone(), sel_idx)
            else: 
                _, probs, logp_ts = gumbel_softmax(confidence.unsqueeze(1), sample_len=sample_len)
                sel_idx = ts_ids
            return sel_idx, logp_ts, confidence
        else:
            return self.inference_ts(confidence, sample_len=sample_len, method=method)
    
    def inference_ts(self, confidence, sample_len=64, method="topk"):
        ## select
        print(f"sample_method: {method}")
        if method == "aks":
            sel_idx = AKS_sampling(confidence.float().cpu().numpy(), sample_len)
            sel_idx = torch.tensor(sel_idx).cuda()
            return sel_idx, confidence

        elif method == "topk":
            sel_length = min(len(confidence), sample_len)
            sel_idx = torch.sort(torch.topk(confidence, dim=0, k=sel_length, largest=True)[1])[0]
            return sel_idx, confidence
        
        elif method == "bin-max":
            ## step1: split bins
            idx = torch.arange(len(confidence)).to(confidence.device)
            sel_length = min(len(confidence), sample_len)
            proposal_idx = generate_uniform_integers(len(confidence)-1, sel_length)
            slots_index = torch.tensor([torch.argmin(torch.abs(x-torch.tensor(proposal_idx))) for x in torch.arange(len(confidence))])
            confidence_slots = group_features_by_cluster(confidence, slots_index)

            ## step2: argmax per bin
            accept_idxs, start = [], 0
            for _, slot in enumerate(confidence_slots):
                accept_idx = slot.argmax() # top-1 index
                slot_idx = idx[start:start+len(slot)].to(confidence.device)
                start += len(slot)
                accept_idx = slot_idx[accept_idx]
                accept_idxs.append(accept_idx) 
            sel_idx = torch.stack(accept_idxs, dim=0)
            return sel_idx, confidence

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
