import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from transformers import CLIPModel, CLIPConfig, CLIPProcessor
import PIL.Image as Image
from .utils import generate_uniform_integers, group_features_by_cluster, AKS_sampling

def positional_encoding(T, C):
    position = torch.arange(T).unsqueeze(1)  # [T, 1]
    div_term = torch.exp(torch.arange(0, C, 2) * (-torch.log(torch.tensor(10000.0)) / C))  # [C//2]
    pe = torch.zeros(1, T, C)  # [1, T, C]

    position = torch.arange(T).unsqueeze(1) / T

    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe

class Simple_SelfAttn(nn.Module):
    def __init__(self, dim=4096, num_heads=8, dropout=0.0) -> None:
        """
        Args
        """
        super().__init__()
        self.Self_q = nn.Linear(dim, dim)
        self.Self_k = nn.Linear(dim, dim)
        self.Self_v = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn_o = nn.Linear(dim, dim)

        self.embed_size = dim
        self.num_heads = num_heads
        self.head_dim = self.embed_size // num_heads
        # self.alpha = nn.Parameter(torch.FloatTensor(1)) # fix the "no size" bug

    def scaled_dot_product_attention(self, q, k, v, mask=None, adj=None):
        if adj is not None:
            g_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) + adj
        else:
            g_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        l_scores = g_scores.clone()
        if mask is not None:
            l_scores = l_scores.masked_fill(mask == 0, -1e6)
        
        g_attn_weights = F.softmax(g_scores, dim=-1)
        l_attn_weights = F.softmax(l_scores, dim=-1)
        
        glb = torch.matmul(g_attn_weights, v)
        lcl = torch.matmul(l_attn_weights, v)
        # alpha = torch.sigmoid(self.alpha)
        # alpha = 0.5
        alpha = 0.0
        output = alpha  * glb + (1 - alpha) * lcl
        return output, g_attn_weights, l_attn_weights

    def forward(self, input_emb, mask=None, adj=None):
        # get bs and token_nums
        n = input_emb.shape[0]
        len = input_emb.shape[1]
        # qkv proj
        q = self.Self_q(input_emb)
        k = self.Self_k(input_emb)
        v = self.Self_v(input_emb)

        # seperate into multi-heads
        q = q.view(n, len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(n, len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(n, len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if adj is not None:
            adj = adj.unsqueeze(dim=1).expand(-1, self.num_heads, -1, -1)
        # calculate attention
        context, _, _ = self.scaled_dot_product_attention(q, k, v, mask, adj)
        context = context.transpose(1, 2).contiguous().view(n, -1, self.embed_size)

        # out = self.dropout(self.ffn_o(context)) + context
        # return out
        return context
    
class MultiModal_Align(nn.Module):
    def __init__(self, dim=768, num_heads=8, dropout=0.0, gamma=0.6, bias=0.2) -> None:
        """
        Args
        """
        super().__init__()
        self.temporal = Simple_SelfAttn(dim, num_heads, dropout)
        self.mlp = nn.Sequential(
                  nn.Linear(dim, dim),
                  nn.ReLU(),
                  nn.Linear(dim, dim)
                  )
    def create_causal_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len))  # 下三角矩阵
        return mask

    def create_window_mask(self, seq_len, window_size=8):
        mask = torch.zeros((seq_len, seq_len))
        w_len = window_size
        for j in range(seq_len):
            for k in range(w_len):
                mask[j, min(max(j - w_len // 2 + k, 0), seq_len - 1)] = 1.

        return mask
    
    def pair_cosine(self, a, b):
        cos_sim = torch.einsum('bnc,bmc->bnm', a, b) # [B, N, M]
        B, N, M = cos_sim.shape
        a_norm = torch.sqrt((a**2).sum(dim=-1))         # [B, N]
        a_norm = a_norm.unsqueeze(-1).expand(-1,-1,M)   # [B, N, M]
        b_norm = torch.sqrt((b**2).sum(dim=-1))         # [B, M]
        b_norm = b_norm.unsqueeze(1).expand(-1,N,-1)    # [B, N, M]
        cos_sim = cos_sim / (a_norm*b_norm + 1e-6)
        return cos_sim

    def forward(self, input_emb, text_emb, clip_scores=None, window_size=None, score_tau=0.025):
        """
        input_emb : [T, d]  cls_token
        text_emb : [1, d]  text_embedding
        """
        # transpose
        T, D = input_emb.shape
        input_emb = input_emb.unsqueeze(0)  # [1, T, D]

        adj = None
        window_mask = self.create_window_mask(input_emb.shape[1], window_size=window_size).to(input_emb.device)  # mme16,lvb24
        
        pos_enc = positional_encoding(input_emb.shape[1], input_emb.shape[2]).to(input_emb.device).to(input_emb.dtype)
        input_emb_time = input_emb + pos_enc
        temporal_attn = self.temporal(input_emb_time, window_mask, adj)
        temporal_attn = self.mlp(temporal_attn) + input_emb

        if text_emb.ndim == 2:
            text_emb = text_emb.unsqueeze(0)
        sim_cross = self.pair_cosine(temporal_attn, text_emb) 
        sim_total = sim_cross[0].mean(dim=-1)

        sim_total = sim_total + clip_scores
        
        # train 
        sim_total = sim_total / score_tau

        return sim_total, temporal_attn  # [N, 1]
    

class TSPOModel(CLIPModel):
    def __init__(self, clip_config):
        super().__init__(clip_config)
        self.selector = MultiModal_Align()

    def extract_feature(self, clip_processor, candidates, problem, processor_type='llava'):
        ## extract clip features
        inputs_text = clip_processor(text=problem, return_tensors="pt", padding=True,truncation=True).to(self.device)
        with torch.no_grad():
            text_features = self.get_text_features(**inputs_text)
        image_list = []
        for j in range(len(candidates)):
            if processor_type == 'llava':
                raw_image = np.array(candidates[j])
            else:
                raw_image = candidates[j].permute(1,2,0).cpu().numpy().astype(np.uint8)
            raw_image = Image.fromarray(raw_image)
            image_list.append(raw_image)
        inputs_image = clip_processor(images=image_list, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            image_features = self.get_image_features(**inputs_image) # [bs, 768]
        clip_scores = torch.nn.CosineSimilarity(dim=-1)(text_features, image_features)

        return image_features, text_features, clip_scores

    def temporal_sampling(self, image_features, text_features, clip_scores, method, window_size, sample_num):
        pred_score, _ = self.selector(image_features, text_features, clip_scores, window_size=window_size)
        ts_ids, _ = self.inference_ts(pred_score, method=method, sample_len=sample_num)
        return ts_ids, pred_score


    def forward(self, clip_processor, candidates, problem, sample_num, window_size=12, method='topk', processor_type='llava'):
        ## extract feature
        image_features, text_features, clip_scores = self.extract_feature(clip_processor, candidates, problem, processor_type)

        ## temporal agent
        ts_ids, pred_score = self.temporal_sampling(image_features, text_features, clip_scores, method, window_size, sample_num)
        
        return ts_ids, pred_score
    

    def inference_ts(self, confidence, method, sample_len):
        ## select
        print(f"sample_method: {method}")
        if method == "topk":
            sel_length = min(len(confidence), sample_len)
            sel_idx = torch.sort(torch.topk(confidence, dim=0, k=sel_length, largest=True)[1])[0]
        
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
        elif method == "aks":
            sel_idx = AKS_sampling(confidence.float().cpu().numpy(), sample_len)
            sel_idx = torch.tensor(sel_idx).cuda()
        return sel_idx, confidence
    
    @classmethod
    def from_merged_components(
        cls,
        clip_model: CLIPModel,
        selector_state_dict: dict,
        **kwargs
    ):
        config = clip_model.config
        model = cls(config).to(clip_model.device).to(clip_model.dtype)
        model.load_state_dict(clip_model.state_dict(), strict=False)
        model.selector.load_state_dict(selector_state_dict)
        
        return model
    
    def save_pretrained(self, save_directory: str, **kwargs):
        super().save_pretrained(save_directory, **kwargs)