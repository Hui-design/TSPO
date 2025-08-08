import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from typing import List
import heapq
import PIL.Image as Image
import torch.nn as nn

def generate_uniform_integers(t, l):
    if l <= 0:
        return []
    if l == 1:
        return [t]
    step = t / (l - 1)  # 计算步长
    return [round(i * step) for i in range(l)]

def extract_clip_features(clip_model, clip_processor, video, text):
    # import pdb; pdb.set_trace()
    inputs_text = clip_processor(text=text, return_tensors="pt", padding=True,truncation=True).to(clip_model.device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs_text)
    # print("text_features", text_features)
    image_list = []
    for j in range(len(video)):
        # raw_image = video[j].permute(1,2,0).numpy().astype(np.uint8)
        raw_image = np.array(video[j])
        raw_image = Image.fromarray(raw_image)
        image_list.append(raw_image)
    inputs_image = clip_processor(images=image_list, return_tensors="pt", padding=True).to(clip_model.device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs_image) # [bs, 768]
    # print("image_features", image_features)
    clip_scores = torch.nn.CosineSimilarity(dim=-1)(text_features, image_features)
    return image_features, text_features, clip_scores        


def assign_to_clusters(features, center_indexes):
    pass


def group_features_by_cluster(features, cluster_indices):
    unique_clusters = torch.unique(cluster_indices)
    grouped_features = []
    
    for cluster in unique_clusters:
        features_in_cluster = features[cluster_indices == cluster]
        grouped_features.append(features_in_cluster)
    
    return grouped_features


def uniform_sampling(lst, num_samples):
    n = len(lst)
    if num_samples <= 0 or num_samples > n:
        return []
    
    step = n // num_samples
    remainder = n % num_samples
    
    result = []
    index = 0
    for i in range(num_samples):
        result.append(lst[index])
        index += step + (1 if i < remainder else 0)
    
    return result

def gumbel_softmax(logits, tau=1.0, sample_len=64):
    selection_probs = F.gumbel_softmax(logits, tau=tau, dim=0)
    # selection_probs = F.softmax(logits, dim=0)

    top_k_values, top_k_indices = torch.topk(selection_probs, sample_len, dim=0)
    one_hot = torch.zeros_like(selection_probs).scatter_(0, top_k_indices, 1.0) # [2,3,4], [4,3,2]
    probs = (one_hot - selection_probs).detach() + selection_probs
    top_indices_list = [indices[0].tolist() for indices in top_k_indices]

    log_probs = F.softmax(logits.squeeze(), dim=0).log()  # no randomness
    top_k_indices = torch.sort(top_k_indices.squeeze(1))[0] # tch add
    return top_k_indices, probs.squeeze(1), log_probs


def meanstd(len_scores, dic_scores, n, fns,t1,t2,all_depth):
    # import pdb; pdb.set_trace()
    split_scores = []
    split_fn = []
    no_split_scores = []
    no_split_fn = []
    i= 0
    for dic_score, fn in zip(dic_scores, fns):
        # normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
        score = dic_score['score']
        depth = dic_score['depth']
        mean = np.mean(score)
        std = np.std(score)

        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_score = [score[t] for t in top_n]
        # print(f"split {i}: ",len(score))
        i += 1
        mean_diff = np.mean(top_score) - mean
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth:
        # elif len(score)>(len_scores/n)*2 and len(score) >= 8:
            score1 = score[:len(score)//2]
            score2 = score[len(score)//2:]
            fn1 = fn[:len(score)//2]
            fn2 = fn[len(score)//2:]                       
            split_scores.append(dict(score=score1,depth=depth+1))
            split_scores.append(dict(score=score2,depth=depth+1))
            split_fn.append(fn1)
            split_fn.append(fn2)
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
    if len(split_scores) > 0:
        all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn,t1,t2,all_depth)
    else:
        all_split_score = []
        all_split_fn = []
    all_split_score = no_split_scores + all_split_score
    all_split_fn = no_split_fn + all_split_fn

    return all_split_score, all_split_fn

def AKS_sampling(score, max_num_frames):
    # import pdb; pdb.set_trace()
    ratio = 1
    t1 = 0.2 # videomme: 0.8; LVB: 0.2
    t2 = -100
    all_depth = 3 # videomme: 5; LVB: 3
    print("t1", t1, " all_depth", all_depth)
    outs = []
    segs = []
    fn = [x for x in range(len(score))]
    num = max_num_frames
    if len(score) >= num:
        normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
        a, b = meanstd(len(score), [dict(score=normalized_data,depth=0)], num, [fn], t1, t2, all_depth)
        segs.append(len(a))
        out = []
        if len(score) >= num:
            for s,f in zip(a,b): 
                f_num = int(num / 2**(s['depth']))
                topk = heapq.nlargest(f_num, range(len(s['score'])), s['score'].__getitem__)
                f_nums = [f[t] for t in topk]
                out.extend(f_nums)
        out.sort()
        return out
    else:
        return fn
