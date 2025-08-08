import torch
import numpy as np
import PIL.Image as Image
import glob, os, re
import random
import matplotlib.pyplot as plt
from decord import VideoReader, cpu
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def repeat_videos(video, repeat_times=4, sample_len=50):
    # video shape is [l, h, w, 3]
    samples = []
    if video.shape[0] <= sample_len:
        for _ in range(repeat_times):
            samples.append(video)
        return samples
    for _ in range(repeat_times):
        indices = np.sort(np.random.choice(video.shape[0], size=sample_len, replace=False))
        samples.append(video[indices])
    return samples

def gen_wrong_video(gen_num, gen_shape):
    b, h, w, c = gen_shape
    wrong_video = np.random.randint(0, 256, (b*gen_num, h, w, c), dtype=np.uint8)
    return wrong_video

def load_video(video_path, max_frames_num=256,fps=1,min_frames_num=50,force_sample=False):
    try:
        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps()/fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i/fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample or len(frame_idx) < min_frames_num:  # debug 0707
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        # import pdb; pdb.set_trace()
    except:
        spare_frames = np.zeros((max_frames_num, 336, 336, 3)).astype(np.uint8)
        frame_time, video_time = None, None
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

def resize_short(video, target_size=336):
    _, H, W, C = video.shape
    vr_frames = torch.from_numpy(video).permute(0, 3, 1, 2).float()  # [T, 3, H, W]
    if H < W:
        new_H = target_size
        new_W = int(W * (target_size / H))
    else:
        new_W = target_size
        new_H = int(H * (target_size / W))
    resized_tensor = F.interpolate(
        vr_frames, 
        size=(new_H, new_W), 
        mode="bilinear",
        align_corners=False 
    )
    video = resized_tensor.permute(0, 2, 3, 1).numpy()  # [T, n, w, 3)
    return video.astype(np.uint8)


def resize_video(video, target_h=480, target_w=640):
    _, H, W, C = video.shape
    vr_frames = torch.from_numpy(video).permute(0, 3, 1, 2).float()  # [T, 3, H, W]
    resized_tensor = F.interpolate(
        vr_frames, 
        size=(target_h, target_w), 
        mode="bilinear",
        align_corners=False 
    )
    video = resized_tensor.permute(0, 2, 3, 1).numpy()  # [T, n, w, 3)
    return video.astype(np.uint8)


def sample_real_frames(data, root, sample_num, target_h=336, target_w=336):
    cur_num = 0
    sample_frames = []
    line = random.sample(data, 1)[0]
    video_path = os.path.join(root, line['video'])
    assert os.path.exists(video_path), print(video_path)
    # print(video_path)
    vr_frames, _, _ = load_video(video_path, max_frames_num=50, fps=1, force_sample=False)
    for i in range(len(vr_frames)):
        resized_pil_image = Image.fromarray(vr_frames[i]).resize((target_w, target_h), Image.BILINEAR)
        resized_array = np.array(resized_pil_image)
        sample_frames.append(resized_array)

    return np.stack(sample_frames, axis=0)

def sample_real_frames_pre(data, root, sample_num, target_h=336, target_w=336):
    cur_num = 0
    sample_frames = []
    while True:
        line = random.sample(data, 1)[0]
        video_path = os.path.join(root, os.path.splitext(line['video'])[0])
        if "llava_hound" in video_path or "shareVideoGPTV" in video_path:
            continue
        assert os.path.exists(video_path), print(video_path)
        image_pathes = sorted(glob.glob(video_path + '/*.jpg'), key=lambda x: int(re.search(r'\d+', x).group()))

        for image_path in image_pathes:
            image = Image.open(image_path)
            resized_pil_image = image.resize((target_w, target_h), Image.BILINEAR)
            resized_array = np.array(resized_pil_image)
            sample_frames.append(resized_array)
            cur_num += 1
            if cur_num >= sample_num:
                break
        if cur_num >= sample_num:
            break
    return np.stack(sample_frames, axis=0)


def shuffle_groups(origin_groups, wrong_groups, t=1):
    # import pdb; pdb.set_trace()
    n1, bs, h, w = origin_groups.shape
    n2, _, _, _ = wrong_groups.shape

    # make sure coudle div t
    if n1 % t != 0 or n2 % t != 0:
        raise ValueError("must be divisible by t")

    groups_ori = n1 // t
    groups_wrong = n2 // t

    # for convinient, seperate to t-len slots, hack by hzf
    ori_reshaped = origin_groups.reshape(groups_ori, t, bs, h, w)
    wrong_reshaped = wrong_groups.reshape(groups_wrong, t, bs, h, w)

    # generate mask
    mask_ori = np.ones((groups_ori, ), dtype=bool)
    mask_wrong = np.zeros((groups_wrong, ), dtype=bool)

    combined = np.concatenate([ori_reshaped, wrong_reshaped])  # [groups_ori+groups_mask, t, bs, h, w]
    combined_mask = np.concatenate([mask_ori, mask_wrong])           # [groups_a+groups_b, t, bs, h, w]
    total_groups = groups_ori + groups_wrong

    # rand
    indices = np.random.permutation(total_groups)
    # import pdb; pdb.set_trace()
    merged_video = np.zeros_like(combined)
    shuffle_mask = np.zeros_like(combined_mask)
    count_ori, count_wrong = 0, 0
    for i, idx in enumerate(indices):
        if idx < len(mask_ori):  
            merged_video[i] = ori_reshaped[count_ori]
            shuffle_mask[i] = True
            count_ori += 1
        else:  
            merged_video[i] = wrong_reshaped[count_wrong]
            shuffle_mask[i] = False
            count_wrong += 1

    # shuffle = combined[indices]  # [total_groups, t, bs, h, w]
    # shuffle_mask = combined_mask[indices]  # [total_groups, t, bs, h, w]

    result = merged_video.reshape(total_groups * t, bs, h, w)
    result_mask = shuffle_mask

    return result, torch.tensor(result_mask)



def shuffle_clips(true_groups, wrong_groups, t=1):
    # try:
    len_group = len(true_groups[0])
    # generate mask
    mask_ori = [np.ones(len_group, dtype=bool) for _ in true_groups]
    mask_wrong = [np.zeros(len_group, dtype=bool) for _ in wrong_groups]

    total_groups = [1] * len(mask_ori) + [0] * len(mask_wrong)
    indices = np.random.permutation(total_groups)
    # import pdb; pdb.set_trace()
    merged_video = np.zeros_like(np.concatenate(true_groups + wrong_groups))
    shuffle_mask = np.zeros_like(np.concatenate(mask_ori + mask_wrong))
    count_ori, count_wrong = 0, 0
    for i, idx in enumerate(indices):
        if idx == 1:  # 来自视频 A
            merged_video[i*len_group:(i+1)*len_group] = true_groups[count_ori]
            shuffle_mask[i*len_group:(i+1)*len_group] = True
            count_ori += 1
        else:  # 来自视频 B
            merged_video[i*len_group:(i+1)*len_group] = wrong_groups[count_wrong]
            shuffle_mask[i*len_group:(i+1)*len_group] = False
            count_wrong += 1
            
    return merged_video, torch.tensor(shuffle_mask)
    # return result, torch.tensor(result_mask)

def shuffle_fixed_clips(true_groups, wrong_groups, t=1):
    # import pdb; pdb.set_trace()
    len_group = len(true_groups[0])
    # generate mask
    mask_ori = [np.ones(len_group, dtype=bool) for _ in true_groups]
    mask_wrong = [np.zeros(len_group, dtype=bool) for _ in wrong_groups]

    # total_groups = [1] * len(mask_ori) + [0] * len(mask_wrong)
    indices = [0] * (len(mask_wrong)//2) + [1] * len(mask_ori) + [0] * (len(mask_wrong) - len(mask_wrong)//2)
    # indices = np.random.permutation(total_groups)
    
    # import pdb; pdb.set_trace()
    merged_video = np.zeros_like(np.concatenate(true_groups + wrong_groups))
    shuffle_mask = np.zeros_like(np.concatenate(mask_ori + mask_wrong))
    count_ori, count_wrong = 0, 0
    for i, idx in enumerate(indices):
        if idx == 1:  
            merged_video[i*len_group:(i+1)*len_group] = true_groups[count_ori]
            shuffle_mask[i*len_group:(i+1)*len_group] = True
            count_ori += 1
        else:  
            merged_video[i*len_group:(i+1)*len_group] = wrong_groups[count_wrong]
            shuffle_mask[i*len_group:(i+1)*len_group] = False
            count_wrong += 1
            
    return merged_video, torch.tensor(shuffle_mask)
    # return result, torch.tensor(result_mask)


def shuffle_clips_1fps(true_groups, wrong_groups, t=1):
    # import pdb; pdb.set_trace()
    # len_group = len(true_groups[0])
    len_true_group = [len(x) for x in true_groups]
    len_false_group = [len(x) for x in wrong_groups] 
    # generate mask
    mask_ori = [np.ones(x, dtype=bool) for x in len_true_group]
    mask_wrong = [np.zeros(x, dtype=bool) for x in len_false_group]

    total_groups = [1] * len(mask_ori) + [0] * len(mask_wrong)
    indices = np.random.permutation(total_groups)
    # import pdb; pdb.set_trace()
    merged_video = np.zeros_like(np.concatenate(true_groups + wrong_groups))
    shuffle_mask = np.zeros_like(np.concatenate(mask_ori + mask_wrong))
    count_ori, count_wrong, cur_idx = 0, 0, 0
    for i, idx in enumerate(indices):
        if idx == 1:  
            len_group = len(true_groups[count_ori])
            merged_video[cur_idx:cur_idx+len_group] = true_groups[count_ori]
            shuffle_mask[cur_idx:cur_idx+len_group] = True
            count_ori += 1
            cur_idx += len_group
        else:  
            len_group = len(wrong_groups[count_wrong])
            merged_video[cur_idx:cur_idx+len_group] = wrong_groups[count_wrong]
            shuffle_mask[cur_idx:cur_idx+len_group] = False
            count_wrong += 1
            cur_idx += len_group
            
    return merged_video.astype(np.uint8), torch.tensor(shuffle_mask)
    # return result, torch.tensor(result_mask)


def plot_smooth_tensor(reward, tensor, clip_tensor, name, output_path='toy_example/save_image_', data_type="time", sigma=1.5):

    data = np.array(tensor) if not isinstance(tensor, np.ndarray) else tensor
    clip_data = np.array(clip_tensor) if not isinstance(tensor, np.ndarray) else tensor

    smoothed = gaussian_filter1d(data, sigma=sigma)
    clip_smoothed = gaussian_filter1d(clip_data, sigma=sigma)
    
    plt.figure(figsize=(5, 2.5))
    plt.plot(smoothed, label=f'Pred Score (Step {name})', color='#ff7f0e')
    # plt.plot(clip_smoothed, alpha=0.3, label='Clip Score', color='#1f77b4')

    if data_type == 'specific':
        plt.title(r'Mean $R_A$: '+f'{reward[:,0].mean():.4f}'+r' Mean $R_T$: '+f'{reward[:,1].mean():.4f}')
    else:
        plt.title(r'Mean $R_A$: '+f'{reward[:,0].mean():.4f}')
    plt.xlabel('Video Frame Index')
    # plt.ylabel('')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.ylim(0., 0.5)

    plt.savefig(output_path+f'{data_type}/'+name, dpi=500, bbox_inches='tight')
    plt.close()
 

def visualize_images(images, save_path=None, nrows=None, ncols=None, figsize=None, idx_list=None):
    T, H, W, C = images.shape
    assert C == 3
    
    if nrows is None and ncols is None:
        ncols = int(np.ceil(np.sqrt(T)))
        nrows = int(np.ceil(T / ncols))
    elif nrows is None:
        nrows = int(np.ceil(T / ncols))
    elif ncols is None:
        ncols = int(np.ceil(T / nrows))
    
    nrows = max(1, min(nrows, T))
    ncols = max(1, min(ncols, T))
    
    if figsize is None:
        figsize = (ncols * 2, nrows * 1.5) 
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if T == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    if idx_list is None:
        idx_list = np.arange(len(images))

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx < T:
                axes[i, j].imshow(images[idx])
                axes[i, j].set_title(f"{idx_list[idx]}", fontsize=18, color='red')
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off') 
    
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    plot_smooth_tensor(torch.randn(20), torch.randn(20), '20')