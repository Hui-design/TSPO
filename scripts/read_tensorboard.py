from tensorboard.backend.event_processing import event_accumulator


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

import numpy as np
import matplotlib.pyplot as plt



def smooth_line(scalars, weight=0.92): 
    last = scalars[0]  
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def get_data(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir,
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })

    ea.Reload()

    loss_data = ea.Scalars("train/rewards/accuracy_reward")
    acc, temporal = [], []

    for i in range(len(loss_data)):
        acc_reward = ea.Scalars("train/rewards/accuracy_reward")[i].value
        if '0.01' not in log_dir:
            temporal_reward = ea.Scalars("train/rewards/temporal_localization_reward")[i].value
        else:
            temporal_reward = ea.Scalars("train/rewards/index_reward")[i].value
        acc.append(acc_reward)
        temporal.append(temporal_reward)
    return acc, temporal

if __name__ == "__main__":
    y1, z1 = get_data(log_dir="ckpt/TSPO/final_10k_16_12_5e-4/runs/Jul27_22-11-10_4d59ed9aaea1343ebd2ab6a469715339-taskrole1-2")
    y2, z2 = get_data(log_dir="ckpt/TSPO/final_10k_16_12_5e-4/runs/Jul27_22-11-10_4d59ed9aaea1343ebd2ab6a469715339-taskrole1-2")
    x = np.arange(len(y1))
    y1_smooth = smooth_line(y1)
    y2_smooth = smooth_line(y2)
    z1_smooth = smooth_line(z1)
    z2_smooth = smooth_line(z2)
    # import pdb; pdb.set_trace()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    alpha = 0.15

    ax1.plot(x, y1, color='#1f77b4', alpha=alpha) 
    ax1.plot(x, y1_smooth, color='#1f77b4', label=r'$\tau$=0.025') 
    ax1.plot(x, y2, color='#ff7f0e', alpha=alpha) 
    ax1.plot(x, y2_smooth, color='#ff7f0e', label=r'$\tau$=0.01') 
    ax1.set_title('Accuracy reward')
    ax1.set_xlabel('training step')
    ax1.set_ylabel('reward')
    # ax1.legend(loc="upper left")
    ax1.set_ylim(0.60, 0.725)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(x, z1, color='#1f77b4', alpha=alpha) 
    ax2.plot(x, z1_smooth, color='#1f77b4', label=r'$\tau$=0.025')  
    ax2.plot(x, z2, color='#ff7f0e', alpha=alpha) 
    ax2.plot(x, z2_smooth, color='#ff7f0e', label=r'$\tau$=0.01')  
    ax2.set_title('Temporal reward')
    ax2.set_xlabel('training step')
    ax2.set_ylabel('reward')
    # ax2.legend(loc="upper left")
    ax2.set_ylim(0.9, 1.)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('RL_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig('RL_curve.pdf', dpi=1000, bbox_inches='tight')

    # import pdb; pdb.set_trace()