import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_metrics(history, save_path):
    epochs = range(1, len(history['d_loss']) + 1)
    
    # Identify which metrics are available
    available_metrics = history.keys()
    
    # Group logically
    groups = [
        (['d_loss', 'g_loss'], 'Adversarial Losses', 'Loss'),
        (['psnr'], 'Peak Signal-to-Noise Ratio (PSNR)', 'dB'),
        (['mse'], 'Mean Squared Error (MSE)', 'Error'),
        (['sharpness_fake', 'sharpness_real'], 'Sharpness (Laplacian Variance)', 'Variance'),
        (['fid'], 'FID Score', 'Score')
    ]
    
    # Filter groups based on availability
    active_groups = []
    for keys, title, ylabel in groups:
        if any(k in history and len(history[k]) > 0 for k in keys):
            active_groups.append((keys, title, ylabel))
            
    if not active_groups:
        print("No metrics to plot.")
        return

    fig, axes = plt.subplots(len(active_groups), 1, figsize=(10, 5 * len(active_groups)))
    if len(active_groups) == 1:
        axes = [axes]
        
    for i, (keys, title, ylabel) in enumerate(active_groups):
        for key in keys:
            if key in history and len(history[key]) > 0:
                metric_len = len(history[key])
                loss_len = len(history['d_loss'])
                if metric_len == loss_len:
                    x = epochs
                else:
                    x = np.linspace(1, loss_len, metric_len)
                
                label = key.replace('_', ' ').title()
                axes[i].plot(x, history[key], label=label, marker='o' if metric_len < 20 else None)
        
        axes[i].set_title(title)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(ylabel)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'Saved metrics plot to {save_path}')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint with history')
    parser.add_argument('--output', type=str, default='./samples/metrics_plot.png', help='Output path for plot')
    args = parser.parse_args()
    
    if os.path.exists(args.checkpoint):
        # weights_only=False fix for PyTorch 2.6+
        try:
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            
        if 'history' in ckpt:
            plot_metrics(ckpt['history'], args.output)
        else:
            print('No history found in checkpoint.')
    else:
        print(f'Checkpoint {args.checkpoint} not found.')
