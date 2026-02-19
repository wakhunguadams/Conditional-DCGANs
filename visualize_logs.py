import re
import matplotlib.pyplot as plt
import os
import numpy as np

def parse_logs(log_path):
    epochs = []
    d_losses = []
    g_losses = []
    lz_metrics = []
    
    # Pattern: Epoch 326: 100%|...| 750/750 [...s, D=0.958, G=1.05, LZ=3.67]
    pattern = re.compile(r"Epoch\s+(\d+):\s+100%.*D=([\d.]+),\s+G=([\d.]+),\s+LZ=([\d.]+)")
    
    if not os.path.exists(log_path):
        print(f"Log file {log_path} not found.")
        return None
        
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                d_losses.append(float(match.group(2)))
                g_losses.append(float(match.group(3)))
                lz_metrics.append(float(match.group(4)))
    
    return epochs, d_losses, g_losses, lz_metrics

def plot_history(epochs, d_losses, g_losses, lz_metrics, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Subplot 1: Adversarial Losses
    ax1.plot(epochs, d_losses, label='D Loss', color='#1f77b4', alpha=0.8, linewidth=1.5)
    ax1.plot(epochs, g_losses, label='G Loss', color='#ff7f0e', alpha=0.8, linewidth=1.5)
    ax1.set_title('Granular Adversarial Losses (from Logs)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # Subplot 2: LZ Diversity Metric
    ax2.plot(epochs, lz_metrics, label='LZ Diversity', color='#2ca02c', linewidth=2)
    # Add a trend line (rolling average)
    if len(lz_metrics) > 10:
        rolling_avg = np.convolve(lz_metrics, np.ones(10)/10, mode='valid')
        ax2.plot(epochs[9:], rolling_avg, label='Trend (MA-10)', color='#d62728', linestyle='--', linewidth=2)
        
    ax2.set_title('LZ Diversity Metric Evolution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('LZ Score', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left')
    
    # Annotate key milestones
    ax2.axhline(y=2.0, color='gray', linestyle=':', alpha=0.5)
    ax2.text(epochs[0], 2.1, 'Target Diversity Threshold (2.0)', color='gray', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    log_file = "logs/train_proj_diversity.log"
    output_file = "samples/granular_history.png"
    
    data = parse_logs(log_file)
    if data:
        epochs, d_losses, g_losses, lz_metrics = data
        plot_history(epochs, d_losses, g_losses, lz_metrics, output_file)
