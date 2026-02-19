import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from models import Generator
from metrics_utils import get_all_metrics, FIDCalculator
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class SimpleImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.png')]
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img

def reconstruct_history():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load real images for comparison
    real_dir = "panda_data/patches_256/0" 
    real_ds = SimpleImageDataset(real_dir, transform=transform)
    real_loader = DataLoader(real_ds, batch_size=64, shuffle=True)
    real_imgs = next(iter(real_loader)).to(device)
    
    fid_calc = FIDCalculator(device)
    
    checkpoint_files = sorted([f for f in os.listdir('checkpoints_proj') if f.startswith('ckpt_epoch_') and f.endswith('.pt')])
    
    history = {
        'epochs': [],
        'mse': [],
        'psnr': [],
        'sharpness_fake': [],
        'sharpness_real': [],
        'fid': [],
        'd_loss': [],
        'g_loss': []
    }
    
    for ckpt_file in tqdm(checkpoint_files, desc="Processing Checkpoints"):
        epoch = int(ckpt_file.split('_')[2].split('.')[0])
        path = os.path.join('checkpoints_proj', ckpt_file)
        
        try:
            state = torch.load(path, map_location=device, weights_only=False)
        except:
            continue
            
        # Store losses from history if available
        if 'history' in state:
            history['d_loss'].append(state['history']['d_loss'][-1] if state['history']['d_loss'] else 0)
            history['g_loss'].append(state['history']['g_loss'][-1] if state['history']['g_loss'] else 0)
        else:
            history['d_loss'].append(0)
            history['g_loss'].append(0)
            
        # Initialize G and load weights
        G = Generator(nz=512, style_dim=512, n_classes=6, ngf=64).to(device)
        G.load_state_dict(state['G'] if 'G' in state else state)
        G.eval()
        
        # Generate fake images
        with torch.no_grad():
            z = torch.randn(64, 512, device=device)
            y = torch.randint(0, 6, (64,), device=device)
            fake_imgs = G(z, y)
        
        metrics = get_all_metrics(real_imgs, fake_imgs, fid_calc)
        
        history['epochs'].append(epoch)
        history['mse'].append(metrics['mse'])
        history['psnr'].append(metrics['psnr'])
        history['sharpness_fake'].append(metrics['sharpness_fake'])
        history['sharpness_real'].append(metrics['sharpness_real'])
        history['fid'].append(metrics['fid'])
        
    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(10, 20))
    
    # 1. Losses
    axes[0].plot(history['epochs'], history['d_loss'], label='D Loss')
    axes[0].plot(history['epochs'], history['g_loss'], label='G Loss')
    axes[0].set_title('Adversarial Losses')
    axes[0].legend()
    
    # 2. Sharpness
    axes[1].plot(history['epochs'], history['sharpness_fake'], label='Fake Sharpness')
    axes[1].plot(history['epochs'], history['sharpness_real'], label='Real Sharpness', linestyle='--')
    axes[1].set_title('Sharpness (Laplacian Variance)')
    axes[1].legend()
    
    # 3. PSNR & MSE
    ax3_twin = axes[2].twinx()
    axes[2].plot(history['epochs'], history['psnr'], label='PSNR (dB)', color='green')
    ax3_twin.plot(history['epochs'], history['mse'], label='MSE', color='red')
    axes[2].set_title('Structural Metrics (PSNR & MSE)')
    axes[2].set_ylabel('PSNR (dB)')
    ax3_twin.set_ylabel('MSE')
    axes[2].legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    
    # 4. FID
    axes[3].plot(history['epochs'], history['fid'], label='FID Score', color='purple')
    axes[3].set_title('Fréchet Inception Distance (FID)')
    axes[3].legend()
    
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    os.makedirs('samples', exist_ok=True)
    plt.savefig('samples/full_metrics_history.png', dpi=150)
    print("Saved full metrics history plot to samples/full_metrics_history.png")

if __name__ == "__main__":
    reconstruct_history()
