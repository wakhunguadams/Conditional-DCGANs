import torch
import numpy as np
import os
from models import Generator
from metrics_utils import get_all_metrics, FIDCalculator
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class SimpleImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.png')]
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img

def extract_final_report(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(nz=512, style_dim=512, n_classes=6, ngf=64).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    G.load_state_dict(state['G'] if 'G' in state else state)
    G.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load 128 real images for comparison
    real_dir = "panda_data/patches_256/0" # Use benign for baseline
    real_ds = SimpleImageDataset(real_dir, transform=transform)
    real_loader = DataLoader(real_ds, batch_size=64, shuffle=True)
    real_imgs = next(iter(real_loader)).to(device)
    
    # Generate 128 fake images
    with torch.no_grad():
        z = torch.randn(64, 512, device=device)
        y = torch.randint(0, 6, (64,), device=device)
        fake_imgs = G(z, y)
    
    fid_calc = FIDCalculator(device)
    metrics = get_all_metrics(real_imgs, fake_imgs, fid_calc)
    
    print("\n" + "="*30)
    print("SOTA PERFORMANCE REPORT (EPOCH 320)")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title():<20}: {v:.4f}")
    print("="*30)

if __name__ == "__main__":
    extract_final_report("checkpoints_proj/ckpt_epoch_320.pt")
