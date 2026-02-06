#!/usr/bin/env python3
"""
Generate synthetic prostate cancer images using trained cDCGAN
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from PIL import Image
import numpy as np
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, num_classes, embed_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        inp = nz + embed_dim
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(inp, ngf*32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*32), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*32, ngf*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*16), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        emb = self.label_emb(labels)
        x = torch.cat([z, emb], 1).view(z.size(0), -1, 1, 1)
        return self.main(x)


def generate_dataset(generator, output_dir, n_per_class=100, device='cuda'):
    """Generate synthetic dataset"""
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    grade_names = ['0_benign', '1_G3+3', '2_G3+4', '3_G4+3', '4_G4+4', '5_high']
    
    for g in range(6):
        gdir = os.path.join(output_dir, f'grade_{grade_names[g]}')
        os.makedirs(gdir, exist_ok=True)
        
        print(f'Generating Grade {g} ({grade_names[g]})...')
        with torch.no_grad():
            for i in tqdm(range(n_per_class)):
                z = torch.randn(1, 128, device=device)
                y = torch.tensor([g], device=device)
                img = ((generator(z, y).squeeze() + 1) / 2).clamp(0, 1)
                img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(img).save(f'{gdir}/syn_{g}_{i:04d}.png')
    
    print(f'\nGenerated {n_per_class * 6} images in {output_dir}')


def generate_grid(generator, output_path, n_per_class=8, device='cuda'):
    """Generate a grid visualization"""
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils
    
    generator.eval()
    samples = []
    
    with torch.no_grad():
        for g in range(6):
            z = torch.randn(n_per_class, 128, device=device)
            y = torch.full((n_per_class,), g, dtype=torch.long, device=device)
            samples.append(generator(z, y))
    
    samples = torch.cat(samples)
    samples = (samples + 1) / 2
    grid = vutils.make_grid(samples.clamp(0, 1), nrow=n_per_class, padding=2)
    
    grade_names = ['Grade 0: Benign', 'Grade 1: G3+3', 'Grade 2: G3+4', 
                   'Grade 3: G4+3', 'Grade 4: G4+4', 'Grade 5: High']
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
    ax.axis('off')
    
    # Add grade labels
    for i, name in enumerate(grade_names):
        y_pos = (i + 0.5) / 6
        ax.text(-0.02, y_pos, name, transform=ax.transAxes, 
               fontsize=12, va='center', ha='right', weight='bold')
    
    plt.title('Generated Prostate Cancer Histopathology Images by ISUP Grade', 
             fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved grid to {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic prostate cancer images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to generator checkpoint')
    parser.add_argument('--output_dir', type=str, default='./synthetic_data', help='Output directory')
    parser.add_argument('--n_per_class', type=int, default=100, help='Number of images per class')
    parser.add_argument('--grid_only', action='store_true', help='Only generate grid visualization')
    parser.add_argument('--grid_samples', type=int, default=8, help='Samples per class in grid')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load generator
    print(f'Loading generator from {args.checkpoint}')
    G = Generator(nz=128, ngf=64, nc=3, num_classes=6, embed_dim=128).to(device)
    
    if args.checkpoint.endswith('.pt'):
        state = torch.load(args.checkpoint, map_location=device)
        if 'G' in state:
            G.load_state_dict(state['G'])
        else:
            G.load_state_dict(state)
    else:
        G.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    G.eval()
    print('Generator loaded successfully')
    
    if args.grid_only:
        output_path = os.path.join(args.output_dir, 'generated_grid.png')
        os.makedirs(args.output_dir, exist_ok=True)
        generate_grid(G, output_path, args.grid_samples, device)
    else:
        generate_dataset(G, args.output_dir, args.n_per_class, device)
        # Also generate grid
        generate_grid(G, os.path.join(args.output_dir, 'generated_grid.png'), 
                     args.grid_samples, device)


if __name__ == '__main__':
    main()
