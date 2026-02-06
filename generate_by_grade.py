#!/usr/bin/env python3
"""
Generate synthetic images for specific ISUP grades
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


def generate_for_grade(generator, grade, n_images, output_dir, device='cuda', batch_size=32):
    """Generate images for a specific grade"""
    grade_names = {
        0: '0_benign',
        1: '1_G3+3',
        2: '2_G3+4',
        3: '3_G4+3',
        4: '4_G4+4',
        5: '5_high'
    }
    
    grade_descriptions = {
        0: 'Grade 0: Benign (no cancer)',
        1: 'Grade 1: Gleason 3+3',
        2: 'Grade 2: Gleason 3+4',
        3: 'Grade 3: Gleason 4+3',
        4: 'Grade 4: Gleason 4+4, 3+5, 5+3',
        5: 'Grade 5: Gleason 4+5, 5+4, 5+5'
    }
    
    generator.eval()
    
    # Create output directory
    grade_dir = os.path.join(output_dir, f'grade_{grade_names[grade]}')
    os.makedirs(grade_dir, exist_ok=True)
    
    print(f'\n{grade_descriptions[grade]}')
    print(f'Generating {n_images} images...')
    
    generated = 0
    with torch.no_grad():
        pbar = tqdm(total=n_images)
        while generated < n_images:
            current_batch = min(batch_size, n_images - generated)
            
            # Generate batch
            z = torch.randn(current_batch, 128, device=device)
            y = torch.full((current_batch,), grade, dtype=torch.long, device=device)
            imgs = generator(z, y)
            
            # Save images
            for i in range(current_batch):
                img = ((imgs[i] + 1) / 2).clamp(0, 1)
                img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_path = os.path.join(grade_dir, f'syn_grade{grade}_{generated:05d}.png')
                Image.fromarray(img).save(img_path)
                generated += 1
                pbar.update(1)
        
        pbar.close()
    
    print(f'✓ Saved {n_images} images to {grade_dir}')
    return grade_dir


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic images for specific grades')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to generator checkpoint')
    parser.add_argument('--grades', type=str, default='0,1,2,3,4,5', 
                       help='Comma-separated grades to generate (e.g., "0,1,2" or "3,4,5")')
    parser.add_argument('--n_per_grade', type=int, default=100, 
                       help='Number of images per grade')
    parser.add_argument('--output_dir', type=str, default='./synthetic_data', 
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for generation')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Parse grades
    grades = [int(g.strip()) for g in args.grades.split(',')]
    grades = [g for g in grades if 0 <= g <= 5]
    
    if not grades:
        print('Error: No valid grades specified. Use grades 0-5.')
        return
    
    print(f'\nGenerating images for grades: {grades}')
    print(f'Images per grade: {args.n_per_grade}')
    print(f'Total images: {len(grades) * args.n_per_grade}')
    
    # Load generator
    print(f'\nLoading generator from {args.checkpoint}')
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
    print('✓ Generator loaded successfully')
    
    # Generate for each grade
    print('\n' + '='*70)
    print('GENERATING SYNTHETIC IMAGES')
    print('='*70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for grade in grades:
        generate_for_grade(G, grade, args.n_per_grade, args.output_dir, 
                          device, args.batch_size)
    
    print('\n' + '='*70)
    print('GENERATION COMPLETE')
    print('='*70)
    print(f'\nGenerated {len(grades) * args.n_per_grade} images')
    print(f'Output directory: {args.output_dir}')
    
    # Show summary
    print('\nGenerated datasets:')
    for grade in grades:
        grade_names = {0: '0_benign', 1: '1_G3+3', 2: '2_G3+4', 
                      3: '3_G4+3', 4: '4_G4+4', 5: '5_high'}
        grade_dir = os.path.join(args.output_dir, f'grade_{grade_names[grade]}')
        count = len([f for f in os.listdir(grade_dir) if f.endswith('.png')])
        print(f'  Grade {grade}: {count} images in {grade_dir}')


if __name__ == '__main__':
    main()
