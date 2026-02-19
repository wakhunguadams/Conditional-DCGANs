import os
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models import Generator, Discriminator
from metrics_utils import get_all_metrics, FIDCalculator

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PANDADataset(Dataset):
    def __init__(self, patches_dir, transform=None, balance=True):
        self.transform = transform
        self.samples = []
        for grade in range(6):
            grade_dir = os.path.join(patches_dir, str(grade))
            if os.path.exists(grade_dir):
                for filename in os.listdir(grade_dir):
                    if filename.endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(grade_dir, filename), grade))
        
        if balance and self.samples:
            counts = {}
            for _, grade in self.samples:
                counts[grade] = counts.get(grade, 0) + 1
            max_count = max(counts.values())
            balanced = []
            for grade in range(6):
                grade_samples = [s for s in self.samples if s[1] == grade]
                if grade_samples:
                    repeats = max_count // len(grade_samples) + 1
                    balanced.extend(grade_samples * repeats)
            random.shuffle(balanced)
            self.samples = balanced[:max_count * 6]
        
        print(f'Loaded {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms(size, augment=True):
    if augment:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def hinge_loss_d(d_real, d_fake):
    loss_real = torch.mean(F.relu(1.0 - d_real))
    loss_fake = torch.mean(F.relu(1.0 + d_fake))
    return loss_real + loss_fake

def hinge_loss_g(d_fake):
    return -torch.mean(d_fake)

def main():
    parser = argparse.ArgumentParser(description='Train SOTA Projected GAN')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='./panda_data/patches_256', help='Data directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume checkpoint')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_dir = './checkpoints'
    samples_dir = './samples'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    dataset = PANDADataset(args.data_dir, get_transforms(256), balance=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    G = Generator(nz=512, style_dim=512).to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.99))
    opt_D = optim.Adam(D.parameters(), lr=args.lr*4, betas=(0.0, 0.99))

    history = {
        'd_loss': [], 'g_loss': [], 
        'mse': [], 'psnr': [], 
        'sharpness_fake': [], 'sharpness_real': [],
        'fid': []
    }
    start_epoch = 1

    fid_calc = FIDCalculator(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt['G'])
        D.load_state_dict(ckpt['D'])
        opt_G.load_state_dict(ckpt['opt_G'])
        opt_D.load_state_dict(ckpt['opt_D'])
        start_epoch = ckpt['epoch'] + 1
        history = ckpt.get('history', history)

    for epoch in range(start_epoch, args.epochs + 1):
        d_losses, g_losses = [], []
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        G.train()
        D.train()
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            bs = imgs.size(0)

            # Train Discriminator
            opt_D.zero_grad()
            d_real = D(imgs, labels)
            
            z = torch.randn(bs, 512, device=device)
            fake = G(z, labels)
            d_fake = D(fake.detach(), labels)
            
            loss_D = hinge_loss_d(d_real, d_fake)
            loss_D.backward()
            opt_D.step()

            # Train Generator (with Diversity Loss)
            opt_G.zero_grad()
            z1 = torch.randn(bs, 512, device=device)
            z2 = torch.randn(bs, 512, device=device)
            fake1 = G(z1, labels)
            fake2 = G(z2, labels)
            
            d_fake_for_g = D(fake1, labels)
            loss_G_adv = hinge_loss_g(d_fake_for_g)
            
            # LZ Loss (Diversity)
            eps = 1e-8
            lz_loss = torch.mean(torch.abs(z1 - z2)) / (torch.mean(torch.abs(fake1 - fake2)) + eps)
            
            loss_G = loss_G_adv + (lz_loss * 0.1)
            loss_G.backward()
            opt_G.step()

            d_losses.append(loss_D.item())
            g_losses.append(loss_G.item())
            pbar.set_postfix({'D': np.mean(d_losses), 'G': np.mean(g_losses), 'LZ': lz_loss.item()})

        history['d_loss'].append(np.mean(d_losses))
        history['g_loss'].append(np.mean(g_losses))

        # Save samples
        if epoch % 5 == 0 or epoch == 1:
            G.eval()
            with torch.no_grad():
                z_test = torch.randn(16, 512, device=device)
                l_test = torch.arange(16, device=device) % 6
                samples = G(z_test, l_test)
                samples = (samples + 1) / 2
                vutils.save_image(samples, f'{samples_dir}/epoch_{epoch:03d}.png', nrow=4)
                
                # Calculate advanced metrics
                batch_metrics = get_all_metrics(imgs, fake.detach(), fid_calc if epoch % 10 == 0 else None)
                for k, v in batch_metrics.items():
                    if k in history:
                        history[k].append(v)
                
                print(f"Metrics - MSE: {batch_metrics['mse']:.4f}, PSNR: {batch_metrics['psnr']:.2f}, Sharp_F: {batch_metrics['sharpness_fake']:.4f}")
                if 'fid' in batch_metrics:
                    print(f"FID: {batch_metrics['fid']:.2f}")

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'G': G.state_dict(),
                'D': D.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
                'history': history,
            }, f'{checkpoint_dir}/ckpt_epoch_{epoch:03d}.pt')

if __name__ == '__main__':
    main()
