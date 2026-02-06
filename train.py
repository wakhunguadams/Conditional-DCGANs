#!/usr/bin/env python3
"""
Conditional DCGAN Training Script for Prostate Cancer Image Synthesis
"""
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.nn.utils import spectral_norm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Config:
    def __init__(self):
        self.image_size = 256
        self.nc = 3
        self.nz = 128
        self.ngf = 64
        self.ndf = 64
        self.num_classes = 6
        self.embed_dim = 128
        self.batch_size = 8
        self.num_epochs = 100
        self.lr_g = 0.0002
        self.lr_d = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.label_smoothing = 0.1
        self.checkpoint_dir = './checkpoints'
        self.samples_dir = './samples'
        self.save_interval = 10
        self.sample_interval = 5


class PANDADataset(Dataset):
    def __init__(self, patches_dir, transform=None, balance=True):
        self.transform = transform
        self.samples = []
        
        for grade in range(6):
            gdir = os.path.join(patches_dir, str(grade))
            if os.path.exists(gdir):
                for f in os.listdir(gdir):
                    if f.endswith(('.png', '.jpg')):
                        self.samples.append((os.path.join(gdir, f), grade))
        
        if balance and self.samples:
            counts = {}
            for _, g in self.samples:
                counts[g] = counts.get(g, 0) + 1
            max_c = max(counts.values())
            balanced = []
            for g in range(6):
                gs = [s for s in self.samples if s[1] == g]
                if gs:
                    balanced.extend(gs * (max_c // len(gs) + 1))
            random.shuffle(balanced)
            self.samples = balanced[:max_c * 6]
        
        print(f'Loaded {len(self.samples)} samples')
        grade_counts = {}
        for _, g in self.samples:
            grade_counts[g] = grade_counts.get(g, 0) + 1
        print('Grade distribution:', grade_counts)
    
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
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


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
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z, labels):
        emb = self.label_emb(labels)
        x = torch.cat([z, emb], 1).view(z.size(0), -1, 1, 1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, ndf, nc, num_classes, img_size):
        super().__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(nc+1, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf*16), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf*16, ndf*32, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf*32), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf*32, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not hasattr(m, 'weight_orig'):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, imgs, labels):
        bs = imgs.size(0)
        lmap = self.label_emb(labels).view(bs, 1, self.img_size, self.img_size)
        x = torch.cat([imgs, lmap], 1)
        return self.main(x).view(-1, 1)


def train_step(G, D, real_imgs, labels, opt_G, opt_D, criterion, config, device):
    bs = real_imgs.size(0)
    real_imgs = real_imgs.to(device)
    labels = labels.to(device)
    
    real_lbl = torch.full((bs, 1), 1 - config.label_smoothing, device=device)
    fake_lbl = torch.zeros(bs, 1, device=device)
    
    # Train D
    D.zero_grad()
    out_real = D(real_imgs, labels)
    loss_real = criterion(out_real, real_lbl)
    
    z = torch.randn(bs, config.nz, device=device)
    fake = G(z, labels)
    out_fake = D(fake.detach(), labels)
    loss_fake = criterion(out_fake, fake_lbl)
    
    loss_D = loss_real + loss_fake
    loss_D.backward()
    opt_D.step()
    
    # Train G
    G.zero_grad()
    z = torch.randn(bs, config.nz, device=device)
    fake = G(z, labels)
    out = D(fake, labels)
    loss_G = criterion(out, real_lbl)
    loss_G.backward()
    opt_G.step()
    
    return loss_D.item(), loss_G.item(), out_real.mean().item(), out_fake.mean().item()


def generate_samples(G, config, device, n_per_class=4):
    G.eval()
    samples = []
    with torch.no_grad():
        for g in range(6):
            z = torch.randn(n_per_class, config.nz, device=device)
            y = torch.full((n_per_class,), g, dtype=torch.long, device=device)
            samples.append(G(z, y))
    G.train()
    return torch.cat(samples)


def save_grid(samples, epoch, config):
    samples = (samples + 1) / 2
    grid = vutils.make_grid(samples.clamp(0, 1), nrow=4, padding=2)
    plt.figure(figsize=(12, 18))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title(f'Epoch {epoch}')
    plt.savefig(f'{config.samples_dir}/epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_history(history, config):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['d_loss'], label='D')
    ax1.plot(history['g_loss'], label='G')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Losses')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['d_real'], label='D(x)')
    ax2.plot(history['d_fake'], label='D(G(z))')
    ax2.axhline(0.5, c='r', ls='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.set_title('Discriminator Output')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{config.samples_dir}/history.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='./panda_data/patches_256', help='Data directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    config = Config()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr_g = args.lr
    config.lr_d = args.lr
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.samples_dir, exist_ok=True)
    
    # Data
    transform = get_transforms(config.image_size)
    dataset = PANDADataset(args.data_dir, transform, balance=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, 
                          shuffle=True, num_workers=2, drop_last=True)
    
    # Models
    G = Generator(config.nz, config.ngf, config.nc, config.num_classes, config.embed_dim).to(device)
    D = Discriminator(config.ndf, config.nc, config.num_classes, config.image_size).to(device)
    
    print(f'Generator params: {sum(p.numel() for p in G.parameters()):,}')
    print(f'Discriminator params: {sum(p.numel() for p in D.parameters()):,}')
    
    # Optimizers
    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))
    opt_D = optim.Adam(D.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))
    sched_G = optim.lr_scheduler.StepLR(opt_G, 50, 0.5)
    sched_D = optim.lr_scheduler.StepLR(opt_D, 50, 0.5)
    
    start_epoch = 1
    if args.resume:
        print(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume)
        G.load_state_dict(ckpt['G'])
        D.load_state_dict(ckpt['D'])
        start_epoch = ckpt['epoch'] + 1
    
    # Training
    history = {'d_loss': [], 'g_loss': [], 'd_real': [], 'd_fake': []}
    
    print(f'\nStarting training for {config.num_epochs} epochs...\n')
    for epoch in range(start_epoch, config.num_epochs + 1):
        d_losses, g_losses, d_reals, d_fakes = [], [], [], []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs}')
        for imgs, labels in pbar:
            ld, lg, dr, df = train_step(G, D, imgs, labels, opt_G, opt_D, criterion, config, device)
            d_losses.append(ld)
            g_losses.append(lg)
            d_reals.append(dr)
            d_fakes.append(df)
            
            pbar.set_postfix({
                'D': f'{np.mean(d_losses):.4f}',
                'G': f'{np.mean(g_losses):.4f}',
                'D(x)': f'{np.mean(d_reals):.3f}',
                'D(G(z))': f'{np.mean(d_fakes):.3f}'
            })
        
        history['d_loss'].append(np.mean(d_losses))
        history['g_loss'].append(np.mean(g_losses))
        history['d_real'].append(np.mean(d_reals))
        history['d_fake'].append(np.mean(d_fakes))
        
        sched_G.step()
        sched_D.step()
        
        if epoch % config.sample_interval == 0 or epoch == 1:
            save_grid(generate_samples(G, config, device), epoch, config)
        
        if epoch % config.save_interval == 0:
            torch.save({
                'G': G.state_dict(),
                'D': D.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
                'epoch': epoch,
                'history': history
            }, f'{config.checkpoint_dir}/ckpt_epoch_{epoch:04d}.pt')
    
    # Save final
    torch.save(G.state_dict(), f'{config.checkpoint_dir}/G_final.pt')
    torch.save(D.state_dict(), f'{config.checkpoint_dir}/D_final.pt')
    plot_history(history, config)
    
    print('\nTraining complete!')
    print(f'Final - D: {history["d_loss"][-1]:.4f}, G: {history["g_loss"][-1]:.4f}')


if __name__ == '__main__':
    main()
