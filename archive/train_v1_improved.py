#!/usr/bin/env python3
"""
Improved Conditional DCGAN with Progressive Training and Advanced Techniques
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
import torch.nn.functional as F


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
        self.batch_size = 16
        self.num_epochs = 200
        self.lr_g = 0.0001  # Lower learning rate
        self.lr_d = 0.0004  # D learns faster
        self.beta1 = 0.0    # Adam beta1 = 0 for stability
        self.beta2 = 0.999
        self.lambda_gp = 10  # Gradient penalty
        self.n_critic = 5    # Train D more than G
        self.checkpoint_dir = './checkpoints_improved'
        self.samples_dir = './samples_improved'
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
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.08),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


class SelfAttention(nn.Module):
    """Self-attention layer for capturing long-range dependencies"""
    def __init__(self, in_channels):
        super().__init__()
        self.query = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.size()
        
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        
        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        return self.gamma * out + x


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1))
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ImprovedGenerator(nn.Module):
    """Improved Generator with self-attention and residual connections"""
    def __init__(self, nz, ngf, nc, num_classes, embed_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        inp = nz + embed_dim
        
        # Initial projection
        self.init = nn.Sequential(
            nn.ConvTranspose2d(inp, ngf*32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*32),
            nn.ReLU(True)
        )
        
        # Upsampling blocks with residual connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf*32, ngf*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*16),
            nn.ReLU(True)
        )
        self.res1 = ResidualBlock(ngf*16)
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True)
        )
        self.res2 = ResidualBlock(ngf*8)
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True)
        )
        self.attn = SelfAttention(ngf*4)  # Self-attention at 32x32
        self.res3 = ResidualBlock(ngf*4)
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True)
        )
        self.res4 = ResidualBlock(ngf*2)
        
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        
        self.final = nn.Sequential(
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
        
        x = self.init(x)
        x = self.res1(self.up1(x))
        x = self.res2(self.up2(x))
        x = self.res3(self.attn(self.up3(x)))
        x = self.res4(self.up4(x))
        x = self.up5(x)
        return self.final(x)


class ImprovedDiscriminator(nn.Module):
    """Improved Discriminator with self-attention and spectral normalization"""
    def __init__(self, ndf, nc, num_classes, img_size):
        super().__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        
        # Downsampling blocks
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(nc+1, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True)
        )
        
        self.attn = SelfAttention(ndf*4)  # Self-attention at 32x32
        
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv6 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf*16, ndf*32, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf*32),
            nn.LeakyReLU(0.2, True)
        )
        
        self.final = spectral_norm(nn.Conv2d(ndf*32, 1, 4, 1, 0, bias=False))
        
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
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attn(self.conv3(x))
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return self.final(x).view(-1, 1)


def compute_gradient_penalty(D, real_imgs, fake_imgs, labels, device):
    """Compute gradient penalty for WGAN-GP"""
    alpha = torch.rand(real_imgs.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    
    d_interpolates = D(interpolates, labels)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_step(G, D, real_imgs, labels, opt_G, opt_D, config, device, step):
    """Improved training step with Wasserstein loss and gradient penalty"""
    bs = real_imgs.size(0)
    real_imgs = real_imgs.to(device)
    labels = labels.to(device)
    
    # Train Discriminator
    for _ in range(config.n_critic):
        D.zero_grad()
        
        # Real images
        d_real = D(real_imgs, labels)
        
        # Fake images
        z = torch.randn(bs, config.nz, device=device)
        fake = G(z, labels).detach()
        d_fake = D(fake, labels)
        
        # Wasserstein loss
        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        
        # Gradient penalty
        gp = compute_gradient_penalty(D, real_imgs, fake, labels, device)
        d_loss_total = d_loss + config.lambda_gp * gp
        
        d_loss_total.backward()
        opt_D.step()
    
    # Train Generator
    G.zero_grad()
    z = torch.randn(bs, config.nz, device=device)
    fake = G(z, labels)
    g_output = D(fake, labels)
    g_loss = -torch.mean(g_output)
    
    g_loss.backward()
    opt_G.step()
    
    return d_loss.item(), g_loss.item(), d_real.mean().item(), d_fake.mean().item()


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
    ax1.set_title('Wasserstein Losses')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['d_real'], label='D(x)')
    ax2.plot(history['d_fake'], label='D(G(z))')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.set_title('Discriminator Output')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{config.samples_dir}/history.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr_g', type=float, default=0.0001, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.0004, help='Discriminator learning rate')
    parser.add_argument('--data_dir', type=str, default='./panda_data/patches_256', help='Data directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    config = Config()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr_g = args.lr_g
    config.lr_d = args.lr_d
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.samples_dir, exist_ok=True)
    
    # Data
    transform = get_transforms(config.image_size)
    dataset = PANDADataset(args.data_dir, transform, balance=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, 
                          shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    
    # Models
    G = ImprovedGenerator(config.nz, config.ngf, config.nc, config.num_classes, config.embed_dim).to(device)
    D = ImprovedDiscriminator(config.ndf, config.nc, config.num_classes, config.image_size).to(device)
    
    print(f'Generator params: {sum(p.numel() for p in G.parameters()):,}')
    print(f'Discriminator params: {sum(p.numel() for p in D.parameters()):,}')
    
    # Optimizers (Adam with beta1=0 for stability)
    opt_G = optim.Adam(G.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))
    opt_D = optim.Adam(D.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))
    
    # Learning rate schedulers
    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, config.num_epochs)
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, config.num_epochs)
    
    start_epoch = 1
    if args.resume:
        print(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt['G'])
        D.load_state_dict(ckpt['D'])
        start_epoch = ckpt['epoch'] + 1
    
    # Training
    history = {'d_loss': [], 'g_loss': [], 'd_real': [], 'd_fake': []}
    
    print(f'\nStarting improved training for {config.num_epochs} epochs...')
    print(f'Using Wasserstein loss with gradient penalty')
    print(f'Self-attention and residual connections enabled\n')
    
    global_step = 0
    for epoch in range(start_epoch, config.num_epochs + 1):
        d_losses, g_losses, d_reals, d_fakes = [], [], [], []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs}')
        for imgs, labels in pbar:
            ld, lg, dr, df = train_step(G, D, imgs, labels, opt_G, opt_D, config, device, global_step)
            d_losses.append(ld)
            g_losses.append(lg)
            d_reals.append(dr)
            d_fakes.append(df)
            global_step += 1
            
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
