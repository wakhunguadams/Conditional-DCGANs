#!/usr/bin/env python3
"""
Simplified CDCGAN Training Script (v2) - Bug Fixes Applied

Key Improvements:
1. Fixed label embedding: 128 dims (was 65,536!)
2. Removed conflicting normalization (Spectral Norm only in D, BatchNorm only in G)
3. Simplified architecture (no self-attention, no residual blocks)
4. Proper label conditioning
5. Stable training configuration (1:1 D:G ratio, BCE loss)
"""
import os
import random
import argparse
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.nn.utils import spectral_norm

from config import TrainingConfig


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PANDADataset(Dataset):
    """PANDA dataset loader with class balancing"""
    
    def __init__(self, patches_dir, transform=None, balance=True):
        self.transform = transform
        self.samples = []
        
        # Load all samples
        for grade in range(6):
            grade_dir = os.path.join(patches_dir, str(grade))
            if os.path.exists(grade_dir):
                for filename in os.listdir(grade_dir):
                    if filename.endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(grade_dir, filename), grade))
        
        # Balance classes if requested
        if balance and self.samples:
            counts = {}
            for _, grade in self.samples:
                counts[grade] = counts.get(grade, 0) + 1
            
            max_count = max(counts.values())
            balanced = []
            
            for grade in range(6):
                grade_samples = [s for s in self.samples if s[1] == grade]
                if grade_samples:
                    # Oversample to match max_count
                    repeats = max_count // len(grade_samples) + 1
                    balanced.extend(grade_samples * repeats)
            
            random.shuffle(balanced)
            self.samples = balanced[:max_count * 6]
        
        print(f'Loaded {len(self.samples)} samples')
        grade_counts = {}
        for _, grade in self.samples:
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
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
    """Get image transformations"""
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
    """
    Simplified Generator with FIXED label embedding
    
    Architecture:
    - Input: [batch, nz + label_embed_dim] concatenated
    - 7 transposed conv layers: 1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
    - BatchNorm + ReLU activations
    - Tanh output
    """
    
    def __init__(self, nz, ngf, nc, num_classes, label_embed_dim):
        super().__init__()
        
        # FIXED: Small label embedding (was img_size * img_size = 65,536!)
        self.label_emb = nn.Embedding(num_classes, label_embed_dim)
        
        input_dim = nz + label_embed_dim
        
        self.main = nn.Sequential(
            # Input: [batch, input_dim, 1, 1]
            nn.ConvTranspose2d(input_dim, ngf*32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*32),
            nn.ReLU(True),
            # State: [batch, ngf*32, 4, 4]
            
            nn.ConvTranspose2d(ngf*32, ngf*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*16),
            nn.ReLU(True),
            # State: [batch, ngf*16, 8, 8]
            
            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # State: [batch, ngf*8, 16, 16]
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # State: [batch, ngf*4, 32, 32]
            
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # State: [batch, ngf*2, 64, 64]
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: [batch, ngf, 128, 128]
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: [batch, nc, 256, 256]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z, labels):
        """
        Forward pass
        
        Args:
            z: noise vector [batch, nz]
            labels: class labels [batch]
        
        Returns:
            Generated images [batch, nc, img_size, img_size]
        """
        # Embed labels
        label_embedding = self.label_emb(labels)  # [batch, label_embed_dim]
        
        # Concatenate noise and label embedding
        x = torch.cat([z, label_embedding], dim=1)  # [batch, nz + label_embed_dim]
        
        # Reshape for conv layers
        x = x.view(x.size(0), -1, 1, 1)  # [batch, nz + label_embed_dim, 1, 1]
        
        return self.main(x)


class Discriminator(nn.Module):
    """
    Simplified Discriminator with FIXED normalization
    
    Architecture:
    - Input: [batch, nc + 1, img_size, img_size] (image + label map)
    - 7 conv layers with Spectral Norm ONLY (no BatchNorm!)
    - LeakyReLU activations
    - Sigmoid output
    """
    
    def __init__(self, ndf, nc, num_classes, img_size, label_embed_dim):
        super().__init__()
        
        self.img_size = img_size
        
        # FIXED: Small label embedding with projection to spatial map
        self.label_emb = nn.Embedding(num_classes, label_embed_dim)
        self.label_proj = nn.Linear(label_embed_dim, img_size * img_size)
        
        # FIXED: Spectral Norm ONLY, no BatchNorm!
        self.main = nn.Sequential(
            # Input: [batch, nc + 1, 256, 256]
            spectral_norm(nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, True),
            # State: [batch, ndf, 128, 128]
            
            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, True),
            # State: [batch, ndf*2, 64, 64]
            
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, True),
            # State: [batch, ndf*4, 32, 32]
            
            spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, True),
            # State: [batch, ndf*8, 16, 16]
            
            spectral_norm(nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, True),
            # State: [batch, ndf*16, 8, 8]
            
            spectral_norm(nn.Conv2d(ndf*16, ndf*16, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, True),
            # State: [batch, ndf*16, 4, 4]
            
            spectral_norm(nn.Conv2d(ndf*16, 1, 4, 1, 0, bias=True)),
            nn.Sigmoid()
            # Output: [batch, 1, 1, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not hasattr(m, 'weight_orig'):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, imgs, labels):
        """
        Forward pass
        
        Args:
            imgs: input images [batch, nc, img_size, img_size]
            labels: class labels [batch]
        
        Returns:
            Discriminator output [batch, 1]
        """
        batch_size = imgs.size(0)
        
        # Embed and project labels to spatial map
        label_embedding = self.label_emb(labels)  # [batch, label_embed_dim]
        label_map = self.label_proj(label_embedding)  # [batch, img_size * img_size]
        label_map = label_map.view(batch_size, 1, self.img_size, self.img_size)
        
        # Concatenate image and label map
        x = torch.cat([imgs, label_map], dim=1)  # [batch, nc + 1, img_size, img_size]
        
        output = self.main(x)
        return output.view(-1, 1)


def train_step(G, D, real_imgs, labels, opt_G, opt_D, criterion, config, device):
    """
    Single training step with 1:1 D:G ratio
    
    Returns:
        Tuple of (d_loss, g_loss, d_real_mean, d_fake_mean)
    """
    batch_size = real_imgs.size(0)
    real_imgs = real_imgs.to(device)
    labels = labels.to(device)
    
    # Labels for BCE loss
    real_label = torch.full((batch_size, 1), 1 - config.label_smoothing, device=device)
    fake_label = torch.zeros(batch_size, 1, device=device)
    
    # ==================
    # Train Discriminator
    # ==================
    D.zero_grad()
    
    # Real images
    output_real = D(real_imgs, labels)
    loss_real = criterion(output_real, real_label)
    
    # Fake images
    z = torch.randn(batch_size, config.nz, device=device)
    fake_imgs = G(z, labels)
    output_fake = D(fake_imgs.detach(), labels)
    loss_fake = criterion(output_fake, fake_label)
    
    # Total discriminator loss
    loss_D = loss_real + loss_fake
    loss_D.backward()
    opt_D.step()
    
    # ==================
    # Train Generator
    # ==================
    G.zero_grad()
    
    # Generate new fake images
    z = torch.randn(batch_size, config.nz, device=device)
    fake_imgs = G(z, labels)
    output = D(fake_imgs, labels)
    
    # Generator loss (fool discriminator)
    loss_G = criterion(output, real_label)
    loss_G.backward()
    opt_G.step()
    
    return (
        loss_D.item(),
        loss_G.item(),
        output_real.mean().item(),
        output_fake.mean().item()
    )


def generate_samples(G, config, device, n_per_class=4):
    """Generate sample images for all classes"""
    G.eval()
    samples = []
    
    with torch.no_grad():
        for grade in range(config.num_classes):
            z = torch.randn(n_per_class, config.nz, device=device)
            labels = torch.full((n_per_class,), grade, dtype=torch.long, device=device)
            samples.append(G(z, labels))
    
    G.train()
    return torch.cat(samples)


def save_sample_grid(samples, epoch, config):
    """Save grid of generated samples"""
    samples = (samples + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    grid = vutils.make_grid(samples.clamp(0, 1), nrow=4, padding=2)
    
    plt.figure(figsize=(12, 18))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title(f'Epoch {epoch}')
    plt.savefig(f'{config.samples_dir}/epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history, config):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['d_loss'], label='D Loss')
    ax1.plot(history['g_loss'], label='G Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training Losses')
    ax1.grid(True, alpha=0.3)
    
    # Discriminator output plot
    ax2.plot(history['d_real'], label='D(x)')
    ax2.plot(history['d_fake'], label='D(G(z))')
    ax2.axhline(0.5, c='r', ls='--', alpha=0.5, label='Target')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Discriminator Output')
    ax2.legend()
    ax2.set_title('Discriminator Outputs')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{config.samples_dir}/training_history.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train CDCGAN v2')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='./panda_data/patches_256', help='Data directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Configuration
    config = TrainingConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.data_dir = args.data_dir
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.samples_dir, exist_ok=True)
    
    # Dataset and dataloader
    transform = get_transforms(config.image_size, augment=True)
    dataset = PANDADataset(config.data_dir, transform, balance=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True
    )
    
    # Models
    G = Generator(
        config.nz,
        config.ngf,
        config.nc,
        config.num_classes,
        config.label_embed_dim
    ).to(device)
    
    D = Discriminator(
        config.ndf,
        config.nc,
        config.num_classes,
        config.image_size,
        config.label_embed_dim
    ).to(device)
    
    print(f'Generator parameters: {sum(p.numel() for p in G.parameters()):,}')
    print(f'Discriminator parameters: {sum(p.numel() for p in D.parameters()):,}')
    
    # Loss and optimizers
    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    opt_D = optim.Adam(D.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    
    # Learning rate schedulers
    sched_G = optim.lr_scheduler.StepLR(opt_G, step_size=50, gamma=0.5)
    sched_D = optim.lr_scheduler.StepLR(opt_D, step_size=50, gamma=0.5)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    history = {
        'd_loss': [],
        'g_loss': [],
        'd_real': [],
        'd_fake': []
    }
    
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        G.load_state_dict(checkpoint['G'])
        D.load_state_dict(checkpoint['D'])
        opt_G.load_state_dict(checkpoint['opt_G'])
        opt_D.load_state_dict(checkpoint['opt_D'])
        start_epoch = checkpoint['epoch'] + 1
        
        # Restore training history if available
        if 'history' in checkpoint:
            history = checkpoint['history']
            print(f'Restored training history: {len(history["d_loss"])} epochs')
        
        # Restore schedulers state
        for _ in range(start_epoch - 1):
            sched_G.step()
            sched_D.step()
    
    print(f'\nStarting training for {config.num_epochs} epochs...')
    print(f'Architecture: Simplified CDCGAN v2')
    print(f'Label embedding: {config.label_embed_dim} dims (FIXED from 65,536!)')
    print(f'Normalization: Spectral Norm (D), BatchNorm (G)')
    print(f'Loss: BCE with label smoothing ({config.label_smoothing})')
    print(f'Training ratio: 1:1 (D:G)\n')
    
    # Training loop
    for epoch in range(start_epoch, config.num_epochs + 1):
        d_losses, g_losses, d_reals, d_fakes = [], [], [], []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs}')
        for imgs, labels in pbar:
            d_loss, g_loss, d_real, d_fake = train_step(
                G, D, imgs, labels, opt_G, opt_D, criterion, config, device
            )
            
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            d_reals.append(d_real)
            d_fakes.append(d_fake)
            
            pbar.set_postfix({
                'D': f'{np.mean(d_losses):.4f}',
                'G': f'{np.mean(g_losses):.4f}',
                'D(x)': f'{np.mean(d_reals):.3f}',
                'D(G(z))': f'{np.mean(d_fakes):.3f}'
            })
        
        # Record epoch metrics
        epoch_d_loss = np.mean(d_losses)
        epoch_g_loss = np.mean(g_losses)
        epoch_d_real = np.mean(d_reals)
        epoch_d_fake = np.mean(d_fakes)
        
        history['d_loss'].append(epoch_d_loss)
        history['g_loss'].append(epoch_g_loss)
        history['d_real'].append(epoch_d_real)
        history['d_fake'].append(epoch_d_fake)
        
        # Print epoch summary
        print(f'Epoch {epoch}/{config.num_epochs} - D: {epoch_d_loss:.4f}, G: {epoch_g_loss:.4f}, D(x): {epoch_d_real:.3f}, D(G(z)): {epoch_d_fake:.3f}')
        
        # Step schedulers
        sched_G.step()
        sched_D.step()
        
        # Generate and save samples
        if epoch % config.sample_interval == 0 or epoch == 1:
            samples = generate_samples(G, config, device)
            save_sample_grid(samples, epoch, config)
        
        # Save checkpoint
        if epoch % config.save_interval == 0:
            checkpoint_path = f'{config.checkpoint_dir}/ckpt_epoch_{epoch:04d}.pt'
            torch.save({
                'epoch': epoch,
                'G': G.state_dict(),
                'D': D.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
                'history': history
            }, checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')
            
            # Save metrics to JSON
            metrics_path = f'{config.checkpoint_dir}/metrics_epoch_{epoch:04d}.json'
            with open(metrics_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            # Also save latest checkpoint
            latest_path = f'{config.checkpoint_dir}/ckpt_latest.pt'
            torch.save({
                'epoch': epoch,
                'G': G.state_dict(),
                'D': D.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
                'history': history
            }, latest_path)
    
    # Save final models
    torch.save(G.state_dict(), f'{config.checkpoint_dir}/G_final.pt')
    torch.save(D.state_dict(), f'{config.checkpoint_dir}/D_final.pt')
    
    # Save final checkpoint
    final_checkpoint_path = f'{config.checkpoint_dir}/ckpt_final.pt'
    torch.save({
        'epoch': config.num_epochs,
        'G': G.state_dict(),
        'D': D.state_dict(),
        'opt_G': opt_G.state_dict(),
        'opt_D': opt_D.state_dict(),
        'history': history
    }, final_checkpoint_path)
    
    # Save final metrics to JSON
    final_metrics_path = f'{config.checkpoint_dir}/metrics_final.json'
    with open(final_metrics_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training history
    plot_training_history(history, config)
    
    print('\nTraining complete!')
    print(f'Final metrics - D: {history["d_loss"][-1]:.4f}, G: {history["g_loss"][-1]:.4f}')
    print(f'D(x): {history["d_real"][-1]:.3f}, D(G(z)): {history["d_fake"][-1]:.3f}')
    print(f'\nCheckpoints saved to: {config.checkpoint_dir}')
    print(f'Samples saved to: {config.samples_dir}')
    print(f'Metrics saved to: {final_metrics_path}')


if __name__ == '__main__':
    main()
