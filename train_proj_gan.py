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

from config import TrainingConfig
from models_proj import StyleGenerator, ProjectedDiscriminator

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

import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser(description='Train Projected GAN')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='./panda_data/patches_256', help='Data directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume checkpoint')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = TrainingConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.checkpoint_dir = './checkpoints_proj'
    config.samples_dir = './samples_diversity'
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.samples_dir, exist_ok=True)

    dataset = PANDADataset(args.data_dir, get_transforms(256), balance=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True)

    G = StyleGenerator(nz=512, style_dim=512).to(device)
    D = ProjectedDiscriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=config.lr, betas=(0.0, 0.99))
    opt_D = optim.Adam(D.parameters(), lr=config.lr*8, betas=(0.0, 0.99)) # Detail Surge: Increased D-lr to force edge formation

    history = {'d_loss': [], 'g_loss': []}
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt['G'])
        D.load_state_dict(ckpt['D'])
        opt_G.load_state_dict(ckpt['opt_G'])
        opt_D.load_state_dict(ckpt['opt_D'])
        start_epoch = ckpt['epoch'] + 1
        history = ckpt.get('history', history)

    for epoch in range(start_epoch, config.num_epochs + 1):
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

            # Train Generator
            opt_G.zero_grad()
            
            # Diversity Loss: G(z1) and G(z2) should be different
            z1 = torch.randn(bs, 512, device=device)
            z2 = torch.randn(bs, 512, device=device)
            fake1 = G(z1, labels) # Reuse some computation if possible, but for diversity we need fresh fakes
            fake2 = G(z2, labels)
            
            d_fake_for_g = D(fake1, labels)
            loss_G_adv = hinge_loss_g(d_fake_for_g)
            
            # Maximize distance between fakes relative to z distance
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
                vutils.save_image(samples, f'{config.samples_dir}/epoch_{epoch:03d}.png', nrow=4)

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'G': G.state_dict(),
                'D': D.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
                'history': history,
            }, f'{config.checkpoint_dir}/ckpt_epoch_{epoch:03d}.pt')

if __name__ == '__main__':
    main()
