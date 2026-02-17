#!/usr/bin/env python3
"""
Training Loop Tests for CDCGAN v2

Tests to verify:
1. Single training step completes
2. Losses are finite (no NaN/Inf)
3. Loss values are in reasonable range (0-10 for BCE)
4. Gradients are computed correctly
5. Optimizer steps work
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import pytest
from train_v2 import Generator, Discriminator, train_step
from config import TrainingConfig


@pytest.fixture
def config():
    """Get default configuration"""
    config = TrainingConfig()
    config.batch_size = 4  # Small batch for testing
    return config


@pytest.fixture
def device():
    """Get device (CPU for testing)"""
    return torch.device('cpu')


@pytest.fixture
def models(config, device):
    """Create models"""
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
    
    return G, D


@pytest.fixture
def optimizers(models, config):
    """Create optimizers"""
    G, D = models
    opt_G = optim.Adam(G.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    opt_D = optim.Adam(D.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    return opt_G, opt_D


def test_single_training_step(models, optimizers, config, device):
    """Test that a single training step completes without errors"""
    G, D = models
    opt_G, opt_D = optimizers
    criterion = nn.BCELoss()
    
    # Create fake batch
    batch_size = config.batch_size
    real_imgs = torch.randn(batch_size, config.nc, config.image_size, config.image_size)
    labels = torch.randint(0, config.num_classes, (batch_size,))
    
    # Training step
    d_loss, g_loss, d_real, d_fake = train_step(
        G, D, real_imgs, labels, opt_G, opt_D, criterion, config, device
    )
    
    # Check that losses are returned
    assert isinstance(d_loss, float), "D loss should be float"
    assert isinstance(g_loss, float), "G loss should be float"
    assert isinstance(d_real, float), "D(x) should be float"
    assert isinstance(d_fake, float), "D(G(z)) should be float"


def test_losses_are_finite(models, optimizers, config, device):
    """Test that losses don't contain NaN or Inf"""
    G, D = models
    opt_G, opt_D = optimizers
    criterion = nn.BCELoss()
    
    # Create fake batch
    batch_size = config.batch_size
    real_imgs = torch.randn(batch_size, config.nc, config.image_size, config.image_size)
    labels = torch.randint(0, config.num_classes, (batch_size,))
    
    # Training step
    d_loss, g_loss, d_real, d_fake = train_step(
        G, D, real_imgs, labels, opt_G, opt_D, criterion, config, device
    )
    
    # Check finite
    assert not (d_loss != d_loss), "D loss is NaN"  # NaN != NaN
    assert not (g_loss != g_loss), "G loss is NaN"
    assert abs(d_loss) < float('inf'), "D loss is Inf"
    assert abs(g_loss) < float('inf'), "G loss is Inf"


def test_loss_values_reasonable(models, optimizers, config, device):
    """Test that loss values are in reasonable range for BCE"""
    G, D = models
    opt_G, opt_D = optimizers
    criterion = nn.BCELoss()
    
    # Create fake batch
    batch_size = config.batch_size
    real_imgs = torch.randn(batch_size, config.nc, config.image_size, config.image_size)
    labels = torch.randint(0, config.num_classes, (batch_size,))
    
    # Training step
    d_loss, g_loss, d_real, d_fake = train_step(
        G, D, real_imgs, labels, opt_G, opt_D, criterion, config, device
    )
    
    # BCE loss should typically be in [0, 10] range
    # If we see values like -180 or +137, something is very wrong!
    assert -10 < d_loss < 10, \
        f"D loss out of reasonable range: {d_loss} (should be in [-10, 10])"
    assert -10 < g_loss < 10, \
        f"G loss out of reasonable range: {g_loss} (should be in [-10, 10])"
    
    # D(x) and D(G(z)) should be in [0, 1] for sigmoid output
    assert 0 <= d_real <= 1, \
        f"D(x) out of range: {d_real} (should be in [0, 1])"
    assert 0 <= d_fake <= 1, \
        f"D(G(z)) out of range: {d_fake} (should be in [0, 1])"


def test_gradients_computed(models, optimizers, config, device):
    """Test that gradients are computed correctly"""
    G, D = models
    opt_G, opt_D = optimizers
    criterion = nn.BCELoss()
    
    # Create fake batch
    batch_size = config.batch_size
    real_imgs = torch.randn(batch_size, config.nc, config.image_size, config.image_size)
    labels = torch.randint(0, config.num_classes, (batch_size,))
    
    # Training step
    train_step(G, D, real_imgs, labels, opt_G, opt_D, criterion, config, device)
    
    # Check that at least some gradients exist
    g_has_grads = any(p.grad is not None for p in G.parameters())
    d_has_grads = any(p.grad is not None for p in D.parameters())
    
    assert g_has_grads, "Generator should have gradients after training step"
    assert d_has_grads, "Discriminator should have gradients after training step"


def test_multiple_training_steps(models, optimizers, config, device):
    """Test that multiple training steps work"""
    G, D = models
    opt_G, opt_D = optimizers
    criterion = nn.BCELoss()
    
    losses = []
    
    for _ in range(5):
        # Create fake batch
        batch_size = config.batch_size
        real_imgs = torch.randn(batch_size, config.nc, config.image_size, config.image_size)
        labels = torch.randint(0, config.num_classes, (batch_size,))
        
        # Training step
        d_loss, g_loss, d_real, d_fake = train_step(
            G, D, real_imgs, labels, opt_G, opt_D, criterion, config, device
        )
        
        losses.append((d_loss, g_loss))
    
    # Check that all losses are finite
    for d_loss, g_loss in losses:
        assert abs(d_loss) < float('inf'), "D loss became Inf"
        assert abs(g_loss) < float('inf'), "G loss became Inf"
        assert not (d_loss != d_loss), "D loss became NaN"
        assert not (g_loss != g_loss), "G loss became NaN"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
