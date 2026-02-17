#!/usr/bin/env python3
"""
Architecture Tests for CDCGAN v2

Tests to verify:
1. Generator output shape is correct
2. Discriminator output shape is correct
3. Label embedding dimension is 128 (not 65,536!)
4. Forward pass completes without errors
5. Gradient flow works (no NaN/Inf)
6. Parameter count is reasonable
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from train_v2 import Generator, Discriminator
from config import TrainingConfig


@pytest.fixture
def config():
    """Get default configuration"""
    return TrainingConfig()


@pytest.fixture
def device():
    """Get device (CPU for testing)"""
    return torch.device('cpu')


def test_generator_output_shape(config, device):
    """Test that generator produces correct output shape"""
    G = Generator(
        config.nz,
        config.ngf,
        config.nc,
        config.num_classes,
        config.label_embed_dim
    ).to(device)
    
    batch_size = 4
    z = torch.randn(batch_size, config.nz, device=device)
    labels = torch.randint(0, config.num_classes, (batch_size,), device=device)
    
    output = G(z, labels)
    
    assert output.shape == (batch_size, config.nc, config.image_size, config.image_size), \
        f"Expected shape {(batch_size, config.nc, config.image_size, config.image_size)}, got {output.shape}"
    
    # Check output is in valid range for tanh
    assert output.min() >= -1 and output.max() <= 1, \
        f"Output should be in [-1, 1], got [{output.min():.3f}, {output.max():.3f}]"


def test_discriminator_output_shape(config, device):
    """Test that discriminator produces correct output shape"""
    D = Discriminator(
        config.ndf,
        config.nc,
        config.num_classes,
        config.image_size,
        config.label_embed_dim
    ).to(device)
    
    batch_size = 4
    imgs = torch.randn(batch_size, config.nc, config.image_size, config.image_size, device=device)
    labels = torch.randint(0, config.num_classes, (batch_size,), device=device)
    
    output = D(imgs, labels)
    
    assert output.shape == (batch_size, 1), \
        f"Expected shape {(batch_size, 1)}, got {output.shape}"
    
    # Check output is in valid range for sigmoid
    assert output.min() >= 0 and output.max() <= 1, \
        f"Output should be in [0, 1], got [{output.min():.3f}, {output.max():.3f}]"


def test_label_embedding_dimension(config, device):
    """Test that label embedding is 128 dims (not 65,536!)"""
    G = Generator(
        config.nz,
        config.ngf,
        config.nc,
        config.num_classes,
        config.label_embed_dim
    ).to(device)
    
    # Check embedding dimension
    assert G.label_emb.embedding_dim == config.label_embed_dim, \
        f"Label embedding should be {config.label_embed_dim} dims, got {G.label_emb.embedding_dim}"
    
    # This was the bug: 65,536 dims!
    assert G.label_emb.embedding_dim != config.image_size * config.image_size, \
        "Label embedding should NOT be img_size * img_size!"
    
    # Check total embedding parameters
    total_params = config.num_classes * config.label_embed_dim
    assert total_params == 6 * 128, \
        f"Expected {6 * 128} embedding params, got {total_params}"


def test_forward_pass_no_errors(config, device):
    """Test that forward pass completes without errors"""
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
    
    batch_size = 2
    z = torch.randn(batch_size, config.nz, device=device)
    labels = torch.randint(0, config.num_classes, (batch_size,), device=device)
    
    # Generator forward pass
    fake_imgs = G(z, labels)
    assert not torch.isnan(fake_imgs).any(), "Generator output contains NaN"
    assert not torch.isinf(fake_imgs).any(), "Generator output contains Inf"
    
    # Discriminator forward pass
    d_output = D(fake_imgs, labels)
    assert not torch.isnan(d_output).any(), "Discriminator output contains NaN"
    assert not torch.isinf(d_output).any(), "Discriminator output contains Inf"


def test_gradient_flow(config, device):
    """Test that gradients flow properly (no NaN/Inf)"""
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
    
    batch_size = 2
    z = torch.randn(batch_size, config.nz, device=device)
    labels = torch.randint(0, config.num_classes, (batch_size,), device=device)
    
    # Forward pass
    fake_imgs = G(z, labels)
    d_output = D(fake_imgs, labels)
    
    # Backward pass
    loss = d_output.mean()
    loss.backward()
    
    # Check gradients
    for name, param in G.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in G.{name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in G.{name}"
    
    for name, param in D.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in D.{name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in D.{name}"


def test_parameter_count(config, device):
    """Test that parameter count is reasonable"""
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
    
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    
    # Generator should have reasonable number of params (not too many)
    assert g_params < 100_000_000, \
        f"Generator has too many parameters: {g_params:,}"
    
    # Discriminator should have reasonable number of params
    assert d_params < 100_000_000, \
        f"Discriminator has too many parameters: {d_params:,}"
    
    print(f"\nParameter counts:")
    print(f"  Generator: {g_params:,}")
    print(f"  Discriminator: {d_params:,}")


def test_no_batchnorm_in_discriminator(config, device):
    """Test that discriminator does NOT use BatchNorm (only Spectral Norm)"""
    D = Discriminator(
        config.ndf,
        config.nc,
        config.num_classes,
        config.image_size,
        config.label_embed_dim
    ).to(device)
    
    # Check that there are NO BatchNorm layers in discriminator
    has_batchnorm = any(isinstance(m, torch.nn.BatchNorm2d) for m in D.modules())
    assert not has_batchnorm, "Discriminator should NOT have BatchNorm layers!"


def test_batchnorm_in_generator(config, device):
    """Test that generator DOES use BatchNorm"""
    G = Generator(
        config.nz,
        config.ngf,
        config.nc,
        config.num_classes,
        config.label_embed_dim
    ).to(device)
    
    # Check that there ARE BatchNorm layers in generator
    has_batchnorm = any(isinstance(m, torch.nn.BatchNorm2d) for m in G.modules())
    assert has_batchnorm, "Generator should have BatchNorm layers!"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
