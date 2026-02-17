#!/usr/bin/env python3
"""
Centralized Configuration for CDCGAN Training
"""
import os
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for CDCGAN training"""
    
    # Data settings
    image_size: int = 256
    num_classes: int = 6
    batch_size: int = 32
    num_workers: int = 4
    
    # Architecture settings
    nz: int = 128  # noise dimension (latent vector size)
    nc: int = 3    # number of image channels (RGB)
    ngf: int = 32  # generator base number of filters
    ndf: int = 64  # discriminator base number of filters
    label_embed_dim: int = 128  # FIXED: was 65,536!
    
    # Training hyperparameters
    num_epochs: int = 100
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    label_smoothing: float = 0.1
    
    # Paths
    data_dir: str = './panda_data/patches_256'
    checkpoint_dir: str = './checkpoints_v2'
    samples_dir: str = './samples_v2'
    log_dir: str = './logs'
    
    # Monitoring
    save_interval: int = 10
    sample_interval: int = 5
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


# Default configuration instance
default_config = TrainingConfig()
