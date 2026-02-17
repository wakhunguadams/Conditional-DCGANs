# Conditional DCGAN for Prostate Cancer Image Synthesis (v2)

**Status:** ✅ **Training Successfully - All Critical Bugs Fixed!**

Implementation of a Conditional Deep Convolutional GAN for synthesizing prostate cancer histopathology images from the PANDA dataset.

> [!IMPORTANT]
> **Version 2 Release**: This repository has been completely overhauled to fix critical bugs in the original implementation. See [QUICK_START_V2.md](QUICK_START_V2.md) for details.

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run training (10 epochs test)
python train_v2.py --epochs 10 --batch_size 16

# Run full training (100 epochs)
python train_v2.py --epochs 100 --batch_size 32
```

**See [QUICK_START_V2.md](QUICK_START_V2.md) for complete guide.**

## What's New in v2

### Critical Bug Fixes ✅

1. **Label Embedding**: Fixed from 65,536 dims → **128 dims**
2. **Normalization**: Removed conflicting BatchNorm from discriminator
3. **Training Ratio**: Changed from 5:1 → **1:1** (D:G)
4. **Loss Values**: Now stable (0.5-5.0) instead of catastrophic (-180 to +137)
5. **Architecture**: Simplified baseline (removed self-attention/residual for now)

### Test Results

- ✅ All automated tests passed (13/13)
- ✅ Training stable with reasonable loss values
- ✅ D loss: ~0.7-1.5 (was -184!)
- ✅ G loss: ~2.0-4.0 (was +106!)

## Overview

This project implements a conditional DCGAN that can generate synthetic prostate cancer biopsy images conditioned on ISUP grade (0-5).

### ISUP Grading System
- **Grade 0**: Benign (no cancer)
- **Grade 1**: Gleason 3+3
- **Grade 2**: Gleason 3+4
- **Grade 3**: Gleason 4+3
- **Grade 4**: Gleason 4+4, 3+5, 5+3
- **Grade 5**: Gleason 4+5, 5+4, 5+5

## Installation

```bash
# Clone repository
git clone https://github.com/wakhunguadams/Conditional-DCGANs.git
cd Conditional-DCGANs

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision tqdm matplotlib numpy pandas pillow scikit-image scipy
```

## Usage

### 1. Training

Train the model on extracted patches:

```bash
# Basic training (100 epochs, batch size 8)
python train.py

# Custom parameters
python train.py --epochs 200 --batch_size 16 --lr 0.0001

# Resume from checkpoint
python train.py --resume checkpoints/ckpt_epoch_0050.pt
```

**Training options:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 0.0002)
- `--data_dir`: Path to patches directory (default: ./panda_data/patches_256)
- `--resume`: Resume from checkpoint

**Outputs:**
- Checkpoints saved every 10 epochs in `./checkpoints/`
- Sample images every 5 epochs in `./samples/`
- Training history plot in `./samples/history.png`

### 2. Generate Synthetic Images

After training, generate synthetic images:

```bash
# Generate 100 images per class
python generate.py --checkpoint checkpoints/G_final.pt --n_per_class 100

# Generate only visualization grid
python generate.py --checkpoint checkpoints/G_final.pt --grid_only --grid_samples 8

# Custom output directory
python generate.py --checkpoint checkpoints/G_final.pt --output_dir ./my_synthetic_data
```

**Generation options:**
- `--checkpoint`: Path to generator checkpoint (required)
- `--output_dir`: Output directory (default: ./synthetic_data)
- `--n_per_class`: Images per class (default: 100)
- `--grid_only`: Only generate visualization grid
- `--grid_samples`: Samples per class in grid (default: 8)

### 3. Using the Notebook

For interactive exploration:

```bash
jupyter notebook cdcgan_prostate_cancer.ipynb
```

The notebook includes:
- Data preprocessing and patch extraction
- Model architecture definitions
- Training loop with visualization
- FID score calculation
- Demo mode with synthetic data

## Project Structure

```
.
├── train.py                    # Training script
├── generate.py                 # Generation script
├── cdcgan_prostate_cancer.ipynb # Jupyter notebook
├── paper_reference.md          # Paper notes
├── README.md                   # This file
├── panda_data/
│   ├── train.csv              # Dataset labels
│   ├── train_images/          # Original TIFF images
│   └── patches_256/           # Extracted 256x256 patches
│       ├── 0/                 # Grade 0 patches
│       ├── 1/                 # Grade 1 patches
│       └── ...
├── checkpoints/               # Model checkpoints
├── samples/                   # Generated samples during training
└── synthetic_data/            # Generated synthetic dataset
    ├── grade_0_benign/
    ├── grade_1_G3+3/
    └── ...
```

## Training Tips

1. **Monitor D(x) and D(G(z))**: Should converge around 0.5
2. **Learning rate**: Start with 0.0002, reduce if unstable
3. **Batch size**: Larger is better (8-16 recommended)
4. **Epochs**: 100-200 epochs typically sufficient
5. **Data augmentation**: Enabled by default (flips, rotations, color jitter)

## Expected Results

- **Training time**: ~2-4 hours for 100 epochs (GPU)
- **Loss convergence**: D and G losses should stabilize
- **Visual quality**: Images should show tissue-like structures
- **Grade differentiation**: Different grades should show distinct patterns

## Technical Details

### Features
- Conditional generation by ISUP grade
- Spectral normalization for training stability
- Label smoothing (0.1) to prevent discriminator overfitting
- Data augmentation (flips, rotations, color jitter)
- Class balancing through oversampling
- Learning rate scheduling (decay every 50 epochs)

### Loss Function
- Binary Cross-Entropy (BCE) loss
- Label smoothing for real labels (0.9 instead of 1.0)

### Optimizers
- Adam optimizer for both G and D
- Learning rate: 0.0002
- Betas: (0.5, 0.999)

## Evaluation

The notebook includes FID (Fréchet Inception Distance) score calculation for quantitative evaluation. Lower FID indicates better quality and diversity.

## References

- Paper: "Image Synthesis for Prostate Cancer Biopsies Using Conditional Deep Convolutional Generative Adversarial Network"
- Dataset: PANDA Challenge (Kaggle)
- Architecture: Based on DCGAN with conditional extensions

## Notes

- The reference paper reports an FID score of 1.3, which is reportedly fabricated
- This implementation focuses on reproducible results
- GPU recommended for training (CPU training is very slow)
- Minimum 8GB GPU memory recommended

## License

This project is for educational and research purposes.
# Conditional-DCGANs
