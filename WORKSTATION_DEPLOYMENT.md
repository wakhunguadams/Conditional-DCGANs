# Workstation Deployment Guide

## Overview

This guide will help you transfer your cDCGAN project to a workstation for long-term training with GPU acceleration.

## What You're Transferring

### Essential Files (Must Transfer)
```
✓ train.py                    # Main training script
✓ generate.py                 # Image generation
✓ generate_by_grade.py        # Grade-specific generation (NEW!)
✓ check_data.py               # Data verification
✓ analyze_results.py          # Results analysis
✓ requirements.txt            # Dependencies
✓ setup_workstation.sh        # Setup script (NEW!)
✓ train_workstation.sh        # Training script (NEW!)
✓ panda_data/                 # Your 525 patches
  ├── patches_256/
  │   ├── 0/ (30 patches)
  │   ├── 1/ (60 patches)
  │   ├── 2/ (105 patches)
  │   ├── 3/ (150 patches)
  │   ├── 4/ (90 patches)
  │   └── 5/ (90 patches)
  └── train.csv
```

### Optional Files (Nice to Have)
```
✓ checkpoints/G_final.pt      # Your trained model (1 epoch)
✓ checkpoints/D_final.pt      # Discriminator
✓ README.md                   # Documentation
✓ GETTING_STARTED.md          # Quick start
✓ paper_reference.md          # Research notes
```

### Files NOT to Transfer (Generated/Large)
```
✗ venv/                       # Will recreate on workstation
✗ samples/                    # Will regenerate
✗ synthetic_data/             # Will regenerate
✗ dataset_samples.png         # Can regenerate
✗ *.ipynb_checkpoints/        # Jupyter cache
```

## Transfer Methods

### Method 1: Using rsync (Recommended)

On your current machine:
```bash
# Create transfer package (excludes venv and generated files)
cd ~/eagle/CDCGANs
rsync -avz --progress \
  --exclude 'venv/' \
  --exclude 'samples/' \
  --exclude 'synthetic_data/' \
  --exclude '*.pyc' \
  --exclude '__pycache__/' \
  --exclude '.ipynb_checkpoints/' \
  ./ username@workstation:/path/to/destination/CDCGANs/
```

### Method 2: Using tar + scp

On your current machine:
```bash
# Create compressed archive
cd ~/eagle
tar -czf cdcgan_project.tar.gz \
  --exclude='CDCGANs/venv' \
  --exclude='CDCGANs/samples' \
  --exclude='CDCGANs/synthetic_data' \
  --exclude='CDCGANs/*.pyc' \
  --exclude='CDCGANs/__pycache__' \
  CDCGANs/

# Transfer to workstation
scp cdcgan_project.tar.gz username@workstation:/path/to/destination/

# On workstation, extract:
ssh username@workstation
cd /path/to/destination/
tar -xzf cdcgan_project.tar.gz
```

### Method 3: Using Git (If you have a repo)

```bash
# On current machine
cd ~/eagle/CDCGANs
git init
git add train.py generate*.py check_data.py analyze_results.py
git add requirements.txt *.sh *.md
git add panda_data/train.csv
git add panda_data/patches_256/
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main

# On workstation
git clone <your-repo-url>
cd CDCGANs
```

### Method 4: Using USB Drive

```bash
# Create package
cd ~/eagle
tar -czf cdcgan_project.tar.gz \
  --exclude='CDCGANs/venv' \
  --exclude='CDCGANs/samples' \
  CDCGANs/

# Copy to USB drive
cp cdcgan_project.tar.gz /media/usb/

# On workstation
cp /media/usb/cdcgan_project.tar.gz ~/
tar -xzf cdcgan_project.tar.gz
```

## Workstation Setup

### Step 1: Transfer Files

Use one of the methods above to transfer your project.

### Step 2: Run Setup Script

```bash
cd /path/to/CDCGANs
chmod +x setup_workstation.sh
./setup_workstation.sh
```

This will:
- Create virtual environment
- Detect and install CUDA PyTorch (if GPU available)
- Install all dependencies
- Create necessary directories
- Verify installation

### Step 3: Verify Data

```bash
source venv/bin/activate
python check_data.py
```

Should show:
```
Total patches: 525
Grade 0: 30 patches
Grade 1: 60 patches
Grade 2: 105 patches
Grade 3: 150 patches
Grade 4: 90 patches
Grade 5: 90 patches
```

### Step 4: Test Training (Optional)

Quick test to ensure everything works:
```bash
source venv/bin/activate
python train.py --epochs 2 --batch_size 16
```

## Training on Workstation

### Option 1: Using Training Script (Recommended)

```bash
./train_workstation.sh
```

This will:
- Train for 200 epochs with batch size 32
- Log everything to `logs/training_TIMESTAMP.log`
- Save checkpoints every 10 epochs
- Generate samples every 5 epochs

### Option 2: Custom Training

```bash
source venv/bin/activate

# Standard training (200 epochs)
python train.py --epochs 200 --batch_size 32

# Extended training (500 epochs)
python train.py --epochs 500 --batch_size 32

# With custom learning rate
python train.py --epochs 200 --batch_size 32 --lr 0.0001
```

### Option 3: Background Training with nohup

For long training sessions that survive logout:
```bash
source venv/bin/activate
nohup python train.py --epochs 200 --batch_size 32 > training.log 2>&1 &

# Check progress
tail -f training.log

# Check if still running
ps aux | grep train.py
```

### Option 4: Using screen/tmux

```bash
# Start screen session
screen -S cdcgan_training

# Activate and train
source venv/bin/activate
python train.py --epochs 200 --batch_size 32

# Detach: Ctrl+A then D
# Reattach: screen -r cdcgan_training
```

## Monitoring Training

### Check Progress

```bash
# View latest samples
ls -lht samples/ | head

# View training log
tail -f logs/training_*.log

# Check GPU usage (if available)
watch -n 1 nvidia-smi
```

### Analyze Results

```bash
source venv/bin/activate
python analyze_results.py
```

## Generate Synthetic Images

### Generate All Grades

```bash
source venv/bin/activate
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 0,1,2,3,4,5 \
  --n_per_grade 100
```

### Generate Specific Grades Only

```bash
# Only high-grade cancer (grades 3, 4, 5)
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 3,4,5 \
  --n_per_grade 200

# Only benign and low-grade (grades 0, 1, 2)
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 0,1,2 \
  --n_per_grade 150

# Only specific grade
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 4 \
  --n_per_grade 500
```

### Generation Options

```bash
# Large dataset with GPU batch processing
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 0,1,2,3,4,5 \
  --n_per_grade 1000 \
  --batch_size 64

# Custom output directory
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 0,1,2,3,4,5 \
  --n_per_grade 100 \
  --output_dir ./my_synthetic_dataset
```

## Expected Performance

### CPU vs GPU Training

| Hardware | Batch Size | Time per Epoch | 200 Epochs |
|----------|------------|----------------|------------|
| CPU (Current) | 8 | ~15 min | ~50 hours |
| Workstation CPU | 16 | ~20 min | ~67 hours |
| GPU (GTX 1080) | 32 | ~2 min | ~7 hours |
| GPU (RTX 3090) | 64 | ~1 min | ~3.5 hours |
| GPU (A100) | 128 | ~30 sec | ~1.7 hours |

### Recommended Settings

| GPU Memory | Batch Size | Epochs | Training Time |
|------------|------------|--------|---------------|
| 8 GB | 16 | 200 | ~8-10 hours |
| 12 GB | 32 | 200 | ~5-7 hours |
| 24 GB | 64 | 200 | ~3-4 hours |
| 40+ GB | 128 | 200 | ~2-3 hours |

## Troubleshooting

### Out of Memory on GPU

```bash
# Reduce batch size
python train.py --epochs 200 --batch_size 16

# Or even smaller
python train.py --epochs 200 --batch_size 8
```

### Training Interrupted

```bash
# Resume from last checkpoint
python train.py --resume checkpoints/ckpt_epoch_0100.pt --epochs 200
```

### Check GPU Usage

```bash
# Monitor GPU
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi

# Check if PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Slow Training on GPU

Make sure you're using CUDA PyTorch:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## File Structure on Workstation

After setup and training:
```
CDCGANs/
├── train.py
├── generate.py
├── generate_by_grade.py        # NEW: Grade-specific generation
├── check_data.py
├── analyze_results.py
├── requirements.txt
├── setup_workstation.sh        # NEW: Setup script
├── train_workstation.sh        # NEW: Training script
├── panda_data/
│   ├── patches_256/            # Your 525 patches
│   └── train.csv
├── venv/                       # Created by setup
├── checkpoints/                # Training checkpoints
│   ├── ckpt_epoch_0010.pt
│   ├── ckpt_epoch_0020.pt
│   ├── ...
│   ├── G_final.pt
│   └── D_final.pt
├── samples/                    # Generated during training
│   ├── epoch_0001.png
│   ├── epoch_0005.png
│   ├── ...
│   └── history.png
├── logs/                       # Training logs
│   └── training_TIMESTAMP.log
└── synthetic_data/             # Generated images
    ├── grade_0_benign/
    ├── grade_1_G3+3/
    ├── grade_2_G3+4/
    ├── grade_3_G4+3/
    ├── grade_4_G4+4/
    └── grade_5_high/
```

## Quick Command Reference

```bash
# Setup
./setup_workstation.sh

# Train
./train_workstation.sh

# Or custom training
python train.py --epochs 200 --batch_size 32

# Generate all grades
python generate_by_grade.py --checkpoint checkpoints/G_final.pt --grades 0,1,2,3,4,5 --n_per_grade 100

# Generate specific grades
python generate_by_grade.py --checkpoint checkpoints/G_final.pt --grades 3,4,5 --n_per_grade 200

# Check results
python analyze_results.py

# Monitor GPU
nvidia-smi
```

## Next Steps After Transfer

1. ✅ Transfer files to workstation
2. ✅ Run `./setup_workstation.sh`
3. ✅ Verify data with `python check_data.py`
4. ✅ Start training with `./train_workstation.sh`
5. ✅ Monitor progress in `logs/` and `samples/`
6. ✅ Generate synthetic images with `generate_by_grade.py`

---

**Ready to deploy!** Transfer your project and start long-term training on the workstation. 🚀
