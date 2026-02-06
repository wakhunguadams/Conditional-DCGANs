# Complete Workstation Setup Guide

## 🚀 Full Setup from Scratch

This guide covers everything from cloning the repository to starting training on your workstation.

## Step 1: Clone Repository

```bash
# SSH to workstation
ssh adamswakhungu@workstation-ip

# Navigate to projects directory
cd /path/to/projects/

# Clone repository
git clone <your-repo-url>
cd CDCGANs
```

## Step 2: Setup Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Step 3: Install Dependencies

### For GPU (Recommended)
```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### For CPU (Fallback)
```bash
pip install torch torchvision
```

### Install Other Dependencies
```bash
pip install numpy pandas pillow matplotlib tqdm scikit-image scipy jupyter kaggle
```

## Step 4: Verify Installation

```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check GPU memory
nvidia-smi
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True
GPU: NVIDIA GeForce RTX 3090 (or your GPU)
```

## Step 5: Download PANDA Dataset

### Option A: Automated Download (Recommended)

```bash
# The kaggle_credentials.json is already in the repo
# Just run the download script
python download_panda_data.py
```

This will:
1. Setup Kaggle credentials
2. Install Kaggle API
3. Download dataset (~100 GB)
4. Extract files
5. Verify download

### Option B: Manual Download

```bash
# Setup Kaggle credentials
mkdir -p ~/.kaggle
cp kaggle_credentials.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Install Kaggle API
pip install kaggle

# Download dataset
kaggle competitions download -c prostate-cancer-grade-assessment -p ./panda_data

# Extract
cd panda_data
unzip train.csv.zip
unzip train_images.zip
cd ..
```

## Step 6: Extract Patches

```bash
# Extract patches (this will take 2-4 hours)
python extract_patches.py

# Or with custom settings for larger dataset
python extract_patches.py --max_per_grade 1000 --max_patches 30
```

Options:
- `--patch_size 256` - Patch size (default: 256)
- `--max_patches 20` - Max patches per image (default: 20)
- `--max_per_grade 500` - Max images per grade (default: 500)
- `--tissue_thresh 0.5` - Tissue threshold (default: 0.5)

## Step 7: Verify Data

```bash
python check_data.py
```

Expected output:
```
============================================================
PANDA Dataset Statistics
============================================================

Grade 0: Benign (no cancer)
  Patches: XXXX

Grade 1: Gleason 3+3
  Patches: XXXX

... (for all grades)

Total patches: XXXXX
============================================================
```

## Step 8: Create Directories

```bash
mkdir -p checkpoints samples synthetic_data logs
```

## Step 9: Start Training

### Option A: Using Tmux (Recommended - Monitor from Anywhere!)

```bash
# Install tmux if not available
sudo apt-get install tmux  # Ubuntu/Debian
# or
sudo yum install tmux      # CentOS/RHEL

# Start tmux session
tmux new -s cdcgan

# Activate environment and train
source venv/bin/activate
python train.py --epochs 200 --batch_size 32

# Detach: Ctrl+B then D
# You can now disconnect - training continues!

# Reconnect anytime from anywhere:
ssh adamswakhungu@workstation-ip
tmux attach -t cdcgan
```

**Why tmux?**
- ✅ Monitor training from home, office, or mobile
- ✅ Split screen to watch logs and GPU simultaneously
- ✅ Resume exactly where you left off
- ✅ Training survives SSH disconnections

See `TMUX_GUIDE.md` for complete tmux tutorial!

### Option B: Tmux with Monitoring Dashboard

```bash
# Start session
tmux new -s cdcgan

# Split into 4 panes for full monitoring
Ctrl+B "    # Split horizontally
Ctrl+B %    # Split vertically (top)
Ctrl+B ↓    # Move down
Ctrl+B %    # Split vertically (bottom)

# Pane 0 (top-left): Training
cd /path/to/CDCGANs
source venv/bin/activate
python train.py --epochs 200 --batch_size 32

# Pane 1 (top-right): Training log
Ctrl+B →
tail -f logs/training.log

# Pane 2 (bottom-left): GPU monitoring
Ctrl+B ↓
watch -n 1 nvidia-smi

# Pane 3 (bottom-right): Samples
Ctrl+B →
watch -n 10 'ls -lht samples/ | head'

# Detach: Ctrl+B D
```

### Option C: Background Training (Simple)

```bash
# Start training in background
nohup python train.py --epochs 200 --batch_size 32 > logs/training.log 2>&1 &

# Check if running
ps aux | grep train.py

# Monitor progress
tail -f logs/training.log
```

### Option D: Using screen (Alternative)

```bash
# Start screen session
screen -S cdcgan_training

# Activate environment and train
source venv/bin/activate
python train.py --epochs 200 --batch_size 32

# Detach: Ctrl+A then D
# Reattach later: screen -r cdcgan_training
```

## Training Configuration

### Recommended Settings by GPU

| GPU | Memory | Batch Size | Epochs | Expected Time |
|-----|--------|------------|--------|---------------|
| GTX 1060 | 6 GB | 16 | 200 | ~10 hours |
| GTX 1080 | 8 GB | 24 | 200 | ~7 hours |
| RTX 2080 | 8 GB | 24 | 200 | ~6 hours |
| RTX 3080 | 10 GB | 32 | 200 | ~5 hours |
| RTX 3090 | 24 GB | 64 | 200 | ~3 hours |
| A100 | 40 GB | 128 | 200 | ~2 hours |

### Custom Training

```bash
# Extended training
python train.py --epochs 500 --batch_size 32

# Lower learning rate
python train.py --epochs 200 --batch_size 32 --lr 0.0001

# Resume from checkpoint
python train.py --resume checkpoints/ckpt_epoch_0100.pt --epochs 200
```

## Monitoring Training

### Check Progress

```bash
# View training log
tail -f logs/training.log

# Check latest samples
ls -lht samples/ | head

# Check checkpoints
ls -lht checkpoints/

# GPU usage
watch -n 1 nvidia-smi
```

### Analyze Results

```bash
source venv/bin/activate
python analyze_results.py
```

## Generate Synthetic Images

### After Training Completes

```bash
source venv/bin/activate

# Generate all grades (100 images each)
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 0,1,2,3,4,5 \
  --n_per_grade 100

# Generate specific grades
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 3,4,5 \
  --n_per_grade 200

# Generate large dataset
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 0,1,2,3,4,5 \
  --n_per_grade 1000 \
  --batch_size 64
```

## Download Results to Local Machine

From your local machine:

```bash
# Download checkpoints
scp -r adamswakhungu@workstation:/path/to/CDCGANs/checkpoints ./

# Download samples
scp -r adamswakhungu@workstation:/path/to/CDCGANs/samples ./

# Download generated images
scp -r adamswakhungu@workstation:/path/to/CDCGANs/synthetic_data ./

# Download logs
scp adamswakhungu@workstation:/path/to/CDCGANs/logs/training.log ./
```

## Complete File Structure

After full setup:

```
CDCGANs/
├── train.py                      # Training script
├── generate_by_grade.py          # Grade-specific generation
├── download_panda_data.py        # Dataset downloader
├── extract_patches.py            # Patch extraction
├── check_data.py                 # Data verification
├── analyze_results.py            # Results analysis
├── kaggle_credentials.json       # Kaggle API credentials
├── requirements.txt              # Dependencies
├── .gitignore                    # Git ignore rules
├── DATA_PREPARATION.md           # Data prep guide
├── WORKSTATION_SETUP.md          # This file
├── venv/                         # Virtual environment
├── panda_data/
│   ├── train.csv                 # Labels
│   ├── train_images/             # Original TIFF files (~100 GB)
│   └── patches_256/              # Extracted patches
│       ├── 0/                    # Grade 0 patches
│       ├── 1/                    # Grade 1 patches
│       ├── 2/                    # Grade 2 patches
│       ├── 3/                    # Grade 3 patches
│       ├── 4/                    # Grade 4 patches
│       └── 5/                    # Grade 5 patches
├── checkpoints/                  # Model checkpoints
│   ├── ckpt_epoch_0010.pt
│   ├── ckpt_epoch_0020.pt
│   ├── ...
│   ├── G_final.pt
│   └── D_final.pt
├── samples/                      # Generated samples
│   ├── epoch_0001.png
│   ├── epoch_0005.png
│   ├── ...
│   └── history.png
├── logs/                         # Training logs
│   └── training.log
└── synthetic_data/               # Generated images
    ├── grade_0_benign/
    ├── grade_1_G3+3/
    ├── grade_2_G3+4/
    ├── grade_3_G4+3/
    ├── grade_4_G4+4/
    └── grade_5_high/
```

## Troubleshooting

### Issue: CUDA Out of Memory
```bash
# Reduce batch size
python train.py --epochs 200 --batch_size 16
```

### Issue: Kaggle Authentication Failed
```bash
# Check credentials
cat ~/.kaggle/kaggle.json

# Fix permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Issue: Training Interrupted
```bash
# Resume from last checkpoint
python train.py --resume checkpoints/ckpt_epoch_0100.pt --epochs 200
```

### Issue: Slow Training
```bash
# Check if using GPU
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall CUDA PyTorch if needed
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Clone & Setup | 10-15 min | One-time |
| Download Dataset | 1-3 hours | Depends on internet |
| Extract Patches | 2-4 hours | Depends on CPU |
| Training (GPU) | 3-7 hours | 200 epochs |
| Generation | 5-10 min | 600 images |

## Storage Requirements

| Component | Size |
|-----------|------|
| Repository | ~50 MB |
| PANDA Dataset | ~100 GB |
| Extracted Patches | ~5-10 GB |
| Checkpoints | ~400 MB |
| Samples | ~100 MB |
| Synthetic Data | ~1-5 GB |
| **Total** | **~105-115 GB** |

## Quick Command Reference

```bash
# Setup
git clone <repo>
cd CDCGANs
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas pillow matplotlib tqdm scikit-image scipy kaggle

# Download data
python download_panda_data.py
python extract_patches.py
python check_data.py

# Train
nohup python train.py --epochs 200 --batch_size 32 > logs/training.log 2>&1 &

# Monitor
tail -f logs/training.log
nvidia-smi

# Generate
python generate_by_grade.py --checkpoint checkpoints/G_final.pt --grades 0,1,2,3,4,5 --n_per_grade 100

# Download results
scp -r adamswakhungu@workstation:/path/to/CDCGANs/synthetic_data ./
```

---

**Complete setup guide!** Follow these steps to get everything running on your workstation. 🚀
