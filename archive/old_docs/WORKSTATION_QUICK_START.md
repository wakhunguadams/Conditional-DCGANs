# Workstation Quick Start Guide

## 🚀 Fast Track: Get Data on Workstation

### Prerequisites
1. SSH access to workstation
2. Kaggle API credentials

---

## Method 1: Automated Setup (Easiest) ⭐

```bash
# 1. Copy credentials to workstation
scp kaggle_credentials.json user@workstation:/path/to/project/

# 2. SSH into workstation
ssh user@workstation

# 3. Navigate to project
cd /path/to/project

# 4. Run automated setup
bash setup_workstation_data.sh
```

**That's it!** The script will:
- Download ~100 GB dataset
- Extract 1500 images per grade
- Extract 35 patches per image
- Verify everything

---

## Method 2: Manual Setup (More Control)

### Step 1: Transfer Credentials
```bash
scp kaggle_credentials.json user@workstation:/path/to/project/
```

### Step 2: SSH and Setup Environment
```bash
ssh user@workstation
cd /path/to/project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install kaggle
```

### Step 3: Download Dataset
```bash
python download_panda_data.py
```

### Step 4: Extract Patches

**Option A: Standard (slower but simple)**
```bash
python extract_patches.py --max_per_grade 1500 --max_patches 35
```

**Option B: Parallel (faster, uses all CPU cores)**
```bash
python extract_patches_parallel.py --max_per_grade 1500 --max_patches 35
```

### Step 5: Verify
```bash
python check_data.py
```

---

## Data Size Configurations

### Small (Like Current Setup)
```bash
python extract_patches.py --max_per_grade 500 --max_patches 20
```
- **Patches**: ~60,000
- **Disk**: ~4 GB
- **Time**: ~30 min

### Medium (Recommended for Workstation)
```bash
python extract_patches_parallel.py --max_per_grade 1500 --max_patches 35
```
- **Patches**: ~315,000
- **Disk**: ~20 GB
- **Time**: ~45 min (parallel)

### Large (Maximum Data)
```bash
python extract_patches_parallel.py --max_per_grade 999999 --max_patches 50
```
- **Patches**: ~600,000+
- **Disk**: ~40 GB
- **Time**: ~2 hours (parallel)

---

## After Data Download

### Start Training
```bash
# In tmux session (recommended)
tmux new -s training
python train_improved.py --epochs 200 --batch_size 64

# Detach: Ctrl+B then D
# Reattach: tmux attach -t training
```

### Monitor Progress
```bash
# Watch training log
tail -f logs/training_*.log

# Check GPU usage
nvidia-smi

# Check samples
ls -lh samples_improved/
```

---

## Troubleshooting

### "Kaggle API not authenticated"
```bash
# Check credentials exist
cat ~/.kaggle/kaggle.json

# If not, copy again
cp kaggle_credentials.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### "Out of disk space"
```bash
# Check space
df -h

# Use smaller configuration
python extract_patches.py --max_per_grade 500 --max_patches 15
```

### "Download too slow"
```bash
# Download can take 1-3 hours depending on connection
# Be patient or run overnight
```

### "Extraction too slow"
```bash
# Use parallel version
python extract_patches_parallel.py --max_per_grade 1500 --max_patches 35 --workers 8
```

---

## Quick Commands Reference

```bash
# Check data status
python check_data.py

# Count patches
find panda_data/patches_256 -name "*.png" | wc -l

# Check disk usage
du -sh panda_data/

# List tmux sessions
tmux ls

# Attach to training session
tmux attach -t training

# Kill tmux session
tmux kill-session -t training
```

---

## Time Estimates

| Task | Time (Standard) | Time (Parallel) |
|------|----------------|-----------------|
| Download dataset | 1-3 hours | 1-3 hours |
| Extract 500/grade | 30 min | 15 min |
| Extract 1500/grade | 1.5 hours | 30 min |
| Extract all data | 3 hours | 1 hour |

---

## What You Get

After setup, you'll have:

```
panda_data/
├── train.csv                    # Metadata for all images
├── train_images/                # ~10,000 TIFF files (~100 GB)
└── patches_256/                 # Extracted patches
    ├── 0/                       # Grade 0 patches
    ├── 1/                       # Grade 1 patches
    ├── 2/                       # Grade 2 patches
    ├── 3/                       # Grade 3 patches
    ├── 4/                       # Grade 4 patches
    └── 5/                       # Grade 5 patches
```

---

## Next Steps

1. ✅ Download data (you're here)
2. 📊 Verify data: `python check_data.py`
3. 🚂 Start training: `python train_improved.py --epochs 200 --batch_size 64`
4. 📈 Monitor progress: `tmux attach -t training`
5. 🎨 Generate samples: `python generate_by_grade.py`

---

## Need Help?

- **Full guide**: See `WORKSTATION_DATA_DOWNLOAD.md`
- **Training guide**: See `TRAINING_COMMANDS.md`
- **Tmux guide**: See `TMUX_GUIDE.md`
- **Deployment**: See `WORKSTATION_DEPLOYMENT.md`
