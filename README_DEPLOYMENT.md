# Quick Deployment Guide

## 🚀 Deploy to Workstation in 3 Steps

### Step 1: Push to Git (Current Machine)
```bash
git init
git add .
git commit -m "cDCGAN for prostate cancer"
git remote add origin <your-repo-url>
git push -u origin main
```

### Step 2: Pull on Workstation (via SSH)
```bash
ssh username@workstation
cd /path/to/projects/
git clone <your-repo-url>
cd CDCGANs
```

### Step 3: Setup and Train
```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas pillow matplotlib tqdm scikit-image scipy

# Verify
python check_data.py

# Train
nohup python train.py --epochs 200 --batch_size 32 > logs/training.log 2>&1 &
```

## 📊 Generate Images by Grade

### All Grades (100 each = 600 total)
```bash
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 0,1,2,3,4,5 \
  --n_per_grade 100
```

### Specific Grades Only
```bash
# High-grade cancer (grades 3, 4, 5)
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 3,4,5 \
  --n_per_grade 200

# Low-grade (grades 0, 1, 2)
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 0,1,2 \
  --n_per_grade 150

# Single grade (e.g., grade 4)
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 4 \
  --n_per_grade 500
```

### Output Structure
```
synthetic_data/
├── grade_0_benign/
│   ├── syn_grade0_00000.png
│   ├── syn_grade0_00001.png
│   └── ...
├── grade_1_G3+3/
├── grade_2_G3+4/
├── grade_3_G4+3/
├── grade_4_G4+4/
└── grade_5_high/
```

## 📁 What's Included

**Core Scripts:**
- `train.py` - Training
- `generate_by_grade.py` - Grade-specific generation ⭐ NEW
- `generate.py` - General generation
- `check_data.py` - Data verification
- `analyze_results.py` - Results analysis

**Data:**
- `panda_data/patches_256/` - 525 patches across 6 grades
- `panda_data/train.csv` - Labels

**Documentation:**
- `SSH_DEPLOYMENT.md` - Detailed SSH deployment guide
- `DEPLOYMENT_CHECKLIST.md` - Step-by-step checklist
- `README.md` - Full documentation
- `GETTING_STARTED.md` - Quick start guide

## 🎯 Key Features

### Grade-Specific Generation
Generate synthetic images for any combination of ISUP grades:
- **Grade 0**: Benign (no cancer)
- **Grade 1**: Gleason 3+3
- **Grade 2**: Gleason 3+4
- **Grade 3**: Gleason 4+3
- **Grade 4**: Gleason 4+4, 3+5, 5+3
- **Grade 5**: Gleason 4+5, 5+4, 5+5

### Flexible Training
- Adjustable epochs, batch size, learning rate
- Automatic checkpointing every 10 epochs
- Sample generation every 5 epochs
- Resume from checkpoint support

### GPU Optimized
- Automatic CUDA detection
- Batch processing for fast generation
- Memory-efficient training

## 📖 Documentation

| File | Purpose |
|------|---------|
| `SSH_DEPLOYMENT.md` | Complete SSH deployment guide |
| `DEPLOYMENT_CHECKLIST.md` | Step-by-step deployment checklist |
| `README.md` | Full technical documentation |
| `GETTING_STARTED.md` | Quick start guide |
| `STATUS.md` | Current project status |

## 💡 Quick Tips

1. **Check GPU before training:**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Monitor training:**
   ```bash
   tail -f logs/training.log
   watch -n 1 nvidia-smi
   ```

3. **Use screen for persistent sessions:**
   ```bash
   screen -S training
   # Detach: Ctrl+A then D
   # Reattach: screen -r training
   ```

4. **Download results:**
   ```bash
   scp -r username@workstation:/path/to/CDCGANs/synthetic_data ./
   ```

---

**Ready to deploy!** See `SSH_DEPLOYMENT.md` for detailed instructions. 🚀
