# Deployment Checklist

## ✅ Pre-Deployment (Current Machine)

### 1. Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit: cDCGAN for prostate cancer image synthesis"
```

### 2. Push to Remote Repository
```bash
# Add your remote (GitHub, GitLab, etc.)
git remote add origin <your-repo-url>
git push -u origin main
```

### 3. Verify What's Being Pushed
```bash
git status
git log --oneline
```

**Files included in repository:**
- ✅ `train.py` - Main training script
- ✅ `generate.py` - Image generation
- ✅ `generate_by_grade.py` - Grade-specific generation
- ✅ `check_data.py` - Data verification
- ✅ `analyze_results.py` - Results analysis
- ✅ `requirements.txt` - Dependencies
- ✅ `.gitignore` - Git ignore rules
- ✅ All documentation (*.md files)
- ✅ `panda_data/patches_256/` - Your 525 patches
- ✅ `panda_data/train.csv` - Labels

**Files excluded (via .gitignore):**
- ❌ `venv/` - Will recreate on workstation
- ❌ `samples/` - Will regenerate
- ❌ `synthetic_data/` - Will regenerate
- ❌ `__pycache__/` - Python cache

## ✅ Deployment (Workstation via SSH)

### 1. SSH to Workstation
```bash
ssh username@workstation-ip
```

### 2. Clone Repository
```bash
cd /path/to/projects/
git clone <your-repo-url>
cd CDCGANs
```

### 3. Setup Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 4. Install PyTorch (GPU Version)
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install Other Dependencies
```bash
pip install numpy pandas pillow matplotlib tqdm scikit-image scipy jupyter
```

### 6. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 7. Verify Data
```bash
python check_data.py
```

Expected output:
```
Total patches: 525
Grade 0: 30 patches
Grade 1: 60 patches
Grade 2: 105 patches
Grade 3: 150 patches
Grade 4: 90 patches
Grade 5: 90 patches
```

### 8. Create Directories
```bash
mkdir -p checkpoints samples synthetic_data logs
```

## ✅ Training

### Option 1: Interactive Training
```bash
source venv/bin/activate
python train.py --epochs 200 --batch_size 32
```

### Option 2: Background Training (Recommended)
```bash
source venv/bin/activate
nohup python train.py --epochs 200 --batch_size 32 > logs/training.log 2>&1 &
```

### Option 3: Using screen (Persistent Session)
```bash
screen -S cdcgan
source venv/bin/activate
python train.py --epochs 200 --batch_size 32
# Detach: Ctrl+A then D
# Reattach: screen -r cdcgan
```

## ✅ Monitoring

### Check Training Progress
```bash
# View log
tail -f logs/training.log

# Check samples
ls -lht samples/

# Check checkpoints
ls -lht checkpoints/

# GPU usage
nvidia-smi
watch -n 1 nvidia-smi
```

### Analyze Results
```bash
source venv/bin/activate
python analyze_results.py
```

## ✅ Generation (After Training)

### Generate All Grades
```bash
source venv/bin/activate
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 0,1,2,3,4,5 \
  --n_per_grade 100
```

### Generate Specific Grades
```bash
# High-grade cancer only (3, 4, 5)
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 3,4,5 \
  --n_per_grade 200

# Single grade
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 4 \
  --n_per_grade 500
```

## ✅ Download Results (From Local Machine)

### Download Checkpoints
```bash
scp -r username@workstation:/path/to/CDCGANs/checkpoints ./
```

### Download Samples
```bash
scp -r username@workstation:/path/to/CDCGANs/samples ./
```

### Download Generated Images
```bash
scp -r username@workstation:/path/to/CDCGANs/synthetic_data ./
```

### Download Logs
```bash
scp username@workstation:/path/to/CDCGANs/logs/training.log ./
```

## Quick Command Reference

### SSH Commands
```bash
# Connect
ssh username@workstation

# Copy files to workstation
scp file.txt username@workstation:/path/

# Copy files from workstation
scp username@workstation:/path/file.txt ./

# Copy directories
scp -r username@workstation:/path/dir ./
```

### Training Commands
```bash
# Activate environment
source venv/bin/activate

# Train
python train.py --epochs 200 --batch_size 32

# Background training
nohup python train.py --epochs 200 --batch_size 32 > logs/training.log 2>&1 &

# Check if running
ps aux | grep train.py

# Kill training
pkill -f train.py
```

### Generation Commands
```bash
# All grades
python generate_by_grade.py --checkpoint checkpoints/G_final.pt --grades 0,1,2,3,4,5 --n_per_grade 100

# Specific grades
python generate_by_grade.py --checkpoint checkpoints/G_final.pt --grades 3,4,5 --n_per_grade 200

# Single grade
python generate_by_grade.py --checkpoint checkpoints/G_final.pt --grades 4 --n_per_grade 500
```

## Troubleshooting

### Issue: Out of Memory
**Solution:**
```bash
python train.py --epochs 200 --batch_size 16  # Reduce batch size
```

### Issue: CUDA Not Available
**Solution:**
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Training Interrupted
**Solution:**
```bash
# Resume from checkpoint
python train.py --resume checkpoints/ckpt_epoch_0100.pt --epochs 200
```

### Issue: Can't Find GPU
**Solution:**
```bash
# Check GPU
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
```

## Expected Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Setup | 10-15 min | One-time setup |
| Training (GPU) | 3-7 hours | 200 epochs, batch size 32 |
| Generation | 5-10 min | 600 images (100 per grade) |
| Download | 5-10 min | Depends on network speed |

## Final Checklist

- [ ] Git repository initialized and pushed
- [ ] SSH access to workstation verified
- [ ] Repository cloned on workstation
- [ ] Virtual environment created
- [ ] PyTorch with CUDA installed
- [ ] Dependencies installed
- [ ] Data verified (525 patches)
- [ ] GPU detected and working
- [ ] Training started
- [ ] Monitoring setup (logs, samples)
- [ ] Results downloaded

---

**You're ready to deploy!** Follow the steps above to train on your workstation. 🚀
