# SSH Deployment Guide

## Quick Deployment via Git + SSH

### Step 1: Prepare Git Repository (Local Machine)

```bash
cd ~/eagle/CDCGANs

# Initialize git (if not already done)
git init

# Add files
git add .
git commit -m "Initial commit: cDCGAN for prostate cancer"

# Push to your remote repository
git remote add origin <your-repo-url>
git push -u origin main
```

### Step 2: SSH to Workstation and Clone

```bash
# SSH to workstation
ssh username@workstation-ip

# Navigate to desired location
cd /path/to/projects/

# Clone repository
git clone <your-repo-url>
cd CDCGANs
```

### Step 3: Setup Environment on Workstation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA (for GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or CPU version if no GPU
# pip install torch torchvision

# Install other dependencies
pip install numpy pandas pillow matplotlib tqdm scikit-image scipy jupyter

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 4: Verify Data

```bash
source venv/bin/activate
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

### Step 5: Start Training

```bash
source venv/bin/activate

# Standard training
python train.py --epochs 200 --batch_size 32

# Or with nohup for long sessions
nohup python train.py --epochs 200 --batch_size 32 > training.log 2>&1 &

# Check progress
tail -f training.log
```

## Essential Commands

### Training Commands

```bash
# Activate environment
source venv/bin/activate

# Basic training
python train.py --epochs 200 --batch_size 32

# Background training (survives logout)
nohup python train.py --epochs 200 --batch_size 32 > training.log 2>&1 &

# Check if running
ps aux | grep train.py

# View log
tail -f training.log

# Kill training if needed
pkill -f train.py
```

### Generation Commands

```bash
# Generate all grades (100 images each)
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 0,1,2,3,4,5 \
  --n_per_grade 100

# Generate specific grades only
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 3,4,5 \
  --n_per_grade 200

# Generate single grade
python generate_by_grade.py \
  --checkpoint checkpoints/G_final.pt \
  --grades 4 \
  --n_per_grade 500
```

### Monitoring Commands

```bash
# Check GPU usage
nvidia-smi
watch -n 1 nvidia-smi

# Check training progress
ls -lht samples/
ls -lht checkpoints/

# Analyze results
python analyze_results.py

# Check disk space
df -h
```

## File Structure

### What's in Git Repository
```
✓ train.py
✓ generate.py
✓ generate_by_grade.py
✓ check_data.py
✓ analyze_results.py
✓ requirements.txt
✓ .gitignore
✓ README.md
✓ *.md (documentation)
✓ panda_data/patches_256/  (your 525 patches)
✓ panda_data/train.csv
```

### What's NOT in Git (Generated on Workstation)
```
✗ venv/
✗ samples/
✗ synthetic_data/
✗ logs/
✗ checkpoints/ (optional - can include if you want)
```

## Recommended Batch Sizes by GPU

| GPU | Memory | Batch Size | Expected Time (200 epochs) |
|-----|--------|------------|---------------------------|
| GTX 1060 | 6 GB | 16 | ~10 hours |
| GTX 1080 | 8 GB | 24 | ~7 hours |
| RTX 2080 | 8 GB | 24 | ~6 hours |
| RTX 3080 | 10 GB | 32 | ~5 hours |
| RTX 3090 | 24 GB | 64 | ~3 hours |
| A100 | 40 GB | 128 | ~2 hours |

## Quick Reference

### One-Time Setup
```bash
ssh username@workstation
cd /path/to/projects/
git clone <your-repo-url>
cd CDCGANs
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas pillow matplotlib tqdm scikit-image scipy
python check_data.py
```

### Start Training

```bash
ssh username@workstation
cd /path/to/projects/CDCGANs
source venv/bin/activate

# Option 1: Using tmux (Recommended - Monitor from anywhere!)
tmux new -s cdcgan
python train.py --epochs 200 --batch_size 32
# Detach: Ctrl+B then D

# Option 2: Background with nohup
nohup python train.py --epochs 200 --batch_size 32 > training.log 2>&1 &
exit
```

### Check Progress (Later)

```bash
ssh username@workstation
cd /path/to/projects/CDCGANs

# If using tmux (reconnect from anywhere!)
tmux attach -t cdcgan

# If using nohup
tail -f training.log
# Or
ls -lht samples/
# Or
source venv/bin/activate && python analyze_results.py
```

### Generate Images (After Training)
```bash
ssh username@workstation
cd /path/to/projects/CDCGANs
source venv/bin/activate
python generate_by_grade.py --checkpoint checkpoints/G_final.pt --grades 0,1,2,3,4,5 --n_per_grade 100
```

### Download Results
```bash
# From your local machine
scp -r username@workstation:/path/to/CDCGANs/checkpoints ./
scp -r username@workstation:/path/to/CDCGANs/samples ./
scp -r username@workstation:/path/to/CDCGANs/synthetic_data ./
```

## Tips

1. **Use `tmux` for persistent sessions (RECOMMENDED)**:
   ```bash
   # Start training
   tmux new -s training
   source venv/bin/activate
   python train.py --epochs 200 --batch_size 32
   # Detach: Ctrl+B then D
   
   # Reconnect anytime from anywhere
   ssh username@workstation
   tmux attach -t training
   ```
   
   **Benefits:**
   - Monitor from home, office, or mobile
   - Split screen for logs + GPU monitoring
   - Survives SSH disconnections
   - See `TMUX_GUIDE.md` for full tutorial

2. **Or use `screen` for persistent sessions**:
   ```bash
   screen -S training
   source venv/bin/activate
   python train.py --epochs 200 --batch_size 32
   # Detach: Ctrl+A then D
   # Reattach later: screen -r training
   ```

2. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Check available GPU memory before training**:
   ```bash
   nvidia-smi --query-gpu=memory.free --format=csv
   ```

4. **Resume interrupted training**:
   ```bash
   python train.py --resume checkpoints/ckpt_epoch_0100.pt --epochs 200
   ```

5. **Generate samples during training** to check progress:
   - Samples saved every 5 epochs in `samples/`
   - View with: `ls -lht samples/`

---

**Ready to deploy!** Push to Git, pull on workstation, and start training. 🚀
