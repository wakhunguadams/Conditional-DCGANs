# Getting Started Guide

## Your Project is Ready! 🎉

Everything is set up and tested. You have **525 patches** extracted from the PANDA dataset, ready for training.

## Quick Commands

### 1. Check Your Data
```bash
source venv/bin/activate
python check_data.py
```
This shows dataset statistics and creates `dataset_samples.png` with sample patches.

### 2. Start Training
```bash
# Easy way
./quick_start.sh

# Or manually
source venv/bin/activate
python train.py --epochs 100 --batch_size 8
```

### 3. Generate Synthetic Images (after training)
```bash
source venv/bin/activate
python generate.py --checkpoint checkpoints/G_final.pt --n_per_class 100
```

## What You Have

### ✅ Data
- **525 patches** (256x256 pixels) organized by ISUP grade
- See `dataset_samples.png` for visualization

### ✅ Scripts
- `train.py` - Main training script
- `generate.py` - Generate synthetic images
- `check_data.py` - Verify dataset
- `quick_start.sh` - One-command training

### ✅ Notebook
- `cdcgan_prostate_cancer.ipynb` - Interactive exploration

### ✅ Documentation
- `README.md` - Full documentation
- `PROJECT_SUMMARY.md` - Project overview
- `paper_reference.md` - Research paper notes

## Training Options

### Basic Training
```bash
python train.py
```
Defaults: 100 epochs, batch size 8, lr 0.0002

### Custom Parameters
```bash
# Faster training (fewer epochs)
python train.py --epochs 50

# Smaller batch (if memory issues)
python train.py --batch_size 4

# Different learning rate
python train.py --lr 0.0001

# All together
python train.py --epochs 200 --batch_size 16 --lr 0.0001
```

### Resume Training
```bash
# If interrupted, resume from checkpoint
python train.py --resume checkpoints/ckpt_epoch_0050.pt
```

## Monitoring Training

### During Training
- Watch the progress bar for losses: `D`, `G`, `D(x)`, `D(G(z))`
- Check `./samples/` folder for generated images (every 5 epochs)
- Checkpoints saved every 10 epochs in `./checkpoints/`

### After Training
- View training curves: `samples/history.png`
- Check final samples: `samples/epoch_0100.png`

### What to Look For
- **D(x)** should be around 0.5-0.7 (discriminator on real images)
- **D(G(z))** should be around 0.3-0.5 (discriminator on fake images)
- Generated images should show tissue-like structures
- Different grades should have distinct visual patterns

## Generation Options

### Full Dataset
```bash
# Generate 100 images per grade (600 total)
python generate.py --checkpoint checkpoints/G_final.pt --n_per_class 100
```

### Visualization Only
```bash
# Just create a grid of samples
python generate.py --checkpoint checkpoints/G_final.pt --grid_only
```

### Custom Output
```bash
# Specify output directory
python generate.py --checkpoint checkpoints/G_final.pt \
                   --output_dir ./my_synthetic_data \
                   --n_per_class 200
```

## Expected Timeline

### CPU Training (Current Setup)
- **1 epoch**: ~15 minutes
- **100 epochs**: ~12-15 hours
- **Recommendation**: Run overnight or use GPU

### GPU Training (If Available)
- **1 epoch**: ~1-2 minutes
- **100 epochs**: ~2-4 hours
- **To enable**: Install CUDA PyTorch
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

## Troubleshooting

### "Out of memory"
```bash
# Reduce batch size
python train.py --batch_size 4
```

### "Training too slow"
```bash
# Reduce epochs for testing
python train.py --epochs 20

# Or install GPU PyTorch (if you have NVIDIA GPU)
```

### "Losses not converging"
```bash
# Try lower learning rate
python train.py --lr 0.0001

# Or adjust in code: config.label_smoothing
```

### "Generated images look bad"
- Train longer (100-200 epochs)
- Check if losses are stable
- Ensure D(x) and D(G(z)) are balanced

## File Structure After Training

```
CDCGANs/
├── train.py                      # Training script
├── generate.py                   # Generation script
├── check_data.py                 # Data verification
├── quick_start.sh                # Quick start
├── cdcgan_prostate_cancer.ipynb  # Notebook
├── README.md                     # Full docs
├── PROJECT_SUMMARY.md            # Overview
├── GETTING_STARTED.md            # This file
├── paper_reference.md            # Paper notes
├── dataset_samples.png           # Data visualization
├── venv/                         # Virtual environment
├── panda_data/
│   ├── train.csv
│   ├── train_images/
│   └── patches_256/              # Your 525 patches
│       ├── 0/  (30 patches)
│       ├── 1/  (60 patches)
│       ├── 2/  (105 patches)
│       ├── 3/  (150 patches)
│       ├── 4/  (90 patches)
│       └── 5/  (90 patches)
├── checkpoints/                  # Created during training
│   ├── ckpt_epoch_0010.pt
│   ├── ckpt_epoch_0020.pt
│   ├── ...
│   ├── G_final.pt
│   └── D_final.pt
├── samples/                      # Created during training
│   ├── epoch_0001.png
│   ├── epoch_0005.png
│   ├── ...
│   └── history.png
└── synthetic_data/               # Created by generate.py
    ├── grade_0_benign/
    ├── grade_1_G3+3/
    ├── grade_2_G3+4/
    ├── grade_3_G4+3/
    ├── grade_4_G4+4/
    ├── grade_5_high/
    └── generated_grid.png
```

## Tips for Best Results

1. **Start Small**: Test with `--epochs 20` first to verify everything works
2. **Monitor Early**: Check `samples/epoch_0005.png` to see if model is learning
3. **Be Patient**: Good results typically need 100+ epochs
4. **Save Often**: Checkpoints are saved every 10 epochs automatically
5. **Compare Grades**: Generated images should show different patterns per grade

## Next Steps

1. ✅ **Verify data**: `python check_data.py`
2. 🚀 **Start training**: `./quick_start.sh`
3. 👀 **Monitor progress**: Check `./samples/` folder
4. 📊 **Review results**: Look at `samples/history.png`
5. 🎨 **Generate images**: Use `generate.py`

## Need Help?

- Check `README.md` for detailed documentation
- Review `PROJECT_SUMMARY.md` for architecture details
- Look at `paper_reference.md` for research background
- Examine `cdcgan_prostate_cancer.ipynb` for interactive examples

---

**You're all set!** Run `./quick_start.sh` to begin training. 🚀

The model will learn to generate synthetic prostate cancer histopathology images conditioned on ISUP grade. After training, you'll be able to generate unlimited synthetic samples for data augmentation.

Good luck with your research! 🔬
