# Quick Reference Guide

## 🎉 Status: First Epoch Complete!

Your cDCGAN has completed 1 epoch. Models are saved and ready to continue training.

## Essential Commands

### Continue Training (Recommended)
```bash
source venv/bin/activate
python train.py --epochs 100 --batch_size 8
```
**Time**: ~12-15 hours on CPU | ~2-4 hours on GPU

### Check Results
```bash
source venv/bin/activate
python analyze_results.py
```

### Generate Synthetic Images
```bash
source venv/bin/activate
python generate.py --checkpoint checkpoints/G_final.pt --n_per_class 100
```

### View Data
```bash
source venv/bin/activate
python check_data.py
```

## Files to View

| File | Description |
|------|-------------|
| `samples/epoch_0001.png` | Generated images after epoch 1 |
| `samples/history.png` | Training loss curves |
| `synthetic_data/generated_grid.png` | Test generation grid |
| `dataset_samples.png` | Original data samples |
| `STATUS.md` | Detailed status report |

## Training Options

| Command | Purpose | Time |
|---------|---------|------|
| `python train.py --epochs 20` | Quick test | ~5 hours |
| `python train.py --epochs 50` | Medium training | ~8 hours |
| `python train.py --epochs 100` | Full training | ~12-15 hours |
| `python train.py --epochs 200` | Extended training | ~24-30 hours |

## Generation Options

| Command | Output |
|---------|--------|
| `--grid_only` | Just visualization grid |
| `--n_per_class 50` | 50 images per grade (300 total) |
| `--n_per_class 100` | 100 images per grade (600 total) |
| `--n_per_class 200` | 200 images per grade (1200 total) |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Too slow | `--batch_size 4` or `--epochs 20` |
| Out of memory | `--batch_size 4` |
| Bad quality | Train more epochs (100+) |
| Losses unstable | `--lr 0.0001` |

## Project Structure

```
✅ Data: 525 patches ready
✅ Models: Generator (203MB) + Discriminator (173MB)
✅ Samples: epoch_0001.png + history.png
✅ Scripts: train.py, generate.py, check_data.py
✅ Docs: README.md, STATUS.md, GETTING_STARTED.md
```

## Next Step

**Continue training for better results:**
```bash
source venv/bin/activate
python train.py --epochs 100 --batch_size 8
```

---

**Need help?** Check `STATUS.md` for detailed information or `README.md` for full documentation.
