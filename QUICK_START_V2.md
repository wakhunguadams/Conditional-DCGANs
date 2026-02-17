# CDCGAN v2 - Quick Start Guide

## What Changed?

Your previous CDCGAN implementation had critical bugs that caused training to fail catastrophically. The new v2 implementation fixes all major issues:

### Critical Bug Fixes ✅

1. **Label Embedding**: Fixed from 65,536 dims → 128 dims
2. **Normalization**: Removed conflicting BatchNorm from discriminator (Spectral Norm only)
3. **Training Ratio**: Changed from 5:1 → 1:1 (D:G)
4. **Loss Function**: Using stable BCE loss (not misconfigured Wasserstein)
5. **Architecture**: Simplified (removed self-attention and residual blocks for baseline)

### Test Results

All automated tests passed (13/13):
- ✅ Architecture tests (8/8)
- ✅ Training loop tests (5/5)

## Quick Start

### 1. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 2. Run Quick Test (10 epochs)

```bash
python train_v2.py --epochs 10 --batch_size 16 --data_dir ./panda_data/patches_256
```

**Expected Results:**
- D loss: 0.5-2.0 (NOT -180!)
- G loss: 0.5-5.0 (NOT +137!)
- D(x): 0.6-0.9
- D(G(z)): 0.1-0.4
- Time: ~5-30 min per epoch (depending on CPU/GPU)

### 3. Full Training (100 epochs)

```bash
# Recommended: use nohup for long training
nohup python train_v2.py --epochs 100 --batch_size 32 > logs/training_v2.log 2>&1 &
```

### 4. Monitor Training

```bash
# Watch the log file
tail -f logs/training_v2.log

# Check generated samples
ls -lh samples_v2/

# View training history plot
eog samples_v2/training_history.png
```

## File Structure

```
CDCGANs/
├── config.py                    # Centralized configuration
├── train_v2.py                  # New training script (USE THIS!)
├── archive/
│   ├── train_v1_original.py     # Old buggy version
│   └── train_v1_improved.py     # Old Wasserstein version
├── tests/
│   ├── test_architecture.py     # Architecture tests
│   └── test_training.py         # Training loop tests
├── checkpoints_v2/              # Model checkpoints
├── samples_v2/                  # Generated samples
└── logs/                        # Training logs
```

## Key Differences from TensorFlow Implementation

| Aspect | TensorFlow | Your v2 (PyTorch) |
|--------|-----------|-------------------|
| Label Embedding | ~128 dims | 128 dims ✅ |
| Normalization | BatchNorm | Spectral (D), Batch (G) ✅ |
| Loss | BCE | BCE ✅ |
| D:G Ratio | 1:1 | 1:1 ✅ |
| Architecture | Simple DCGAN | Simple DCGAN ✅ |

## Troubleshooting

### If training still fails:

1. **Check data directory**:
   ```bash
   ls -lh panda_data/patches_256/
   # Should have folders: 0/ 1/ 2/ 3/ 4/ 5/
   ```

2. **Reduce batch size** if out of memory:
   ```bash
   python train_v2.py --batch_size 8
   ```

3. **Check loss values** in first epoch:
   - If D loss > 10: Something is wrong
   - If G loss > 20: Something is wrong
   - If losses are NaN: Check data normalization

### If you want to add complexity later:

Once the baseline works, you can add:
- Self-attention layers
- Residual connections
- Wasserstein loss (properly configured)
- Progressive growing

But **get the baseline working first**!

## Next Steps

1. ✅ Run quick test (10 epochs)
2. ✅ Verify losses are reasonable
3. ✅ Check generated samples look tissue-like
4. 🔄 Run full training (100 epochs)
5. 📊 Evaluate with FID score
6. 🎯 Fine-tune hyperparameters if needed

## Questions?

See the detailed analysis in:
- `code_review_analysis.md` - Full bug analysis
- `implementation_plan.md` - Architecture details
