# 🎮 Training Control Commands

## Current Training Status
- **Status**: ✅ RUNNING
- **Process ID**: 4
- **Architecture**: Improved (Self-attention + Residual + Wasserstein)
- **Target**: 200 epochs
- **Estimated Time**: ~40 hours

---

## 📊 Check Training Progress

### View Current Output
Ask Kiro to check the process output:
```
"Check the training process output"
```

### View Generated Samples
```bash
ls -lh samples_improved/
```

### Check Latest Sample
```bash
ls -lht samples_improved/epoch_*.png | head -1
```

### View Training Curves
```bash
ls -lh samples_improved/history.png
```

---

## 🔍 Monitor Commands

### Check Process Status
Ask Kiro:
```
"Is the training still running?"
"Show me the latest training output"
```

### Check Disk Space
```bash
df -h .
```

### Count Completed Epochs
```bash
ls samples_improved/epoch_*.png | wc -l
```

### View Checkpoints
```bash
ls -lh checkpoints_improved/
```

---

## ⏸️ Control Training

### Stop Training (if needed)
Ask Kiro:
```
"Stop the training process"
```

Or manually:
```bash
# Find process
ps aux | grep train_improved

# Kill process
kill <PID>
```

### Resume Training
```bash
# Resume from last checkpoint
python train_improved.py --epochs 200 --batch_size 16 \
    --resume checkpoints_improved/ckpt_epoch_XXXX.pt
```

---

## 📈 Progress Milestones

### Check at These Epochs
- **Epoch 10** (~2 hours): Basic patterns
- **Epoch 20** (~4 hours): Clear structures
- **Epoch 50** (~10 hours): Good quality
- **Epoch 100** (~20 hours): Very good quality
- **Epoch 200** (~40 hours): Excellent quality

### How to Check Current Epoch
Ask Kiro:
```
"What epoch is the training at?"
"Show me the latest training output"
```

---

## 🎯 After Training Completes

### Generate Synthetic Images
```bash
source venv/bin/activate
python generate_by_grade.py --checkpoint checkpoints_improved/G_final.pt \
    --grades 0,1,2,3,4,5 --n_per_grade 100
```

### Analyze Results
```bash
python analyze_results.py
```

### Compare Quality
```bash
# View old training (epoch 10)
ls -lh samples/epoch_0010.png

# View new training (epoch 10)
ls -lh samples_improved/epoch_0010.png

# View final result (epoch 200)
ls -lh samples_improved/epoch_0200.png
```

---

## 🚨 Troubleshooting

### If Training Stops
1. Check process status
2. Look for error messages
3. Check disk space
4. Resume from last checkpoint

### If Out of Memory
```bash
# Restart with smaller batch size
python train_improved.py --epochs 200 --batch_size 8 \
    --resume checkpoints_improved/ckpt_epoch_XXXX.pt
```

### If Quality Seems Poor
- **Wait**: Need at least 50-100 epochs
- **Check samples**: Look at epoch 50, 100, 200
- **Compare**: With old training at same epoch

---

## 📞 Quick Reference

| Action | Command |
|--------|---------|
| Check status | Ask Kiro: "Show training output" |
| View samples | `ls -lh samples_improved/` |
| View checkpoints | `ls -lh checkpoints_improved/` |
| Check epoch | Ask Kiro: "What epoch?" |
| Stop training | Ask Kiro: "Stop training" |
| Resume training | `python train_improved.py --resume ...` |

---

## ⏱️ Time Estimates

| Epochs | Time (CPU) | Time (GPU) |
|--------|------------|------------|
| 10 | 2 hours | 15 minutes |
| 20 | 4 hours | 30 minutes |
| 50 | 10 hours | 1.5 hours |
| 100 | 20 hours | 3 hours |
| 200 | 40 hours | 6 hours |

---

## 🎯 What to Do Now

1. **Let it run**: Training is automatic
2. **Check at epoch 20**: ~4 hours from now
3. **Compare quality**: With old epoch 10
4. **Be patient**: Quality improves gradually
5. **Check final**: At epoch 200 (~40 hours)

---

**Training is running! You can close this terminal and check back later.** 🚀
