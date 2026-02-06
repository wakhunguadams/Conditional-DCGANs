# Tmux Quick Reference

## Essential Commands

### Start Training with Tmux
```bash
ssh adamswakhungu@workstation-ip
cd /path/to/CDCGANs
tmux new -s cdcgan
source venv/bin/activate
python train.py --epochs 200 --batch_size 32
```

### Detach and Disconnect
```bash
Ctrl+B D    # Detach from tmux
exit        # Disconnect from SSH
# Training continues running!
```

### Reconnect from Anywhere
```bash
ssh adamswakhungu@workstation-ip
tmux attach -t cdcgan
# You're back in your training session!
```

## Key Bindings

All tmux commands start with `Ctrl+B` (press both, release, then press next key)

### Session
```
Ctrl+B D         Detach from session
Ctrl+B $         Rename session
```

### Windows
```
Ctrl+B C         Create new window
Ctrl+B N         Next window
Ctrl+B P         Previous window
Ctrl+B 0-9       Go to window number
Ctrl+B ,         Rename window
```

### Panes
```
Ctrl+B "         Split horizontal
Ctrl+B %         Split vertical
Ctrl+B ↑↓←→      Navigate panes
Ctrl+B O         Next pane
Ctrl+B X         Close pane
Ctrl+B Z         Zoom pane (toggle fullscreen)
```

### Useful
```
Ctrl+B ?         Show all key bindings
Ctrl+B [         Scroll mode (Q to exit)
Ctrl+B T         Show time
```

## Command Line

```bash
# Session management
tmux new -s name              Create session
tmux ls                       List sessions
tmux attach -t name           Attach to session
tmux kill-session -t name     Kill session

# From inside tmux
Ctrl+B :                      Enter command mode
```

## Training Dashboard Setup

```bash
# Start session
tmux new -s cdcgan

# Create 4-pane layout
Ctrl+B "    # Split horizontal
Ctrl+B %    # Split vertical (top)
Ctrl+B ↓    # Move down
Ctrl+B %    # Split vertical (bottom)

# Result:
# ┌─────────────┬─────────────┐
# │  Training   │  Log tail   │
# ├─────────────┼─────────────┤
# │  GPU watch  │  Samples    │
# └─────────────┴─────────────┘

# Setup each pane:
# Top-left: Training
python train.py --epochs 200 --batch_size 32

# Top-right: Log
Ctrl+B →
tail -f logs/training.log

# Bottom-left: GPU
Ctrl+B ↓
watch -n 1 nvidia-smi

# Bottom-right: Samples
Ctrl+B →
watch -n 10 'ls -lht samples/ | head'
```

## Mobile Access

### Android (Termux)
```bash
pkg install openssh
ssh adamswakhungu@workstation-ip
tmux attach -t cdcgan
```

### iOS (Blink Shell or iSH)
```bash
ssh adamswakhungu@workstation-ip
tmux attach -t cdcgan
```

## Common Workflows

### Basic Training
```bash
tmux new -s train
cd /path/to/CDCGANs && source venv/bin/activate
python train.py --epochs 200 --batch_size 32
Ctrl+B D
```

### Training + Monitoring
```bash
tmux new -s train
# Start training in top pane
python train.py --epochs 200 --batch_size 32
Ctrl+B "
# Monitor GPU in bottom pane
watch -n 1 nvidia-smi
Ctrl+B D
```

### Multiple Experiments
```bash
# Experiment 1
tmux new -s exp1
python train.py --epochs 200 --batch_size 32 --lr 0.0002
Ctrl+B D

# Experiment 2
tmux new -s exp2
python train.py --epochs 200 --batch_size 32 --lr 0.0001
Ctrl+B D

# List all
tmux ls

# Attach to any
tmux attach -t exp1
```

## Troubleshooting

### Install tmux
```bash
sudo apt-get install tmux  # Ubuntu/Debian
sudo yum install tmux      # CentOS/RHEL
```

### Can't attach
```bash
tmux ls                    # List sessions
tmux attach -d -t cdcgan   # Force detach others and attach
```

### Session lost
```bash
ps aux | grep tmux         # Check if running
tmux ls                    # List sessions
tail logs/training.log     # Check what happened
```

---

**Print this and keep it handy!** 📋
