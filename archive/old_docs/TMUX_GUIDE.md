# Tmux Training Guide

## Why Use Tmux?

Tmux allows you to:
- ✅ Keep training running after disconnecting from SSH
- ✅ Monitor training from anywhere (home, office, mobile)
- ✅ Split screen to watch logs and GPU usage simultaneously
- ✅ Resume sessions instantly without losing context

## Quick Start

### Start Training with Tmux

```bash
# SSH to workstation
ssh adamswakhungu@workstation-ip

# Start tmux session
tmux new -s cdcgan

# Activate environment and train
cd /path/to/CDCGANs
source venv/bin/activate
python train.py --epochs 200 --batch_size 32

# Detach: Press Ctrl+B then D
# You can now disconnect from SSH - training continues!
```

### Reconnect and Monitor

```bash
# SSH back in (from anywhere)
ssh adamswakhungu@workstation-ip

# Reattach to session
tmux attach -t cdcgan

# You'll see your training exactly where you left it!
```

## Complete Tmux Workflow

### 1. Create Training Session

```bash
ssh adamswakhungu@workstation-ip
cd /path/to/CDCGANs

# Create named session
tmux new -s cdcgan

# Inside tmux, start training
source venv/bin/activate
python train.py --epochs 200 --batch_size 32
```

### 2. Split Screen for Monitoring

While training is running:

```bash
# Split horizontally (Ctrl+B then ")
Ctrl+B "

# Now you have two panes
# Top pane: Training running
# Bottom pane: New shell

# In bottom pane, monitor GPU
watch -n 1 nvidia-smi

# Switch between panes: Ctrl+B then arrow keys
```

### 3. Create Multiple Windows

```bash
# Create new window (Ctrl+B then C)
Ctrl+B C

# Window 0: Training
# Window 1: Monitoring (current)

# In window 1, check samples
cd /path/to/CDCGANs
ls -lht samples/

# Switch windows:
# Ctrl+B 0  -> Go to window 0 (training)
# Ctrl+B 1  -> Go to window 1 (monitoring)
# Ctrl+B N  -> Next window
# Ctrl+B P  -> Previous window
```

### 4. Detach and Disconnect

```bash
# Detach from tmux (Ctrl+B then D)
Ctrl+B D

# Exit SSH
exit

# Training continues running on workstation!
```

### 5. Reconnect from Anywhere

```bash
# From home, office, or mobile
ssh adamswakhungu@workstation-ip

# List sessions
tmux ls

# Reattach
tmux attach -t cdcgan

# You're back in your training session!
```

## Advanced Setup: Multi-Pane Dashboard

Create a monitoring dashboard:

```bash
# Start session
tmux new -s cdcgan

# Split into 4 panes
Ctrl+B "    # Split horizontally
Ctrl+B %    # Split vertically (top pane)
Ctrl+B ↓    # Move to bottom pane
Ctrl+B %    # Split vertically (bottom pane)

# Now you have 4 panes:
# ┌─────────┬─────────┐
# │ Pane 0  │ Pane 1  │
# ├─────────┼─────────┤
# │ Pane 2  │ Pane 3  │
# └─────────┴─────────┘

# Setup each pane:
# Pane 0 (top-left): Training
cd /path/to/CDCGANs
source venv/bin/activate
python train.py --epochs 200 --batch_size 32

# Pane 1 (top-right): Training log
Ctrl+B →    # Move to pane 1
tail -f logs/training.log

# Pane 2 (bottom-left): GPU monitoring
Ctrl+B ↓    # Move to pane 2
watch -n 1 nvidia-smi

# Pane 3 (bottom-right): File monitoring
Ctrl+B →    # Move to pane 3
watch -n 5 'ls -lht samples/ | head -10'
```

## Essential Tmux Commands

### Session Management

```bash
# Create new session
tmux new -s session_name

# List sessions
tmux ls

# Attach to session
tmux attach -t session_name

# Detach from session
Ctrl+B D

# Kill session
tmux kill-session -t session_name

# Rename session
Ctrl+B $
```

### Window Management

```bash
# Create new window
Ctrl+B C

# Switch windows
Ctrl+B 0-9      # Go to window number
Ctrl+B N        # Next window
Ctrl+B P        # Previous window
Ctrl+B L        # Last window

# Rename window
Ctrl+B ,

# Close window
Ctrl+B &
```

### Pane Management

```bash
# Split panes
Ctrl+B "        # Split horizontally
Ctrl+B %        # Split vertically

# Navigate panes
Ctrl+B ↑↓←→     # Arrow keys
Ctrl+B O        # Next pane
Ctrl+B ;        # Last pane

# Resize panes
Ctrl+B Ctrl+↑   # Resize up
Ctrl+B Ctrl+↓   # Resize down
Ctrl+B Ctrl+←   # Resize left
Ctrl+B Ctrl+→   # Resize right

# Close pane
Ctrl+B X        # Confirm with Y
```

### Useful Commands

```bash
# Show time
Ctrl+B T

# List all key bindings
Ctrl+B ?

# Reload tmux config
Ctrl+B : then type "source-file ~/.tmux.conf"
```

## Recommended Tmux Configuration

Create `~/.tmux.conf` on workstation:

```bash
cat > ~/.tmux.conf << 'EOF'
# Enable mouse support
set -g mouse on

# Increase scrollback buffer
set -g history-limit 10000

# Start windows at 1 instead of 0
set -g base-index 1
set -g pane-base-index 1

# Renumber windows when one is closed
set -g renumber-windows on

# Enable 256 colors
set -g default-terminal "screen-256color"

# Status bar
set -g status-style bg=black,fg=white
set -g status-left '[#S] '
set -g status-right '%Y-%m-%d %H:%M'

# Highlight active window
setw -g window-status-current-style bg=blue,fg=white,bold

# Pane borders
set -g pane-border-style fg=white
set -g pane-active-border-style fg=blue

# Easy config reload
bind r source-file ~/.tmux.conf \; display "Config reloaded!"
EOF

# Apply configuration
tmux source-file ~/.tmux.conf
```

## Training Workflow Examples

### Example 1: Basic Training

```bash
# Start session
tmux new -s training

# Train
cd /path/to/CDCGANs
source venv/bin/activate
python train.py --epochs 200 --batch_size 32

# Detach: Ctrl+B D
# Disconnect: exit
```

### Example 2: Training with Monitoring

```bash
# Start session
tmux new -s training

# Split screen
Ctrl+B "

# Top pane: Training
cd /path/to/CDCGANs
source venv/bin/activate
python train.py --epochs 200 --batch_size 32

# Bottom pane: GPU monitoring
Ctrl+B ↓
watch -n 1 nvidia-smi

# Detach: Ctrl+B D
```

### Example 3: Full Dashboard

```bash
# Start session
tmux new -s dashboard

# Create 4 panes (see Advanced Setup above)

# Pane 0: Training
python train.py --epochs 200 --batch_size 32

# Pane 1: Log
tail -f logs/training.log

# Pane 2: GPU
watch -n 1 nvidia-smi

# Pane 3: Samples
watch -n 10 'ls -lht samples/ | head'

# Detach: Ctrl+B D
```

### Example 4: Multiple Training Sessions

```bash
# Session 1: Main training
tmux new -s train_main
python train.py --epochs 200 --batch_size 32
Ctrl+B D

# Session 2: Experiment with different LR
tmux new -s train_exp
python train.py --epochs 200 --batch_size 32 --lr 0.0001
Ctrl+B D

# Session 3: Monitoring
tmux new -s monitor
watch -n 1 nvidia-smi
Ctrl+B D

# List all sessions
tmux ls

# Attach to any session
tmux attach -t train_main
```

## Mobile Monitoring

You can monitor training from your phone!

### Using Termux (Android)

```bash
# Install Termux from F-Droid or Play Store
# Inside Termux:
pkg install openssh
ssh adamswakhungu@workstation-ip
tmux attach -t cdcgan
```

### Using iSH (iOS)

```bash
# Install iSH from App Store
# Inside iSH:
apk add openssh
ssh adamswakhungu@workstation-ip
tmux attach -t cdcgan
```

### Using Blink Shell (iOS)

```bash
# Install Blink Shell (paid app, best experience)
# Connect via SSH
ssh adamswakhungu@workstation-ip
tmux attach -t cdcgan
```

## Troubleshooting

### Issue: Tmux not installed

```bash
# Ubuntu/Debian
sudo apt-get install tmux

# CentOS/RHEL
sudo yum install tmux

# macOS
brew install tmux
```

### Issue: Can't attach to session

```bash
# List sessions
tmux ls

# If session exists but can't attach
tmux attach -d -t cdcgan

# If session doesn't exist
tmux new -s cdcgan
```

### Issue: Session disappeared

```bash
# Check if tmux server is running
ps aux | grep tmux

# List sessions
tmux ls

# If no sessions, training may have crashed
# Check logs
tail -100 logs/training.log
```

### Issue: Panes too small

```bash
# Resize panes
Ctrl+B Ctrl+↑↓←→

# Or kill and recreate with better layout
Ctrl+B X    # Kill pane
Ctrl+B "    # Split again
```

## Quick Reference Card

```
SESSION MANAGEMENT
tmux new -s name     Create session
tmux ls              List sessions
tmux attach -t name  Attach to session
Ctrl+B D             Detach from session

WINDOW MANAGEMENT
Ctrl+B C             Create window
Ctrl+B N             Next window
Ctrl+B P             Previous window
Ctrl+B 0-9           Go to window #

PANE MANAGEMENT
Ctrl+B "             Split horizontal
Ctrl+B %             Split vertical
Ctrl+B ↑↓←→          Navigate panes
Ctrl+B X             Close pane

USEFUL
Ctrl+B ?             Help
Ctrl+B T             Show time
Ctrl+B [             Scroll mode (Q to exit)
```

## Best Practices

1. **Name your sessions descriptively**
   ```bash
   tmux new -s cdcgan_200epochs
   tmux new -s cdcgan_experiment_lr0001
   ```

2. **Use multiple windows for organization**
   - Window 0: Training
   - Window 1: Monitoring
   - Window 2: Analysis

3. **Enable mouse support** (in ~/.tmux.conf)
   - Click to switch panes
   - Drag to resize panes
   - Scroll to view history

4. **Keep a monitoring pane open**
   - Always have `nvidia-smi` or `htop` visible
   - Catch issues early

5. **Document your sessions**
   ```bash
   # Create a sessions.txt file
   echo "cdcgan_main: Main training, 200 epochs, batch 32" >> sessions.txt
   echo "cdcgan_exp1: Experiment with lr=0.0001" >> sessions.txt
   ```

---

**Master tmux and monitor your training from anywhere!** 🚀
