#!/bin/bash
# Prepare project for transfer to workstation

echo "=========================================="
echo "Preparing cDCGAN Project for Transfer"
echo "=========================================="
echo ""

# Get current directory
PROJECT_DIR=$(pwd)
PROJECT_NAME=$(basename "$PROJECT_DIR")
TRANSFER_DIR="${PROJECT_DIR}_transfer"
ARCHIVE_NAME="cdcgan_workstation_$(date +%Y%m%d_%H%M%S).tar.gz"

echo "Project: $PROJECT_NAME"
echo "Location: $PROJECT_DIR"
echo ""

# Create transfer directory
echo "Creating transfer package..."
mkdir -p "$TRANSFER_DIR"

# Copy essential files
echo "Copying essential files..."

# Python scripts
cp train.py "$TRANSFER_DIR/"
cp generate.py "$TRANSFER_DIR/"
cp generate_by_grade.py "$TRANSFER_DIR/"
cp check_data.py "$TRANSFER_DIR/"
cp analyze_results.py "$TRANSFER_DIR/"

# Setup and training scripts
cp setup_workstation.sh "$TRANSFER_DIR/"
cp train_workstation.sh "$TRANSFER_DIR/"
cp requirements.txt "$TRANSFER_DIR/"

# Documentation
cp README.md "$TRANSFER_DIR/" 2>/dev/null || true
cp GETTING_STARTED.md "$TRANSFER_DIR/" 2>/dev/null || true
cp WORKSTATION_DEPLOYMENT.md "$TRANSFER_DIR/"
cp paper_reference.md "$TRANSFER_DIR/" 2>/dev/null || true

# Data
echo "Copying data (this may take a while)..."
cp -r panda_data "$TRANSFER_DIR/"

# Checkpoints (if they exist and are small enough)
if [ -d "checkpoints" ]; then
    echo "Copying checkpoints..."
    mkdir -p "$TRANSFER_DIR/checkpoints"
    cp checkpoints/*.pt "$TRANSFER_DIR/checkpoints/" 2>/dev/null || true
fi

# Make scripts executable
chmod +x "$TRANSFER_DIR"/*.sh
chmod +x "$TRANSFER_DIR"/*.py

# Create archive
echo ""
echo "Creating compressed archive..."
cd "$(dirname "$TRANSFER_DIR")"
tar -czf "$ARCHIVE_NAME" "$(basename "$TRANSFER_DIR")"

# Get sizes
TRANSFER_SIZE=$(du -sh "$TRANSFER_DIR" | cut -f1)
ARCHIVE_SIZE=$(du -sh "$ARCHIVE_NAME" | cut -f1)

echo ""
echo "=========================================="
echo "Transfer Package Ready!"
echo "=========================================="
echo ""
echo "Transfer directory: $TRANSFER_DIR"
echo "  Size: $TRANSFER_SIZE"
echo ""
echo "Compressed archive: $ARCHIVE_NAME"
echo "  Size: $ARCHIVE_SIZE"
echo ""
echo "Contents:"
echo "  ✓ Training scripts (train.py, generate*.py)"
echo "  ✓ Setup scripts (setup_workstation.sh, train_workstation.sh)"
echo "  ✓ Data (525 patches in panda_data/)"
echo "  ✓ Documentation (*.md files)"
echo "  ✓ Checkpoints (if available)"
echo ""
echo "=========================================="
echo "Transfer Methods"
echo "=========================================="
echo ""
echo "Method 1: Using scp (recommended)"
echo "  scp $ARCHIVE_NAME username@workstation:/path/to/destination/"
echo ""
echo "Method 2: Using rsync"
echo "  rsync -avz --progress $TRANSFER_DIR/ username@workstation:/path/to/destination/CDCGANs/"
echo ""
echo "Method 3: Copy to USB"
echo "  cp $ARCHIVE_NAME /media/usb/"
echo ""
echo "On workstation, extract with:"
echo "  tar -xzf $ARCHIVE_NAME"
echo "  cd $(basename "$TRANSFER_DIR")"
echo "  ./setup_workstation.sh"
echo ""
