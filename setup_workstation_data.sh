#!/bin/bash
# Automated workstation data download and setup script

set -e  # Exit on error

echo "======================================================================"
echo "WORKSTATION DATA SETUP"
echo "======================================================================"
echo ""
echo "This script will:"
echo "  1. Check for Kaggle credentials"
echo "  2. Download PANDA dataset (~100 GB)"
echo "  3. Extract patches optimized for workstation"
echo "  4. Verify data integrity"
echo ""
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration - ADJUST THESE FOR YOUR WORKSTATION
MAX_PER_GRADE=1500    # Number of images to process per grade
MAX_PATCHES=35        # Patches to extract per image
PATCH_SIZE=256        # Patch size in pixels
TISSUE_THRESH=0.5     # Tissue content threshold

echo "Configuration:"
echo "  Max images per grade: $MAX_PER_GRADE"
echo "  Max patches per image: $MAX_PATCHES"
echo "  Patch size: ${PATCH_SIZE}x${PATCH_SIZE}"
echo "  Tissue threshold: $TISSUE_THRESH"
echo ""

# Ask for confirmation
read -p "Continue with this configuration? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled"
    exit 0
fi

# Step 1: Check for Kaggle credentials
echo ""
echo "======================================================================"
echo "STEP 1: Checking Kaggle Credentials"
echo "======================================================================"

if [ ! -f "kaggle_credentials.json" ]; then
    echo -e "${RED}✗ kaggle_credentials.json not found${NC}"
    echo ""
    echo "Please create kaggle_credentials.json with:"
    echo '{'
    echo '  "username": "your_kaggle_username",'
    echo '  "key": "your_kaggle_api_key"'
    echo '}'
    echo ""
    echo "Get your API key from: https://www.kaggle.com/settings/account"
    exit 1
fi

echo -e "${GREEN}✓ Found kaggle_credentials.json${NC}"

# Step 2: Setup Python environment
echo ""
echo "======================================================================"
echo "STEP 2: Setting Up Python Environment"
echo "======================================================================"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q kaggle

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 3: Download dataset
echo ""
echo "======================================================================"
echo "STEP 3: Downloading PANDA Dataset"
echo "======================================================================"
echo ""
echo -e "${YELLOW}WARNING: This will download ~100 GB of data${NC}"
echo "This may take 30 minutes to several hours depending on your connection"
echo ""

if [ -f "panda_data/train.csv" ] && [ -d "panda_data/train_images" ]; then
    echo -e "${GREEN}✓ Dataset already downloaded${NC}"
    
    # Count existing files
    TIFF_COUNT=$(find panda_data/train_images -name "*.tiff" 2>/dev/null | wc -l)
    echo "  Found $TIFF_COUNT TIFF files"
    
    read -p "Re-download dataset? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python download_panda_data.py
    fi
else
    python download_panda_data.py
fi

# Step 4: Extract patches
echo ""
echo "======================================================================"
echo "STEP 4: Extracting Patches"
echo "======================================================================"
echo ""
echo "This will extract patches with the following settings:"
echo "  - Max images per grade: $MAX_PER_GRADE"
echo "  - Max patches per image: $MAX_PATCHES"
echo "  - Patch size: ${PATCH_SIZE}x${PATCH_SIZE}"
echo ""
echo -e "${YELLOW}This may take 1-3 hours depending on settings${NC}"
echo ""

if [ -d "panda_data/patches_256" ]; then
    PATCH_COUNT=$(find panda_data/patches_256 -name "*.png" 2>/dev/null | wc -l)
    echo -e "${YELLOW}✓ Found existing patches: $PATCH_COUNT${NC}"
    
    read -p "Re-extract patches? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing old patches..."
        rm -rf panda_data/patches_256
        
        echo "Extracting new patches..."
        python extract_patches.py \
            --max_per_grade $MAX_PER_GRADE \
            --max_patches $MAX_PATCHES \
            --patch_size $PATCH_SIZE \
            --tissue_thresh $TISSUE_THRESH
    fi
else
    echo "Extracting patches..."
    python extract_patches.py \
        --max_per_grade $MAX_PER_GRADE \
        --max_patches $MAX_PATCHES \
        --patch_size $PATCH_SIZE \
        --tissue_thresh $TISSUE_THRESH
fi

# Step 5: Verify data
echo ""
echo "======================================================================"
echo "STEP 5: Verifying Data"
echo "======================================================================"

python check_data.py

# Step 6: Summary
echo ""
echo "======================================================================"
echo "SETUP COMPLETE!"
echo "======================================================================"
echo ""

# Calculate statistics
TOTAL_PATCHES=$(find panda_data/patches_256 -name "*.png" 2>/dev/null | wc -l)
DISK_USAGE=$(du -sh panda_data/ 2>/dev/null | cut -f1)

echo -e "${GREEN}✓ Data download and extraction complete${NC}"
echo ""
echo "Statistics:"
echo "  Total patches: $TOTAL_PATCHES"
echo "  Disk usage: $DISK_USAGE"
echo ""

# Show patches per grade
echo "Patches per grade:"
for grade in 0 1 2 3 4 5; do
    if [ -d "panda_data/patches_256/$grade" ]; then
        COUNT=$(find panda_data/patches_256/$grade -name "*.png" 2>/dev/null | wc -l)
        echo "  Grade $grade: $COUNT patches"
    fi
done

echo ""
echo "======================================================================"
echo "NEXT STEPS"
echo "======================================================================"
echo ""
echo "1. Start training:"
echo "   python train_improved.py --epochs 200 --batch_size 64"
echo ""
echo "2. Or use tmux for background training:"
echo "   tmux new -s training"
echo "   python train_improved.py --epochs 200 --batch_size 64"
echo "   # Press Ctrl+B then D to detach"
echo ""
echo "3. Monitor training:"
echo "   tmux attach -t training"
echo ""
echo "4. See TRAINING_COMMANDS.md for more options"
echo ""
echo "======================================================================"
