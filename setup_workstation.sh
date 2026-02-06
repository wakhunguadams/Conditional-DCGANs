#!/bin/bash
# Workstation Setup Script for cDCGAN Training

echo "=========================================="
echo "cDCGAN Workstation Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Check for CUDA
echo ""
echo "Checking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    
    echo ""
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "⚠ No NVIDIA GPU detected, installing CPU version"
    echo "Installing PyTorch (CPU)..."
    pip install torch torchvision
fi

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install numpy pandas pillow matplotlib tqdm scikit-image scipy jupyter

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python3 -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p samples
mkdir -p synthetic_data
mkdir -p logs

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start training:"
echo "  source venv/bin/activate"
echo "  python train.py --epochs 100 --batch_size 16"
echo ""
echo "For GPU training with larger batch size:"
echo "  python train.py --epochs 200 --batch_size 32"
echo ""
