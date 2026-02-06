#!/bin/bash
# Quick start script for training the cDCGAN

echo "=== Conditional DCGAN for Prostate Cancer Image Synthesis ==="
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"

echo ""
echo "Starting training..."
echo "- Epochs: 100"
echo "- Batch size: 8"
echo "- Learning rate: 0.0002"
echo ""
echo "Checkpoints will be saved to: ./checkpoints/"
echo "Sample images will be saved to: ./samples/"
echo ""
echo "Press Ctrl+C to stop training"
echo ""

# Start training
python train.py --epochs 100 --batch_size 8 --lr 0.0002

echo ""
echo "Training complete!"
echo "To generate synthetic images, run:"
echo "  python generate.py --checkpoint checkpoints/G_final.pt --n_per_class 100"
