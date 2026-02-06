#!/bin/bash
# Workstation Training Script with logging and monitoring

# Configuration
EPOCHS=200
BATCH_SIZE=32
LR=0.0002
LOG_DIR="./logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# Create log directory
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "cDCGAN Training on Workstation"
echo "=========================================="
echo "Configuration:"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LR}"
echo "  Log file: ${LOG_FILE}"
echo "=========================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Check GPU
echo "GPU Information:"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    python3 -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
fi
echo ""

# Start training with logging
echo "Starting training..."
echo "Press Ctrl+C to stop"
echo ""

python train.py \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Log saved to: ${LOG_FILE}"
echo "Checkpoints in: ./checkpoints/"
echo "Samples in: ./samples/"
echo ""
echo "To generate synthetic images:"
echo "  python generate_by_grade.py --checkpoint checkpoints/G_final.pt --grades 0,1,2,3,4,5 --n_per_grade 100"
echo ""
