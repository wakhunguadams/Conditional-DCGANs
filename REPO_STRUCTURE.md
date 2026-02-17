# Repository Structure

```
CDCGANs/
├── README.md                           # Main documentation (updated for v2)
├── QUICK_START_V2.md                   # Quick start guide for v2
├── paper_reference.md                  # Research paper reference
│
├── config.py                           # ✨ NEW: Centralized configuration
├── train_v2.py                         # ✨ NEW: Fixed training script
│
├── generate.py                         # Generate images from trained model
├── generate_by_grade.py                # Generate by specific grade
├── analyze_results.py                  # Analyze training results
├── check_data.py                       # Data verification utility
│
├── tests/                              # ✨ NEW: Test suite
│   ├── test_architecture.py            #   Architecture validation (8 tests)
│   └── test_training.py                #   Training loop validation (5 tests)
│
├── archive/                            # Old implementations and docs
│   ├── train_v1_original.py            #   Original buggy version
│   ├── train_v1_improved.py            #   Wasserstein version (also buggy)
│   └── old_docs/                       #   26 archived documentation files
│
├── checkpoints/                        # Old v1 checkpoints
├── checkpoints_v2/                     # ✨ NEW: v2 checkpoints
├── checkpoints_improved/               # Old improved checkpoints
│
├── samples/                            # Old v1 samples
├── samples_v2/                         # ✨ NEW: v2 generated samples
│
├── logs/                               # Training logs
│   ├── training_v2_test.log            #   Current 10-epoch test
│   ├── training_10epochs.log           #   Old v1 training (failed)
│   └── training_improved_200epochs.log #   Old improved training (failed)
│
├── panda_data/                         # Dataset
│   └── patches_256/                    #   256x256 patches by grade (0-5)
│
├── tensor/                             # TensorFlow reference implementation
│   ├── Conditional_GAN_Data_preprocessing.ipynb
│   └── prostate-cancer-downloading-dataset.ipynb
│
└── venv/                               # Python virtual environment
```

## Key Files

### Essential (Keep)
- `README.md` - Main documentation
- `QUICK_START_V2.md` - Quick start guide
- `paper_reference.md` - Research reference
- `config.py` - Configuration
- `train_v2.py` - Training script (USE THIS!)
- `tests/` - Test suite

### Archived (Old)
- `archive/train_v1_*.py` - Old buggy implementations
- `archive/old_docs/` - 26 redundant documentation files

### Generated (During Training)
- `checkpoints_v2/` - Model checkpoints
- `samples_v2/` - Generated samples
- `logs/` - Training logs
