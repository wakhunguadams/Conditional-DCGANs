#!/usr/bin/env python3
"""
Analyze training results and show summary
"""
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image

def analyze_training():
    print("=" * 70)
    print("TRAINING RESULTS ANALYSIS")
    print("=" * 70)
    
    # Check checkpoints
    print("\n📁 CHECKPOINTS:")
    if os.path.exists('checkpoints'):
        ckpts = [f for f in os.listdir('checkpoints') if f.endswith('.pt')]
        for ckpt in sorted(ckpts):
            size = os.path.getsize(f'checkpoints/{ckpt}') / (1024**2)
            print(f"  ✓ {ckpt:<30} ({size:.1f} MB)")
        
        # Load final checkpoint to check history
        if 'G_final.pt' in ckpts:
            print("\n📊 MODEL INFO:")
            G = torch.load('checkpoints/G_final.pt', map_location='cpu')
            total_params = sum(p.numel() for p in G.values() if isinstance(p, torch.Tensor))
            print(f"  Generator parameters: {total_params:,}")
    else:
        print("  ⚠ No checkpoints found")
    
    # Check samples
    print("\n🖼️  GENERATED SAMPLES:")
    if os.path.exists('samples'):
        samples = [f for f in os.listdir('samples') if f.startswith('epoch_') and f.endswith('.png')]
        print(f"  Total sample images: {len(samples)}")
        for sample in sorted(samples)[:5]:  # Show first 5
            epoch = sample.replace('epoch_', '').replace('.png', '')
            size = os.path.getsize(f'samples/{sample}') / (1024**2)
            print(f"  ✓ Epoch {epoch:<4} ({size:.1f} MB)")
        if len(samples) > 5:
            print(f"  ... and {len(samples) - 5} more")
        
        # Check history plot
        if os.path.exists('samples/history.png'):
            print(f"\n  ✓ Training curves saved: samples/history.png")
    else:
        print("  ⚠ No samples found")
    
    # Check synthetic data
    print("\n🎨 SYNTHETIC DATA:")
    if os.path.exists('synthetic_data'):
        if os.path.exists('synthetic_data/generated_grid.png'):
            size = os.path.getsize('synthetic_data/generated_grid.png') / (1024**2)
            print(f"  ✓ Visualization grid: generated_grid.png ({size:.1f} MB)")
        
        # Check for generated datasets
        grades = [d for d in os.listdir('synthetic_data') if d.startswith('grade_')]
        if grades:
            print(f"\n  Generated datasets:")
            for grade_dir in sorted(grades):
                path = f'synthetic_data/{grade_dir}'
                count = len([f for f in os.listdir(path) if f.endswith('.png')])
                print(f"    ✓ {grade_dir:<20} {count:>4} images")
        else:
            print("  ℹ No full dataset generated yet")
            print("    Run: python generate.py --checkpoint checkpoints/G_final.pt --n_per_class 100")
    else:
        print("  ⚠ No synthetic data folder")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Training status
    if os.path.exists('checkpoints/G_final.pt'):
        print("✅ Training completed successfully!")
        print("✅ Generator model saved")
        print("✅ Sample images generated")
        
        print("\n📋 NEXT STEPS:")
        print("  1. View training curves: samples/history.png")
        print("  2. Check generated samples: samples/epoch_0001.png")
        print("  3. View visualization: synthetic_data/generated_grid.png")
        print("\n  To generate full synthetic dataset:")
        print("    python generate.py --checkpoint checkpoints/G_final.pt --n_per_class 100")
        
        print("\n  To continue training:")
        print("    python train.py --epochs 100 --batch_size 8")
    else:
        print("⚠ Training not complete or no checkpoints found")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    analyze_training()
