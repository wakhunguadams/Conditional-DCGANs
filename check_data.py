#!/usr/bin/env python3
"""
Check dataset and visualize sample patches
"""
import os
import matplotlib.pyplot as plt
from PIL import Image
import random

def check_dataset(patches_dir='./panda_data/patches_256'):
    """Check dataset statistics and show samples"""
    
    print("=" * 60)
    print("PANDA Dataset Statistics")
    print("=" * 60)
    
    grade_names = {
        0: "Grade 0: Benign (no cancer)",
        1: "Grade 1: Gleason 3+3",
        2: "Grade 2: Gleason 3+4",
        3: "Grade 3: Gleason 4+3",
        4: "Grade 4: Gleason 4+4, 3+5, 5+3",
        5: "Grade 5: Gleason 4+5, 5+4, 5+5"
    }
    
    total = 0
    samples_per_grade = {}
    
    for grade in range(6):
        gdir = os.path.join(patches_dir, str(grade))
        if os.path.exists(gdir):
            files = [f for f in os.listdir(gdir) if f.endswith(('.png', '.jpg'))]
            count = len(files)
            samples_per_grade[grade] = files
            total += count
            print(f"\n{grade_names[grade]}")
            print(f"  Patches: {count}")
        else:
            print(f"\n{grade_names[grade]}")
            print(f"  Patches: 0 (directory not found)")
            samples_per_grade[grade] = []
    
    print(f"\n{'=' * 60}")
    print(f"Total patches: {total}")
    print(f"{'=' * 60}\n")
    
    # Visualize samples
    if total > 0:
        print("Generating sample visualization...")
        fig, axes = plt.subplots(6, 4, figsize=(12, 18))
        fig.suptitle('Sample Patches by ISUP Grade', fontsize=16, y=0.995)
        
        for grade in range(6):
            files = samples_per_grade[grade]
            if files:
                # Show 4 random samples per grade
                samples = random.sample(files, min(4, len(files)))
                for i, fname in enumerate(samples):
                    img_path = os.path.join(patches_dir, str(grade), fname)
                    img = Image.open(img_path)
                    axes[grade, i].imshow(img)
                    axes[grade, i].axis('off')
                
                # Fill remaining slots if less than 4 samples
                for i in range(len(samples), 4):
                    axes[grade, i].axis('off')
                
                axes[grade, 0].set_ylabel(f'Grade {grade}', fontsize=12, rotation=0, 
                                         labelpad=40, va='center')
            else:
                for i in range(4):
                    axes[grade, i].text(0.5, 0.5, 'No data', 
                                       ha='center', va='center', fontsize=10)
                    axes[grade, i].axis('off')
        
        plt.tight_layout()
        output_path = 'dataset_samples.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
        plt.close()
    
    return total

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check PANDA dataset')
    parser.add_argument('--data_dir', type=str, default='./panda_data/patches_256',
                       help='Path to patches directory')
    args = parser.parse_args()
    
    check_dataset(args.data_dir)
