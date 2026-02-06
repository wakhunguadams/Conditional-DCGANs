#!/usr/bin/env python3
"""
Extract patches from PANDA whole slide images
"""
import os
import sys
import argparse
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path

def extract_patches(image_path, patch_size=256, max_patches=20, tissue_thresh=0.5):
    """Extract patches from a whole slide image"""
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return []
    
    w, h = img.size
    patches = []
    
    # Extract patches in a grid
    for i in range((h - patch_size) // patch_size + 1):
        for j in range((w - patch_size) // patch_size + 1):
            x, y = j * patch_size, i * patch_size
            if x + patch_size > w or y + patch_size > h:
                continue
            
            # Extract patch
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            arr = np.array(patch)
            
            # Check tissue content (not too white/background)
            tissue_ratio = np.mean(np.mean(arr, axis=2) < 220)
            
            # Check variance (not too uniform)
            variance = np.var(arr)
            
            if tissue_ratio >= tissue_thresh and variance > 100:
                patches.append((patch, tissue_ratio))
    
    # Sort by tissue content and return top patches
    patches.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in patches[:max_patches]]

def main():
    parser = argparse.ArgumentParser(description='Extract patches from PANDA dataset')
    parser.add_argument('--data_dir', type=str, default='./panda_data',
                       help='PANDA data directory')
    parser.add_argument('--output_dir', type=str, default='./panda_data/patches_256',
                       help='Output directory for patches')
    parser.add_argument('--patch_size', type=int, default=256,
                       help='Patch size (default: 256)')
    parser.add_argument('--max_patches', type=int, default=20,
                       help='Maximum patches per image (default: 20)')
    parser.add_argument('--max_per_grade', type=int, default=500,
                       help='Maximum images to process per grade (default: 500)')
    parser.add_argument('--tissue_thresh', type=float, default=0.5,
                       help='Tissue content threshold (default: 0.5)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("PANDA PATCH EXTRACTION")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"Max patches per image: {args.max_patches}")
    print(f"Max images per grade: {args.max_per_grade}")
    print(f"Tissue threshold: {args.tissue_thresh}")
    print("=" * 70)
    
    # Load CSV
    csv_path = Path(args.data_dir) / 'train.csv'
    if not csv_path.exists():
        print(f"✗ Error: {csv_path} not found")
        print("Please run download_panda_data.py first")
        return 1
    
    df = pd.read_csv(csv_path)
    print(f"\n✓ Loaded {len(df)} images from train.csv")
    print("\nGrade distribution:")
    for grade, count in df['isup_grade'].value_counts().sort_index().items():
        print(f"  Grade {grade}: {count} images")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    for grade in range(6):
        (output_dir / str(grade)).mkdir(parents=True, exist_ok=True)
    
    # Process images
    images_dir = Path(args.data_dir) / 'train_images'
    if not images_dir.exists():
        print(f"✗ Error: {images_dir} not found")
        return 1
    
    print(f"\n✓ Found train_images directory")
    print("\nExtracting patches...")
    
    counts = {i: 0 for i in range(6)}
    total_patches = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        grade = row['isup_grade']
        
        # Skip if we have enough for this grade
        if counts[grade] >= args.max_per_grade:
            continue
        
        # Get image path
        img_path = images_dir / f"{row['image_id']}.tiff"
        if not img_path.exists():
            continue
        
        # Extract patches
        patches = extract_patches(
            img_path,
            patch_size=args.patch_size,
            max_patches=args.max_patches,
            tissue_thresh=args.tissue_thresh
        )
        
        # Save patches
        for idx, patch in enumerate(patches):
            patch_name = f"{row['image_id']}_p{idx}.png"
            patch_path = output_dir / str(grade) / patch_name
            patch.save(patch_path)
            total_patches += 1
        
        if patches:
            counts[grade] += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nTotal patches extracted: {total_patches}")
    print("\nPatches per grade:")
    for grade in range(6):
        grade_dir = output_dir / str(grade)
        patch_count = len(list(grade_dir.glob('*.png')))
        print(f"  Grade {grade}: {patch_count} patches")
    
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("1. Verify data: python check_data.py")
    print("2. Start training: python train.py --epochs 200 --batch_size 32")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
