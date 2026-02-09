#!/usr/bin/env python3
"""
Extract patches from PANDA whole slide images - PARALLEL VERSION
Optimized for multi-core workstations
"""
import os
import sys
import argparse
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

def extract_patches_from_image(image_path, patch_size=256, max_patches=20, tissue_thresh=0.5):
    """Extract patches from a single image"""
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
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

def process_single_image(row_data, images_dir, output_dir, patch_size, max_patches, tissue_thresh):
    """Process a single image (for parallel execution)"""
    image_id, grade = row_data
    
    # Get image path
    img_path = images_dir / f"{image_id}.tiff"
    if not img_path.exists():
        return 0
    
    # Extract patches
    patches = extract_patches_from_image(
        img_path,
        patch_size=patch_size,
        max_patches=max_patches,
        tissue_thresh=tissue_thresh
    )
    
    # Save patches
    patch_count = 0
    for idx, patch in enumerate(patches):
        patch_name = f"{image_id}_p{idx}.png"
        patch_path = output_dir / str(grade) / patch_name
        patch.save(patch_path)
        patch_count += 1
    
    return patch_count

def main():
    parser = argparse.ArgumentParser(description='Extract patches from PANDA dataset (PARALLEL)')
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
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()
    
    # Determine number of workers
    if args.workers is None:
        args.workers = max(1, cpu_count() - 1)  # Leave one core free
    
    print("=" * 70)
    print("PANDA PATCH EXTRACTION (PARALLEL)")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"Max patches per image: {args.max_patches}")
    print(f"Max images per grade: {args.max_per_grade}")
    print(f"Tissue threshold: {args.tissue_thresh}")
    print(f"Parallel workers: {args.workers}")
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
    
    # Prepare data for parallel processing
    # Limit images per grade
    images_to_process = []
    for grade in range(6):
        grade_df = df[df['isup_grade'] == grade].head(args.max_per_grade)
        for _, row in grade_df.iterrows():
            images_to_process.append((row['image_id'], row['isup_grade']))
    
    print(f"\nProcessing {len(images_to_process)} images with {args.workers} workers...")
    print("This may take a while...\n")
    
    # Create partial function with fixed parameters
    process_func = partial(
        process_single_image,
        images_dir=images_dir,
        output_dir=output_dir,
        patch_size=args.patch_size,
        max_patches=args.max_patches,
        tissue_thresh=args.tissue_thresh
    )
    
    # Process in parallel with progress bar
    total_patches = 0
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, images_to_process),
            total=len(images_to_process),
            desc="Extracting patches"
        ))
        total_patches = sum(results)
    
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
    print("2. Start training: python train_improved.py --epochs 200 --batch_size 64")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
