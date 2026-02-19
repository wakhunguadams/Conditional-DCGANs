import os
import torch
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from models import Generator

def generate_organized(checkpoint_path, n_per_grade=100, output_root="dataset_export"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    syn_root = os.path.join(output_root, "synthetic")
    org_root = os.path.join(output_root, "original")
    local_patches = "panda_data/patches_256"
    
    # Initialize Generator
    G = Generator(nz=512, style_dim=512, n_classes=6, ngf=64).to(device)
    # weights_only=False fix for PyTorch 2.6+
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # Older torch versions don't have weights_only
        state = torch.load(checkpoint_path, map_location=device)
        
    if 'G' in state:
        G.load_state_dict(state['G'])
    else:
        G.load_state_dict(state)
    G.eval()
    print(f"Loaded weights from {checkpoint_path}")
    
    grade_names = {
        0: 'benign',
        1: 'G3x3',
        2: 'G3x4',
        3: 'G4x3',
        4: 'G4x4',
        5: 'high'
    }

    for grade in range(6):
        # 1. Generate Synthetic
        grade_syn_dir = os.path.join(syn_root, f"grade_{grade}_{grade_names[grade]}")
        os.makedirs(grade_syn_dir, exist_ok=True)
        print(f"Generating Grade {grade} synthetic images...")
        
        with torch.no_grad():
            for i in tqdm(range(n_per_grade)):
                z = torch.randn(1, 512, device=device)
                y = torch.tensor([grade], device=device)
                img = G(z, y)
                img = ((img + 1) / 2).clamp(0, 1)
                img = (img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(grade_syn_dir, f"syn_{i:04d}.png"))
        
        # 2. Copy Original
        grade_org_dir = os.path.join(org_root, f"grade_{grade}_{grade_names[grade]}")
        os.makedirs(grade_org_dir, exist_ok=True)
        local_grade_dir = os.path.join(local_patches, str(grade))
        
        if os.path.exists(local_grade_dir):
            files = [f for f in os.listdir(local_grade_dir) if f.endswith('.png')]
            to_copy = files[:min(len(files), n_per_grade)]
            print(f"Copying {len(to_copy)} original Grade {grade} images...")
            for f in to_copy:
                shutil.copy(os.path.join(local_grade_dir, f), os.path.join(grade_org_dir, f))
        else:
            print(f"Warning: Original patches for Grade {grade} not found in {local_grade_dir}")

    print(f"\nAll images organized in {output_root}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints_proj/ckpt_epoch_320.pt")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    
    generate_organized(args.checkpoint, args.count)
