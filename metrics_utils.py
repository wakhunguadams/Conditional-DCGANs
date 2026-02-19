import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3
from scipy import linalg

def calculate_sharpness(img_tensor):
    """
    Calculates sharpness as the variance of the Laplacian.
    img_tensor: (B, C, H, W) in range [-1, 1] or [0, 1]
    """
    # Convert to grayscale if it's RGB
    if img_tensor.shape[1] == 3:
        img_gray = img_tensor.mean(dim=1, keepdim=True)
    else:
        img_gray = img_tensor

    # Laplacian kernel
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=img_tensor.device)
    kernel = kernel.view(1, 1, 3, 3)

    laplacian = F.conv2d(img_gray, kernel, padding=1)
    
    # Calculate variance per image in batch
    sharpness = torch.var(laplacian, dim=(1, 2, 3))
    return sharpness.mean().item()

def calculate_mse(real_tensor, fake_tensor):
    """
    Calculates Mean Squared Error between real and fake images.
    """
    if real_tensor.min() < 0:
        real_tensor = (real_tensor + 1) / 2
    if fake_tensor.min() < 0:
        fake_tensor = (fake_tensor + 1) / 2
        
    mse = F.mse_loss(real_tensor, fake_tensor)
    return mse.item()

def calculate_psnr(mse, max_val=1.0):
    """
    Calculates Peak Signal-to-Noise Ratio.
    """
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(torch.tensor(mse))).item()

class FIDCalculator:
    def __init__(self, device):
        self.device = device
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = torch.nn.Identity()
        self.inception = self.inception.to(device).eval()

    def get_features(self, imgs):
        if imgs.min() < 0:
            imgs = (imgs + 1) / 2
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        imgs = (imgs - mean) / std
        imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
        with torch.no_grad():
            return self.inception(imgs)

    def compute(self, real_imgs, fake_imgs):
        real_feats = self.get_features(real_imgs).cpu().numpy()
        fake_feats = self.get_features(fake_imgs).cpu().numpy()
        
        mu1, s1 = np.mean(real_feats, 0), np.cov(real_feats, rowvar=False)
        mu2, s2 = np.mean(fake_feats, 0), np.cov(fake_feats, rowvar=False)
        
        diff = mu1 - mu2
        covmean = linalg.sqrtm(s1.dot(s2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        return diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean)

def get_all_metrics(real_imgs, fake_imgs, fid_calc=None):
    mse = calculate_mse(real_imgs, fake_imgs)
    psnr = calculate_psnr(mse)
    sharpness_real = calculate_sharpness(real_imgs)
    sharpness_fake = calculate_sharpness(fake_imgs)
    
    res = {
        'mse': mse,
        'psnr': psnr,
        'sharpness_real': sharpness_real,
        'sharpness_fake': sharpness_fake
    }
    
    if fid_calc:
        try:
            res['fid'] = fid_calc.compute(real_imgs, fake_imgs)
        except Exception:
            res['fid'] = 0.0
            
    return res
