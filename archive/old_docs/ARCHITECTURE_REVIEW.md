# Architecture Review & Improvements

## Current Architecture Analysis

### ✅ What You're Doing Well

1. **Conditional Generation**
   - ✅ Label embedding for both G and D
   - ✅ Proper concatenation of noise and label
   - ✅ Label map injection in discriminator

2. **Training Stability**
   - ✅ Spectral normalization in discriminator
   - ✅ Label smoothing (0.1)
   - ✅ Proper weight initialization
   - ✅ Learning rate scheduling

3. **Data Pipeline**
   - ✅ Class balancing through oversampling
   - ✅ Data augmentation (flips, rotations, color jitter)
   - ✅ Proper normalization

4. **Architecture Basics**
   - ✅ DCGAN-style architecture
   - ✅ BatchNorm in generator
   - ✅ LeakyReLU in discriminator
   - ✅ Tanh output in generator

### ⚠️ Potential Issues & Improvements

## 1. Discriminator Label Conditioning

**Current Issue:**
```python
self.label_emb = nn.Embedding(num_classes, img_size * img_size)  # 256*256 = 65,536 dims!
lmap = self.label_emb(labels).view(bs, 1, self.img_size, self.img_size)
```

**Problem:** 
- Embedding dimension is HUGE (65,536 for 256x256 images)
- This creates a massive parameter space for just 6 classes
- Label information might be too diluted

**Recommended Fix:**
```python
# Option 1: Smaller embedding + spatial replication
self.label_emb = nn.Embedding(num_classes, 128)
# Then expand spatially

# Option 2: Projection-based conditioning
self.label_proj = nn.Sequential(
    nn.Embedding(num_classes, 128),
    nn.Linear(128, img_size * img_size),
    nn.LeakyReLU(0.2)
)

# Option 3: Auxiliary classifier (AC-GAN style)
# Add classification head to discriminator
```

## 2. Generator Architecture Depth

**Current:** 6 transposed conv layers (4x4 → 256x256)

**Consideration:**
- Medical images have fine details
- Might benefit from:
  - Skip connections (U-Net style)
  - Self-attention layers
  - Progressive growing

**Recommended Addition:**
```python
# Add self-attention at 64x64 resolution
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H * W)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).view(B, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x
```

## 3. Loss Function

**Current:** Binary Cross-Entropy (BCE)

**Considerations:**
- BCE is standard but can be unstable
- Medical images might benefit from perceptual loss

**Recommended Additions:**

### Option A: Wasserstein Loss (WGAN-GP)
```python
# More stable training
def gradient_penalty(D, real, fake, labels, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True, retain_graph=True
    )[0]
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# In training:
loss_D = -torch.mean(D(real, labels)) + torch.mean(D(fake, labels)) + lambda_gp * gp
loss_G = -torch.mean(D(fake, labels))
```

### Option B: Perceptual Loss
```python
# Add VGG-based perceptual loss for medical realism
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
    
    def forward(self, fake, real):
        fake_features = self.vgg(fake)
        real_features = self.vgg(real)
        return F.l1_loss(fake_features, real_features)

# In training:
loss_G = criterion(D(fake, labels), real_lbl) + 0.1 * perceptual_loss(fake, real)
```

### Option C: Hinge Loss
```python
# Often more stable than BCE
def d_hinge_loss(real_logits, fake_logits):
    return torch.mean(F.relu(1.0 - real_logits)) + torch.mean(F.relu(1.0 + fake_logits))

def g_hinge_loss(fake_logits):
    return -torch.mean(fake_logits)
```

## 4. Discriminator Architecture

**Current Issue:**
- BatchNorm after spectral norm might conflict
- Spectral norm is for normalization, BatchNorm adds another layer

**Recommended Fix:**
```python
# Remove BatchNorm when using Spectral Norm
self.main = nn.Sequential(
    spectral_norm(nn.Conv2d(nc+1, ndf, 4, 2, 1, bias=False)),
    nn.LeakyReLU(0.2, True),
    spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)),
    nn.LeakyReLU(0.2, True),  # Remove BatchNorm
    # ... continue without BatchNorm
)

# OR use LayerNorm instead
nn.GroupNorm(1, ndf*2)  # LayerNorm equivalent
```

## 5. Missing Components for Medical Images

### A. Texture Preservation
```python
# Add texture loss
def texture_loss(fake, real):
    # Gram matrix for style/texture
    def gram_matrix(x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    return F.l1_loss(gram_matrix(fake), gram_matrix(real))
```

### B. Multi-Scale Discriminator
```python
# Better for capturing details at different scales
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_scales=3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            Discriminator(...) for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)
    
    def forward(self, x, labels):
        outputs = []
        for i, D in enumerate(self.discriminators):
            outputs.append(D(x, labels))
            if i < len(self.discriminators) - 1:
                x = self.downsample(x)
        return outputs
```

### C. Progressive Growing
```python
# Start with low resolution, gradually increase
# Helps with training stability and detail generation
```

## 6. Evaluation Metrics

**Currently Missing:**
- FID score implementation
- Inception Score
- Medical-specific metrics

**Recommended Addition:**
```python
from scipy import linalg
from torchvision.models import inception_v3

class FIDCalculator:
    def __init__(self):
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()
        self.inception.eval()
    
    def calculate_fid(self, real_images, fake_images):
        # Extract features
        real_features = self.get_features(real_images)
        fake_features = self.get_features(fake_images)
        
        # Calculate statistics
        mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(0), np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        diff = mu1 - mu2
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid
```

## 7. Training Improvements

### A. Two Time-Scale Update Rule (TTUR)
```python
# Different learning rates for G and D
opt_G = optim.Adam(G.parameters(), lr=0.0001, betas=(0.0, 0.9))
opt_D = optim.Adam(D.parameters(), lr=0.0004, betas=(0.0, 0.9))
```

### B. Gradient Accumulation
```python
# For larger effective batch sizes
accumulation_steps = 4
for i, (imgs, labels) in enumerate(dataloader):
    loss = train_step(...)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        opt_G.step()
        opt_D.step()
        opt_G.zero_grad()
        opt_D.zero_grad()
```

### C. Exponential Moving Average (EMA)
```python
# Smoother generator for better samples
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

# Use EMA model for generation
ema_G = EMA(G, decay=0.999)
```

## Recommended Priority Order

### High Priority (Implement First)
1. **Fix discriminator label conditioning** - Reduce embedding size
2. **Remove BatchNorm from discriminator** - Conflicts with spectral norm
3. **Add FID score** - For proper evaluation
4. **Implement EMA** - Better generation quality

### Medium Priority
5. **Add self-attention** - Better detail capture
6. **Try Hinge loss** - More stable training
7. **Multi-scale discriminator** - Better at different scales

### Low Priority (Experiments)
8. **Perceptual loss** - If quality issues persist
9. **Progressive growing** - If training is unstable
10. **WGAN-GP** - Alternative if BCE fails

## Suggested Improved Architecture

```python
class ImprovedGenerator(nn.Module):
    def __init__(self, nz, ngf, nc, num_classes):
        super().__init__()
        # Smaller label embedding
        self.label_emb = nn.Embedding(num_classes, 128)
        inp = nz + 128
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(inp, ngf*32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*32), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*32, ngf*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*16), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            # Add self-attention here at 32x32
            SelfAttention(ngf*8),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            # Add self-attention here at 128x128
            SelfAttention(ngf*2),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

class ImprovedDiscriminator(nn.Module):
    def __init__(self, ndf, nc, num_classes, img_size):
        super().__init__()
        # Smaller label embedding + projection
        self.label_emb = nn.Embedding(num_classes, 128)
        self.label_proj = nn.Sequential(
            nn.Linear(128, img_size * img_size),
            nn.LeakyReLU(0.2)
        )
        
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(nc+1, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),  # No BatchNorm
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),
            # Add self-attention at 32x32
            SelfAttention(ndf*4),
            spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf*8, ndf*16, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf*16, ndf*32, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf*32, 1, 4, 1, 0))
        )
```

## Summary

Your architecture is solid and follows best practices, but there are some improvements that could significantly enhance performance:

**Must Fix:**
- Discriminator label embedding size (currently 65K dims for 6 classes!)
- BatchNorm + Spectral Norm conflict

**Should Add:**
- FID score for evaluation
- EMA for better generation
- Self-attention for detail capture

**Nice to Have:**
- Perceptual loss for medical realism
- Multi-scale discriminator
- Better loss functions (Hinge/WGAN)

Your current architecture will work, but these improvements could take it from "good" to "publication-quality"! 🚀
