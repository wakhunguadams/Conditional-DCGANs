import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torchvision.models as models

class MappingNetwork(nn.Module):
    def __init__(self, nz, embed_dim, style_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(nz + embed_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim)
        )

    def forward(self, z, labels):
        y = self.label_emb(labels)
        x = torch.cat([z, y], dim=1)
        return self.net(x)

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, style):
        style = self.fc(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        return self.norm(x) * (1 + gamma) + beta

class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.adain1 = AdaIN(style_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.adain2 = AdaIN(style_dim, out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.activation(self.adain1(self.conv1(x), style))
        x = self.activation(self.adain2(self.conv2(x), style))
        return x

class Generator(nn.Module):
    def __init__(self, nz=512, style_dim=512, n_classes=6, ngf=64):
        super().__init__()
        self.mapping = MappingNetwork(nz, 128, style_dim, n_classes)
        self.const = nn.Parameter(torch.randn(1, ngf*16, 4, 4))
        
        self.block1 = SynthesisBlock(ngf*16, ngf*8, style_dim, upsample=True)  # 8x8
        self.block2 = SynthesisBlock(ngf*8, ngf*4, style_dim, upsample=True)   # 16x16
        self.block3 = SynthesisBlock(ngf*4, ngf*2, style_dim, upsample=True)   # 32x32
        self.block4 = SynthesisBlock(ngf*2, ngf*2, style_dim, upsample=True)   # 64x64
        self.block5 = SynthesisBlock(ngf*2, ngf, style_dim, upsample=True)     # 128x128
        self.block6 = SynthesisBlock(ngf, ngf, style_dim, upsample=True)       # 256x256
        
        self.to_rgb = nn.Sequential(
            nn.Conv2d(ngf, 3, 1, 1, 0),
            nn.Tanh()
        )

    def forward(self, z, labels):
        style = self.mapping(z, labels)
        x = self.const.repeat(z.size(0), 1, 1, 1)
        x = self.block1(x, style)
        x = self.block2(x, style)
        x = self.block3(x, style)
        x = self.block4(x, style)
        x = self.block5(x, style)
        x = self.block6(x, style)
        return self.to_rgb(x)

class Discriminator(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        # Use EfficientNet as a frozen feature extractor
        backbone = models.efficientnet_b0(pretrained=True)
        self.features = backbone.features
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

        # Independent heads for different resolutions
        self.head1 = self._make_head(24)   # Level 2: 64x64
        self.head2 = self._make_head(40)   # Level 3: 32x32
        self.head3 = self._make_head(112)  # Level 5: 16x16
        self.head4 = self._make_head(320)  # Level 7: 8x8
        
        # Project features into common space for label matching
        self.proj = spectral_norm(nn.Conv2d(320, 512, 1)) # Map High-res features to label space
        self.cls_emb = nn.Embedding(n_classes, 512)

    def _make_head(self, in_channels):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 128, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 1, 3, 1, 1))
        )

    def forward(self, x, labels):
        # EfficientNet features
        feats = []
        curr_x = x
        for i, layer in enumerate(self.features):
            curr_x = layer(curr_x)
            if i in [2, 3, 5, 7]: # Capture features at different depths
                feats.append(curr_x)
        
        # 1. Standard Adversarial Score (Multiple Scales)
        outputs = []
        outputs.append(self.head1(feats[0])) # 64x64
        outputs.append(self.head2(feats[1])) # 32x32
        outputs.append(self.head3(feats[2])) # 16x16
        outputs.append(self.head4(feats[3])) # 8x8
        
        base_out = sum([out.view(out.size(0), -1).mean(dim=1) for out in outputs]) / len(outputs)
        
        # 2. Projection Discrimination (Class Conditioning)
        # Use the highest-level features for class matching
        h_feat = self.proj(feats[3]) # [B, 512, 8, 8]
        h_feat = h_feat.mean(dim=[2, 3]) # Global Average Pool -> [B, 512]
        
        c_emb = self.cls_emb(labels) # [B, 512]
        proj_out = (h_feat * c_emb).sum(dim=1) # Dot product matching
        
        return base_out + proj_out
