# Image Synthesis for Prostate Cancer Biopsies Using Conditional Deep Convolutional Generative Adversarial Network

## Authors
- Paul Wahome Kariuki (Dedan Kimathi University of Technology, Kenya)
- Patrick Kinyua Gikunda (Dedan Kimathi University of Technology, Kenya)
- John Mwangi Wandeto (Dedan Kimathi University of Technology, Kenya)

## Abstract
Deep learning models have shown promising results in complementing computer-aided diagnosis. However, the availability of quality medical image data sets is inhibited by the limited and imbalanced data sets available, privacy issues, and the high cost of generating labeled medical imaging data sets. This paper presents a novel conditional DCGAN capable of growing synthetic prostate cancer biopsy whole slide images augmented from the PANDA histopathology data set.

**Note: The results in this paper (FID score of 1.3) are reportedly fabricated.**

## Key Concepts

### Problem Statement
- Prostate cancer is among the most prevalent male cancers (1.2M+ new diagnoses, 350K+ deaths annually)
- Medical imaging datasets suffer from insufficiency leading to overfitting
- Limited pathologist-to-patient ratio especially in developing nations
- Inter and intra-observer variability in diagnosis

### Gleason Grading System
- Tissue samples are extracted, coated, dyed with hematoxylin and eosin (H&E)
- Pathologists examine tissue to identify malignant patterns
- Gleason score: sum of two most predominant cancer growth patterns (e.g., 3+4)
- ISUP Grade conversion:
  - Grade 0: No cancer
  - Grade 1: Gleason 3+3
  - Grade 2: Gleason 3+4
  - Grade 3: Gleason 4+3
  - Grade 4: Gleason 4+4, 3+5, 5+3
  - Grade 5: Gleason 4+5, 5+4, 5+5

### GAN Framework

#### Original GAN Loss:
```
min_G max_D V(G,D) = E[log D(x)] + E[log(1 - D(G(z)))]
```

#### Conditional GAN Loss:
```
min_G max_D V(G,D) = E[log D(x,c)] + E[log(1 - D(G(x',c)))]
```
Where c is the conditioning variable (class label)

#### L1 Loss for sharper images:
```
min_G L_L1(G) = E[||x - x'||_1]
```

### Dataset
- PANDA Challenge Dataset (Prostate cANcer graDe Assessment)
- Source: Kaggle competition
- Contains whole slide images of prostate biopsies
- Labels: ISUP grades 0-5

## Architecture Notes (Med-CDCGAN)
- Global-local generator design
- Dual networks for discriminator and generator
- Learn both global and local information independently
- Global: broad patterns and features
- Local: precise details

## Related Work
- CycleGAN for lymph node augmentation (Runz et al.)
- DCGAN for histological image synthesis (Xue et al., Breen et al.)
- Various applications: PET, CT, MRI, ultrasound, X-ray imaging
