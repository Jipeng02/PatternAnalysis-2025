# MambaIRv2 for Grayscale Image Colorization

**Student ID:** s4839921  
**Project:** Pattern Analysis - Image Colorization using State Space Models

---

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Algorithm Description](#algorithm-description)
- [Architecture Visualization](#architecture-visualization)
- [How It Works](#how-it-works)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Training Strategy](#training-strategy)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Example Results](#example-results)
- [Data Splits](#data-splits)
- [References](#references)
- [Reproducibility](#reproducibility)

---

## Overview

This project implements **MambaIRv2**, a state-space model (SSM) based architecture for automatic grayscale image colorization. The model leverages the Selective Structured State Space Model (S4) with visual perception enhancements to predict plausible color channels (a, b in CIE Lab color space) from grayscale luminance (L channel) inputs.

The implementation is based on the MambaIR architecture [1] and adapted for the colorization task with custom loss functions combining perceptual quality (LPIPS) and color accuracy (Lab color distance).

---

## Problem Statement

**Grayscale Image Colorization** is the task of automatically predicting plausible color information for grayscale images. This is an ill-posed problem since a single grayscale image can correspond to multiple valid colorizations. The challenge is to:

1. Learn semantic understanding of objects (e.g., sky → blue, grass → green)
2. Preserve fine-grained texture details
3. Generate perceptually realistic and vibrant colors
4. Avoid color bleeding and artifacts at object boundaries

This implementation addresses these challenges using a state space model with:
- **Long-range dependency modeling** via Mamba blocks
- **Perceptual loss** (LPIPS) for photorealistic quality
- **Lab color space** optimization for better perceptual uniformity

---

## Algorithm Description

**MambaIRv2** is a hierarchical vision transformer that replaces traditional self-attention with **Selective State Space Models (Mamba)**, offering:

- **Linear complexity** O(N) vs. quadratic O(N²) in transformers
- **Long-range context** modeling for semantic understanding
- **Hierarchical feature extraction** with 6-layer deep architecture
- **Window-based processing** (16×16 patches) for efficiency
- **Convolutional feed-forward networks** for local detail preservation

### Key Components:
1. **Shallow feature extraction** (`conv_first`): 3×3 Conv layer
2. **Deep feature extraction**: 6 Mamba-based transformer blocks
3. **Reconstruction** (`conv_after_body`, `conv_last`): Residual refinement + output projection
4. **Loss functions**:
   - **Lab L1 Loss**: Weighted color (a,b) and luminance (L) difference
   - **LPIPS Loss**: VGG-based perceptual similarity
   - **Combined**: Lab + LPIPS for semantic coherence and realism

---

## Architecture Visualization

```
Input Grayscale (1 channel)
         ↓ [Stack to 3 channels]
    [B, 3, 128, 128]
         ↓
   ┌─────────────┐
   │ conv_first  │  ← 3×3 Conv, embed_dim=174
   │   (3→174)   │
   └─────────────┘
         ↓
   ┌─────────────┐
   │  Mamba × 6  │  ← 6 Hierarchical Transformer Blocks
   │   Blocks    │    • Window size: 16×16
   │             │    • d_state: 16 (SSM state dimension)
   │  depths:    │    • num_heads: [6,6,6,6,6,6]
   │ [6,6,6,6,6,6]│   • ConvFFN kernel: 5×5
   └─────────────┘
         ↓
   ┌─────────────┐
   │conv_after   │  ← Residual connection
   │   _body     │
   └─────────────┘
         ↓
   ┌─────────────┐
   │  conv_last  │  ← Output projection (174→3)
   └─────────────┘
         ↓
  Output RGB (3 channels)
    [B, 3, 128, 128]
```

### Loss Computation Flow:
```
Predicted RGB ──┬──→ RGB→Lab ──→ L1(a,b channels) ──→ loss_ab
                │
                └──→ LPIPS(VGG) ───────────────────→ loss_lpips
                
Ground Truth RGB ─→ RGB→Lab ──→ L1(L channel) ────→ loss_L

Total Loss = w_ab·loss_ab + w_L·loss_L + w_lpips·loss_lpips
```

---

## How It Works

### 1. **Input Processing**
- Grayscale images are converted to 3-channel tensors (stacked replication)
- Resized to 128×128 for consistent processing
- Images are cropped to multiples of window_size (16) to prevent alignment issues

### 2. **Color Space Transformation**
- RGB images are converted to **CIE Lab** color space:
  - **L channel**: Lightness (0-100)
  - **a channel**: Green ↔ Red (-128 to 127)
  - **b channel**: Blue ↔ Yellow (-128 to 127)
- Lab space is perceptually uniform, making L1 distances more meaningful

### 3. **Two-Stage Training**

**Stage 1: Shallow Layer Fine-tuning** (5 epochs)
- Freeze backbone (Mamba blocks)
- Train only `conv_first`, `conv_after_body`, `conv_last`
- Dataset: COCO 2017 Validation (5,000 images), no augmentation
- Loss: Lab L1 only (w_ab=1.0, w_L=0.15)
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)

**Stage 2: Full Model Training** (15 epochs)
- Unfreeze all parameters
- Dataset: COCO 2017 Validation with augmentation (brightness, contrast, saturation, hue)
- Loss: Lab L1 + LPIPS (w_lpips=0.5)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=5, eta_min=1e-6)
- Gradient accumulation: 4 steps (effective batch size = 4)

### 4. **Inference**
- Input: Grayscale image (any size)
- Crop to window_size multiples
- Forward pass through model
- Output: RGB colorized image (clipped to [0,1])

---

## Dependencies

### Core Libraries
| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.1.1+cu118 | Deep learning framework |
| `torchvision` | 0.16.1+cu118 | Image transformations |
| `torchaudio` | 2.1.1+cu118 | Audio processing (dependency) |
| `mamba-ssm` | 1.0.1 | Selective State Space Model |
| `causal-conv1d` | 1.1.1 | Causal convolutions for Mamba |
| `lpips` | 0.1.4 | LPIPS perceptual loss |
| `einops` | 0.8.1 | Tensor rearrangement operations |
| `Pillow` | 11.3.0 | Image I/O |
| `numpy` | 1.26.4 | Numerical operations |
| `opencv-python` | 4.9.0.80 | Computer vision utilities |

### Dataset & Data Processing
| Package | Version | Purpose |
|---------|---------|---------|
| `datasets` | 4.3.0 | HuggingFace datasets (ImageNet) |
| `pandas` | 2.3.3 | Data manipulation |
| `pyarrow` | 22.0.0 | Efficient columnar data |
| `dill` | 0.4.0 | Extended pickling support |
| `multiprocess` | 0.70.16 | Parallel data processing |
| `xxhash` | 3.6.0 | Fast hashing for datasets |

### Model & Training
| Package | Version | Purpose |
|---------|---------|---------|
| `timm` | 1.0.20 | Vision model utilities |
| `transformers` | 4.30.2 | Transformer utilities |
| `peft` | 0.17.1 | Parameter-efficient fine-tuning |
| `accelerate` | 1.10.1 | Distributed training |
| `huggingface-hub` | 0.35.3 | Model hub integration |
| `safetensors` | 0.6.2 | Safe tensor serialization |
| `tokenizers` | 0.13.3 | Fast text tokenization |

### Utilities
| Package | Version | Purpose |
|---------|---------|---------|
| `tqdm` | 4.67.1 | Progress bars |
| `scipy` | 1.15.3 | Scientific computing |
| `PyYAML` | 6.0.3 | Configuration files |
| `requests` | 2.32.5 | HTTP library |
| `fsspec` | 2025.9.0 | Filesystem spec |
| `filelock` | 3.19.1 | Platform-independent file locking |

### Networking & HTTP
| Package | Version | Purpose |
|---------|---------|---------|
| `aiohttp` | 3.13.1 | Async HTTP client/server |
| `httpx` | 0.28.1 | Next-gen HTTP client |
| `httpcore` | 1.0.9 | HTTP core library |
| `aiohappyeyeballs` | 2.6.1 | Fast async DNS resolver |

### System & Development
| Package | Version | Purpose |
|---------|---------|---------|
| `psutil` | 7.1.0 | System monitoring |
| `regex` | 2025.9.18 | Advanced regex operations |
| `redo` | 3.0.0 | Retry operations |
| `SQLAlchemy` | 2.0.44 | SQL toolkit |
| `Jinja2` | 3.1.6 | Template engine |

### CUDA Requirements
- **CUDA**: 11.8 (compatible with torch 2.1.1+cu118)
- **Triton**: 2.1.0 (for efficient CUDA kernels)
- **GPU**: NVIDIA A100 (≥40GB VRAM) or better recommended
  - Minimum compute capability: 8.0 (Ampere architecture)
  - Training requires >20 GB VRAM due to model size and batch processing
- **Not recommended**: Consumer GPUs with <20GB VRAM (RTX 3090, RTX 4090, V100 16GB)

### Complete Dependencies
See `requirements.txt` for the complete list of all dependencies including:
- **Math & Computation**: sympy (1.14.0), mpmath (1.3.0), networkx (3.3)
- **Networking**: Twisted (25.5.0), Automat (25.4.16), hyperlink (21.0.0)
- **Data Formats**: simplejson (3.20.2), tomli (2.3.0), furl (2.1.4)
- **Build Tools**: ninja (1.13.0), buildtools (1.0.6), docopt (0.6.2)
- **Additional**: hf-xet (1.1.10), orderedmultidict (1.0.1), and more

Total: **70+ packages** ensuring full compatibility across different systems and use cases.

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/Jipeng02/PatternAnalysis-2025.git
cd PatternAnalysis-2025/recognition/s4839921_mamba
```

### 2. Install Dependencies
```bash
# Install all required packages from requirements.txt
pip install -r requirements.txt
```

**Note**: This project requires a **Linux** environment for proper compilation and execution.
- **mamba-ssm** and **causal-conv1d** require CUDA 11.8 toolkit and a compatible C++ compiler (gcc 7+)
- **Triton** kernels are optimized for Linux systems
- macOS and Windows are not supported

### 3. Download Pretrained Weights
Download the pretrained MambaIRv2 model for color denoising:

```bash
# Download from MambaIR official release
wget https://github.com/csguoh/MambaIR/releases/download/v1.0/mambairv2_ColorDN_15.pth

# Or using curl
curl -L https://github.com/csguoh/MambaIR/releases/download/v1.0/mambairv2_ColorDN_15.pth -o mambairv2_ColorDN_15.pth
```

**Alternative**: Download manually from [MambaIR Releases](https://github.com/csguoh/MambaIR/releases/tag/v1.0) and place the file in the project root directory.

**Required File:**
- `mambairv2_ColorDN_15.pth` - Pretrained MambaIRv2 weights for color denoising

**Note**: The pretrained weights are essential for training. They provide a good initialization point learned from image restoration tasks, which helps the model converge faster on the colorization task.

### 4. Verify Installation
```bash
python -c "import torch; import mamba_ssm; import lpips; print('✓ All dependencies installed')"
python -c "print('CUDA available:', torch.cuda.is_available())"
```

### Quick Start Summary
```bash
# Complete installation in one go:
git clone https://github.com/Jipeng02/PatternAnalysis-2025.git
cd PatternAnalysis-2025/recognition/s4839921_mamba

pip install -r requirements.txt

# Download pretrained weights
wget https://github.com/csguoh/MambaIR/releases/download/v1.0/mambairv2_ColorDN_15.pth

# Download dataset
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d ./datasets_COCO2017/

# Verify setup
python -c "import torch, mamba_ssm, lpips; print('✓ Ready to train!')"
```

---

## Dataset & Preprocessing

### Dataset
- **Source**: [COCO 2017 Validation Set](http://cocodataset.org/#download)
- **Size**: 5,000 RGB images
- **License**: Creative Commons Attribution 4.0

### Download Instructions
```bash
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d ./datasets_COCO2017/
```

### Preprocessing Pipeline

#### Stage 1 (No Augmentation)
```python
transform = T.Compose([
    T.Resize((128, 128)),  # Resize to fixed size
    T.ToTensor()           # Convert to [0,1] tensor
])
```

#### Stage 2 (With Augmentation)
```python
augmentations = [
    ColorJitter(brightness=0.3),  # ±30% brightness
    ColorJitter(contrast=0.3),    # ±30% contrast
    ColorJitter(saturation=0.4),  # ±40% saturation
    ColorJitter(hue=0.1)          # ±10% hue shift
]
# Applied with 70% probability per sample
```

**Justification**: 
- **Stage 1**: Clean data ensures stable convergence of shallow layers
- **Stage 2**: Augmentation improves robustness to lighting variations and prevents overfitting
- **Brightness/Contrast**: Simulates different exposure conditions
- **Saturation/Hue**: Encourages model to learn color distributions rather than memorizing

### Grayscale Conversion
```python
# Convert RGB to grayscale using standard weights
gray = 0.299*R + 0.587*G + 0.114*B
# Stack to 3 channels for model compatibility
gray_input = gray.repeat(3, 1, 1)
```

---

## Training Strategy

### Data Splits

| Split | Images | Purpose | Augmentation |
|-------|--------|---------|--------------|
| **Training** | 5,000 | Model training | Stage 1: No<br>Stage 2: Yes |
| **Validation** | 0 | Not used (limited dataset) | N/A |
| **Test** | User-provided | Inference | No |

**Justification**:
- **No validation split**: With only 5,000 images, splitting would reduce training data significantly
- **Early stopping not used**: Fixed epochs (5 + 15) based on empirical convergence
- **Model selection**: Best model saved based on training loss (Stage 2)
- **Generalization**: Augmentation in Stage 2 acts as regularization


### Training Configuration

```python
# Stage 1
epochs = 5
batch_size = 2
learning_rate = 3e-4
weight_decay = 1e-4
loss_weights = {w_ab: 1.0, w_L: 0.15}

# Stage 2
epochs = 15
batch_size = 1 (accumulation_steps=4)
learning_rate = 1e-4 → 1e-6 (cosine decay)
weight_decay = 1e-4
loss_weights = {w_ab: 1.0, w_L: 0.15, w_lpips: 0.5}
```

### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Prevents gradient explosion in deep Mamba blocks.

---

## Usage

### Training

This project provides **two training approaches**:

#### Option 1: Standard Training (COCO Dataset - Two-Stage)

**Prerequisites**: 
- Ensure `mambairv2_ColorDN_15.pth` is in the project root directory
- Download COCO 2017 Validation dataset to `./datasets_COCO2017/val2017/`

**Run Training:**
```bash
python train.py
```

**Training Process:**
- **Stage 1** (5 epochs): Shallow layer fine-tuning
  - Freezes Mamba blocks (backbone)
  - Trains only conv_first, conv_after_body, conv_last
  - Uses clean COCO images without augmentation
  - Loss: Lab L1 only (w_ab=1.0, w_L=0.15)
  - Output: `./checkpoints/stage_1.pth`

- **Stage 2** (15 epochs): Full model training
  - Unfreezes all parameters
  - Uses augmented COCO images (color jitter)
  - Loss: Lab L1 + LPIPS (w_lpips=0.5)
  - Outputs:
    - `./checkpoints/stage_2.pth` (final)
    - `./checkpoints/best_stage_2.pth` (lowest loss)

---

#### Option 2: Improved Training (ImageNet Dataset - Single-Stage)

**Prerequisites**: 
- Ensure `mambairv2_ColorDN_15.pth` is in the project root directory
- Ensure at least more 7GB disk for dataset cache
- **No manual dataset download required** - Dataset loads automatically from HuggingFace

**Basic Usage:**
```bash
python train_improved.py
```

**Advanced Usage with Custom Parameters:**
```bash
# Train with 200 classes (instead of default 100)
python train_improved.py --num_classes 200

# Train with more samples per epoch
python train_improved.py --samples_per_epoch 5000

# Train for more epochs with different learning rate
python train_improved.py --epochs 15 --lr 1e-4

# Full ImageNet training (1000 classes, will take ~100+ hours)
python train_improved.py --num_classes 1000 --samples_per_epoch 10000 --epochs 20
```

**Available Arguments:**
```bash
--num_classes         Number of ImageNet classes (default: 100, max: 1000)
--samples_per_epoch   Samples per epoch (default: 3000)
--epochs              Number of epochs (default: 10)
--batch_size          Batch size (default: 1)
--accumulation_steps  Gradient accumulation (default: 4)
--lr                  Learning rate (default: 5e-5)
--pretrained_path     Path to pretrained weights (default: ./mambairv2_ColorDN_15.pth)
--save_dir            Checkpoint directory (default: ./checkpoints_imagenet)
--vis_dir             Visualization directory (default: ./vis_imagenet)
```

**Training Features:**
- **Dynamic Sampling**: Randomly samples from dataset each epoch for better generalization
- **NaN Handling**: Robust error detection and recovery
- **Safe Loss Functions**: Improved Lab conversion with clamping
- **Automatic Visualization**: Saves comparison images every 2 epochs
- **Gradient Accumulation**: Effective batch size = batch_size × accumulation_steps

**Outputs:**
- `./checkpoints_imagenet/best_model.pth` - Best model (lowest loss)
- `./checkpoints_imagenet/final_model.pth` - Final model
- `./vis_imagenet/epoch_*.png` - Visualization grids (grayscale | prediction | ground truth)

**Example Visualization Grid:**
```
Row 1: Grayscale inputs (4 samples)
Row 2: Model predictions (4 samples)
Row 3: Ground truth colors (4 samples)
```

---

#### Which Training Method to Choose?

| Aspect | Standard (train.py) | Improved (train_improved.py) |
|--------|---------------------|------------------------------|
| **Dataset** | COCO 2017 (5K images, 80 classes) | ImageNet-1K (1.28M images, 1000 classes) |
| **Training Time** | ~20 hours (Stage 1+2) | ~12-100+ hours (depends on num_classes) |
| **Good For** | Natural landscapes, cool tones | Diverse objects, warm tones |
| **Limitations** | Weak on flowers, faces, sunsets | Requires HuggingFace, more compute |
| **Setup** | Manual dataset download | Automatic dataset loading |
| **Recommended For** | Quick baseline, limited compute | Better generalization, more diversity |

**Recommendation:**
- **Start with `train.py`** if you want quick results on natural landscapes
- **Use `train_improved.py`** if you want better warm-tone performance and have compute resources

---

### Inference

#### Single Image
```python
from module import MambaIRv2
import torch
from PIL import Image
import torchvision.transforms as T

# Load model
model = MambaIRv2(...).to('cuda')
model.load_state_dict(torch.load('final_model.pth')['params'])
model.eval()

# Process image
img = Image.open('input.jpg').convert('L')
gray = T.ToTensor()(img).repeat(3,1,1).unsqueeze(0).cuda()

with torch.no_grad():
    colorized = model(gray).clamp(0,1)

# Save result
T.ToPILImage()(colorized.squeeze(0).cpu()).save('output.jpg')
```

#### Batch Inference
```bash
# Edit predict.py to set:
# - img_dir = '/path/to/grayscale/images'
# - model checkpoint path

python predict.py
# Output: {img_dir}/predict_output/*.jpg
```

**Script Features**:
- Automatically finds all images (png, jpg, jpeg, bmp, tif, tiff)
- Preserves original filenames
- Handles variable image sizes (crops to window_size multiples)
- GPU acceleration when available

---

## Example Results

### Test Dataset
- **Source**: [Greyscale Colorization Dataset](https://github.com/gayanku/greyscale-colorization)
- **Inference Images**: All images starting with "P" in the test folder
- **Inference Results**: [Google Drive - Colorized Output](https://drive.google.com/drive/folders/1TXJ62DXF53DmY4QGtSMxIKL_Khe8ndNz?usp=share_link)

### Input/Output Comparison

**Successful Cases - Natural Landscapes:**

Images G1, G2, G3, G17, G18, G19 show good colorization results with accurate blue sky and green vegetation identification.

**Challenging Cases - Warm Tones:**

Human faces, flowers, birds, and sunsets show poor colorization due to limited warm tone prediction.

*Complete results available in the [Google Drive folder](https://drive.google.com/drive/folders/1TXJ62DXF53DmY4QGtSMxIKL_Khe8ndNz?usp=share_link)*

### Training Loss Curves

**Stage 1 (5 Epochs) - Lab-ab Loss:**

| Epoch | Avg Lab-ab Loss |
|-------:|----------------:|
| 1 | 18.370 |
| 2 | 18.085 |
| 3 | 17.975 |
| 4 | 17.914 |
| 5 | 17.866 |


![Stage 1 - Lab-ab Loss](./images/stage1_loss.png)

**Stage 1 Limitation - Why Full Model Training is Necessary:**

After 5 epochs of training only the shallow layers (`conv_first`, `conv_last`, `conv_after_body`), the model shows limited colorization capability:

| Input (Grayscale) | Stage 1 Output | Ground Truth |
|:-----------------:|:--------------:|:------------:|
| ![Grayscale Input](./images/input_grayscale_stage_1.png) | ![Stage 1 Prediction](./images/predicted_stage_1.png) | ![Ground Truth](./images/ground_truth_stage_1.png) |

**Observations:**
- ✗ **Global color filter effect**: The model applies an overall color tint rather than semantic colorization
- ✗ **No object-level understanding**: Cannot distinguish sky, grass, buildings - everything gets similar color treatment
- ✗ **Loss convergence plateau**: As shown below, the loss converges quickly and plateaus around 17.87

![Stage 1 Convergence Analysis](./images/stage1_contine.png)

**Why this happens:**
1. **Frozen backbone**: The 6 Mamba blocks (which provide semantic understanding) remain frozen
2. **Limited capacity**: Only shallow conv layers cannot learn complex object-color mappings
3. **Feature extraction bottleneck**: Without training the Mamba blocks, features lack semantic richness

**Solution:** Unfreeze all parameters in Stage 2 to enable the model to learn proper semantic colorization through the full network, including the Mamba blocks that provide long-range context and object understanding.

---

**Stage 2 (15 Epochs) - Lab Loss:**

The training was conducted in 3 runs of 5 epochs each:

| Epoch | Lab Loss | Run |
|-------|----------|-----|
| 1 | 17.417 | Run 1 |
| 2 | 15.890 | Run 1 |
| 3 | 14.700 | Run 1 |
| 4 | 13.829 | Run 1 |
| 5 | 13.215 | Run 1 |
| 6 | 14.096 | Run 2 |
| 7 | 13.839 | Run 2 |
| 8 | 13.433 | Run 2 |
| 9 | 12.817 | Run 2 |
| 10 | 12.354 | Run 2 |
| 11 | 13.327 | Run 3 |
| 12 | 13.133 | Run 3 |
| 13 | 12.688 | Run 3 |
| 14 | 12.084 | Run 3 |
| 15 | 11.595 | Run 3 ← Best |

![Stage 2 - Total Loss (Lab + LPIPS)](./images/stage2_loss.png)

**Training Observations:**
- Stage 1 shows steady convergence with Lab-ab loss decreasing from 18.37 to 17.87
- Stage 2 demonstrates significant improvement across 15 epochs, from 17.42 to 11.60
- Best model achieved at Epoch 15 with Lab Loss of 11.595
- Training conducted in 3 runs due to resource constraints, each run continuing from previous checkpoint

---

### Stage 2 Results - Visual Comparison

After full model training (all parameters unfrozen), the model shows significant improvement in semantic colorization. Below are representative examples showing both strengths and limitations:

**✓ Good Example - Natural Landscape (Cool Tones):**

| Test Image | Input (Grayscale) | Output (Colorized) |
|:----------:|:-----------------:|:------------------:|
| **1** | ![G1 Input](./images/G_1.jpg) | ![G1 Output](./images/P_1.jpg) |


*Success: The model correctly identifies and colorizes natural landscapes with blue sky and green vegetation, demonstrating strong semantic understanding for cool-toned outdoor scenes.*

**✗ Bad Example - Warm Tones:**

| Test Image | Input (Grayscale) | Output (Colorized) |
|:----------:|:-----------------:|:------------------:|
| **21** | ![G21 Input](./images/G_21.jpg) | ![G21 Output](./images/P_21.jpg) |


*Failure: The model struggles with warm-toned subjects (human faces, flowers, sunsets). Colors appear muted, desaturated, or incorrectly mapped, highlighting the cool-tone bias from training data.*

---

### Qualitative Analysis

**Strengths:**
- ✓ **Excellent natural landscape colorization**: Successfully colorizes natural scenes (G1, G2, G3, G17, G18, G19)
- ✓ **Accurate semantic understanding of cool tones**: Correctly identifies blue sky and green vegetation/trees
- ✓ **Strong sky-ground segmentation**: Properly separates and colorizes different landscape elements
- ✓ **Consistent natural color mapping**: Reliable performance on outdoor/nature scenes with cool color palettes
- ✓ **Smooth color transitions**: No visible bleeding artifacts in landscape images
- ✓ **Fine texture preservation**: Maintains grayscale details while adding color

**Limitations:**
- ✗ **Poor warm tone prediction**: Cannot accurately colorize warm colors (red, orange, yellow, pink)
- ✗ **Skin tone challenges**: Human faces and portraits not properly colorized
- ✗ **Flower colorization failure**: Red, pink, and yellow flowers appear muted or incorrect
- ✗ **Bird plumage issues**: Colorful feathers not accurately rendered
- ✗ **Sunset/sunrise weakness**: Warm orange and red sunset tones missing or desaturated
- ✗ **Limited warm color diversity**: Model biased toward cool tones (blues, greens) from training data

**Root Cause Analysis:**

The model's bias toward cool tones (blue/green) and failure on warm tones (red/orange/yellow) stems from **severe class imbalance and domain bias in the COCO 2017 Validation dataset**:

**1. Dataset Composition - Urban/Indoor Dominance:**

COCO ("Common Objects in Context") is designed for object detection in **urban and indoor scenes**, not general-purpose colorization. The top categories reveal this bias:

- **Urban objects (>2,500 images):** `person` (2,693), `car` (535), `traffic light` (191), `truck` (250), `bus` (189), `stop sign` (69)
- **Indoor objects (>2,500 images):** `chair` (580), `dining table` (501), `cup` (390), `bottle` (379), `couch` (195), `tv` (207)

These scenes are dominated by **neutral cool tones**:
- Sky → blue (outdoor urban scenes)
- Roads/buildings → gray/brown (asphalt, concrete)
- Trees/grass → green (parks, street vegetation)
- Indoor walls → white/beige (typical interior design)

The model learned these repetitive, high-frequency color patterns very well, leading to excellent performance on natural landscapes with blue sky and green vegetation.

**2. Missing Warm-Tone Categories:**

Critical warm-tone subjects are either **absent** or **severely underrepresented**:

- **`flower`**: **Not included in COCO's 80 classes** - The model has never been explicitly trained to recognize flowers as important objects requiring vibrant colors (red, pink, yellow, purple)
- **`bird`**: Only 125 images - Insufficient to learn the diverse plumage colors (from brown sparrows to colorful parrots)
- **`sunset/sunrise`**: **Not a COCO category** - COCO focuses on objects, not lighting conditions or atmospheric scenes. Most images are taken in daytime or well-lit indoor conditions. The model lacks exposure to the global warm-tone lighting (orange, red, purple) characteristic of golden hour photography

**3. Human Face Limitation - Annotation Scale Mismatch:**

Although `person` is the #1 category (2,693 images), this does **not** translate to good skin tone colorization:

- **Annotation scope:** COCO's `person` boxes typically cover the **entire body** (full-body shots of people walking, cycling, playing sports)
- **Lack of close-ups:** Very few high-resolution **facial portraits** exist in the dataset
- **Pixel-level sparsity:** Even when faces appear, they occupy only a small fraction of pixels in distant/full-body shots
- **Result:** The model learned to colorize "human shapes" (clothing, body contours) but lacks sufficient fine-grained data to learn the **subtle, continuous warm gradients** of human skin tones (varying complexions, blush, warm undertones)

**4. Lab Color Space Imbalance:**

The training data distribution created an **asymmetric representation in Lab space**:

- **a-channel** (green ↔ red): Over-representation of negative values (green vegetation) vs. positive values (red/pink objects)
- **b-channel** (blue ↔ yellow): Over-representation of negative values (blue sky) vs. positive values (yellow/orange objects)

The L1 loss with equal weighting (`w_ab=1.0`) failed to correct this imbalance, as the model optimized for the **majority class colors** (cool tones).

**5. "Garbage In, Garbage Out" - Data Quality Issue:**

This is a classic case of data bias propagation. The model's failures are not architectural limitations of MambaIRv2, but rather **faithful reproduction of COCO's domain-specific biases**:

- **What COCO provides:** Urban streets, indoor homes, common objects with neutral/cool colors
- **What COCO lacks:** Portraits, flowers, artistic lighting, warm-toned natural scenes
- **Model behavior:** Learns to colorize the world as "COCO-like" → excellent on streets/parks, poor on portraits/flowers/sunsets

---

### Improvement Attempts

#### Experiment: ImageNet-1K Fine-tuning

To address the warm-tone bias issue, I conducted an additional experiment using a more diverse dataset:

**Dataset Change:**
- **New Dataset**: [ImageNet-1K-128x128](https://huggingface.co/datasets/benjamin-paine/imagenet-1k-128x128)
- **Dataset Statistics**: 1,000 categories, 1.28M images total
- **Subset Used**: 100 randomly selected categories (due to computational constraints)
- **Rationale**: ImageNet contains more diverse object categories including flowers, animals, and various natural scenes with warmer color palettes

**Training Configuration:**
- **Training Time**: ~12 hours on A100 40GB
- **Epochs**: 15 (full model fine-tuning)
- **Parameters**: All parameters unfrozen
- **Results**: [Google Drive - ImageNet Fine-tuned Results](https://drive.google.com/drive/folders/1XR9T2PQXWMKC2Og2bDcVpdEJWSebdHpS?usp=share_link)

**Training Loss Curve (ImageNet - 15 Epochs):**

The improved training on ImageNet dataset shows comprehensive loss tracking across 15 epochs:

![ImageNet Training Loss](./images/improved_training_loss.png)

**Loss Analysis:**
- **Total Loss (Blue)**: Decreased from 15.7 to 13.6, showing steady convergence
- **Lab Loss (Green)**: Mirrors total loss closely, indicating Lab color space optimization is the dominant component
- **LPIPS Loss (Red)**: Decreased from 0.245 to 0.18, demonstrating improved perceptual quality
- **Best Model**: Achieved at Epoch 15 with Total Loss of 13.62

**Key Observations:**
- ✓ **Smooth convergence**: No significant oscillations or instability throughout training
- ✓ **Consistent improvement**: Both Lab and LPIPS losses decrease together, indicating balanced optimization
- ✓ **Lower final loss vs COCO**: ImageNet training achieves 13.62 compared to COCO's 11.60, but note that different datasets make direct loss comparison less meaningful
- ⚠ **Plateauing trend**: Loss curve shows some flattening after Epoch 10, suggesting potential benefit from longer training

---

**Comparative Results:**

Below are side-by-side comparisons showing improvements and trade-offs:

**Case 1: Flower Colorization (Improvement) ✓**

| COCO-trained (Old) | ImageNet-trained (New) | Notes |
|:------------------:|:----------------------:|:------|
| ![P16 COCO](./images/P_16.jpg) | ![G16 ImageNet](./images/G_16.jpg) | **Old (COCO):**<br>✗ Fails to colorize petals<br>✗ Neutral/desaturated tones<br><br>**New (ImageNet):**<br>✓ Visible color prediction<br>✓ Warmer tones on petals<br>⚠ Still not perfect |

*Analysis*: The ImageNet-trained model shows clear improvement in flower colorization. While the colors are not perfect, the model now attempts to add warm tones (green) to the petals, whereas the COCO-trained model produced only neutral grays.

**Case 2: Ship/Maritime Scene (Degradation) ✗**

| COCO-trained (Old) | ImageNet-trained (New) | Notes |
|:------------------:|:----------------------:|:------|
| ![P17 COCO](./images/P_17.jpg) | ![G17 ImageNet](./images/G_17.jpg) | **Old (COCO):**<br>✓ Accurate building color<br>✓ Basic color of ship<br><br>**New (ImageNet):**<br>✗ incorrect tones |

*Analysis*: Maybe 100 categories does not contain ship,so it performs bad.

---

**Discussion:**

**Partial Success - Category-Dependent Improvement:**

The ImageNet fine-tuning experiment reveals a **trade-off** rather than universal improvement:

- ✓ **Gains**: Better warm-tone prediction for flowers and some organic subjects (previously missing in COCO)
- ✗ **Losses**: Degraded performance on other scenes (well-represented in COCO)
- ⚠ **Root Cause**: Only 100/1,000 ImageNet categories were used due to computational limits

**Persistent Cool-Tone Bias:**

An intriguing observation: **Regardless of dataset changes, the model consistently favors cool tones (blue, green) over warm tones (red, orange, yellow).**

This persistent bias across different datasets suggests the issue may not be purely data-driven, but could stem from:

1. **Loss Function Limitation**: 
   - Lab L1 loss treats all color channels equally (`w_ab=1.0`)
   - Cool tones (negative a/b values) may be easier to predict due to lower variance in natural images
   - Warm tones require precise, high-magnitude positive a/b values, which are penalized more heavily by L1 loss

2. **Model Architecture Bias**:
   - MambaIRv2 was pretrained on image denoising tasks (neutral color preservation)
   - The architecture may inherently favor conservative color predictions (closer to grayscale = cooler tones)
   - Linear state space models might struggle with high-frequency color variations (vibrant reds/yellows)

3. **Incomplete Dataset Coverage**:
   - **Computational Feasibility**: Training on full ImageNet-1K (1.28M images, 1,000 categories) would require ~100+ hours
   - **Current Limitation**: Only 100 categories tested (10% of dataset diversity)
   - **Hypothesis**: Full ImageNet coverage might resolve remaining warm-tone issues, but this remains unverified

**Future Work:**

To conclusively determine whether the warm-tone bias is dataset-related or architectural:

1. **Full ImageNet Training** (~100-150 hours on A100):
   - Train on all 1,000 categories with 1.28M images
   - If warm-tone issues persist → architectural/loss function problem
   - If issues resolve → dataset coverage problem

2. **Loss Function Redesign**:
   - Weighted Lab loss with higher penalty for warm-tone errors: `w_a_pos > w_a_neg`, `w_b_pos > w_b_neg`
   - Perceptual loss specifically for warm color regions
   - Adversarial loss to encourage vibrant colorization

3. **Architecture Modifications**:
   - Replace pretrained weights with colorization-specific pretraining
   - Add color-specific attention mechanisms
   - Multi-scale color prediction heads

**Current Limitation Acknowledgment:**

Due to time and computational constraints (100+ hours for full ImageNet training), the root cause of the persistent cool-tone bias cannot be definitively determined. The partial ImageNet experiment (100 categories, 12 hours) shows promising but inconsistent improvements, suggesting that dataset diversity is a contributing factor, though potentially not the sole cause.


---

## References

[1] **MambaIR: A Simple Baseline for Image Restoration with State-Space Model**  
    Hang Guo, Jinmin Li, Tao Dai, Zhihao Ouyang, Xudong Ren, Shu-Tao Xia  
    *arXiv preprint arXiv:2402.15648*, 2024  
    [GitHub](https://github.com/csguoh/MambaIR)

[2] **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**  
    Albert Gu, Tri Dao  
    *arXiv preprint arXiv:2312.00752*, 2023

[3] **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)**  
    Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang  
    *CVPR 2018*  
    [GitHub](https://github.com/richzhang/PerceptualSimilarity)

[4] **COCO 2017 Validation Dataset**  
    [Hugging Face Dataset Viewer](https://huggingface.co/datasets/rafaelpadilla/coco2017/viewer/default/val)

---

## Reproducibility

### Hardware Environment
- **GPU**: NVIDIA A100 (≥40GB) or better (H100, A100 80GB)
  - **Minimum VRAM**: 20 GB
  - **Recommended**: A100 40GB or higher
  - **Not suitable**: Consumer GPUs (RTX 3090/4090), V100 16GB
- **RAM**: ≥32 GB
- **Storage**: ≥15 GB (packages and dataset)

### Software Environment
- **OS**: Linux (Ubuntu 20.04+ recommended)
  - **Required**: Linux-based system for CUDA 11.8 and Mamba SSM compilation
  - **Not supported**: macOS, Windows
- **CUDA**: 11.8
- **Python**: 3.8 - 3.11
- **PyTorch**: 2.1.1+cu118
- **Triton**: 2.1.0

### Random Seed (Add to train.py and train_improvement for full reproducibility)
```python
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Expected Training Time (only for train.py)
- **Stage 1**: ~5 hours (5 epochs, ~1 hour per epoch) on A100 40GB
- **Stage 2**: ~15 hours (15 epochs, ~1 hour per epoch) on A100 40GB
- **Total**: ~20 hours on A100 40GB or better

**Note**: Training time is approximately **1 hour per epoch**. H100 or A100 80GB may provide slight speedup but epoch time will remain similar due to model complexity.

### Checkpoint Files (only for train.py)
```
Project Root:
├── mambairv2_ColorDN_15.pth  # Pretrained weights (required for training)

checkpoints/
├── stage_1.pth               # After 5 epochs (shallow layers)
├── best_stage_2.pth          # Lowest loss during Stage 2
└── stage_2.pth               # Final model after 15 epochs
```

### Pretrained Weights
- **File**: `mambairv2_ColorDN_15.pth`
- **Purpose**: Initialization for training (from MambaIR image restoration)
- **Size**: ~100 MB (approximate)

**Why pretrained weights?**
The model is initialized with weights pretrained on image denoising tasks. This provides:
1. Better feature extraction capabilities from the start
2. Faster convergence during training
3. Improved final performance compared to random initialization

### Inference Speed
- **Single 128×128 image**: ~1s (A100GPU)
- **Single 1k image**: ~5s (A100GPU)

---

## License

This project builds upon [MambaIR](https://github.com/csguoh/MambaIR) which is licensed under the Apache License 2.0.

COCO dataset is licensed under Creative Commons Attribution 4.0.

---

## Contact

**Student ID**: s4839921  
**Course**: COMP3710 - Pattern Analysis  
**Institution**: University of Queensland  
**Year**: 2025

For questions or issues, please open an issue in this repository.

---

## Acknowledgments

- MambaIR authors for the base architecture
- COCO dataset creators for training data
- PyTorch and Mamba SSM development teams
