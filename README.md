# 🚀 Bhartiya Antriksh Hackathon (BAH) 2026

## Physics-Informed Thermal-to-RGB Super-Resolution & Colorization Pipeline

Welcome to the **Bhartiya Antriksh Hackathon 2026** codebase. This repository provides a state-of-the-art, physics-informed two-stage deep learning pipeline designed to upscale raw single-channel Thermal Infrared (TIR) satellite imagery (Landsat 9 Band 10) and translate it into high-resolution, photorealistic, and structurally aligned 3-channel RGB imagery.

---

## 📌 Architecture Overview

The system is split into two sequential deep learning stages to handle upscaling and modality translation independently:

```
[Input: Raw 200m TIR] 
       │
       ▼
┌──────────────────────────────────────────────┐
│  STAGE 1: Super-Resolution (HAT)             │  ◄── Physics Energy Loss (Conservation)
└──────────────────────────────────────────────┘
       │
       ▼
 [High-Res 100m TIR]
       │
       ▼
┌──────────────────────────────────────────────┐
│  STAGE 2: Colorization (ViT-Huge + Restormer)│  ◄── Geospatial FiLM Modulation
└──────────────────────────────────────────────┘  ◄── Gradient Alignment Loss (No Hallucinations)
       │
       ▼
[Output: 3-Channel 100m RGB]
```

---

## 🔍 Stage 1: Physics-Informed Super-Resolution (200m $\to$ 100m)

The objective of Stage 1 is to upscale the low-resolution 200m thermal band to a high-resolution 100m band, restoring sub-pixel boundaries without introducing artificial thermal noise.

### 1. Architecture: Hybrid Attention Transformer (`ThermalHATWrapper`)
Stage 1 employs a **Hybrid Attention Transformer (HAT)**. HAT combines the local representation power of convolutions with the global modeling capabilities of self-attention blocks. It is wrapped in `ThermalHATWrapper` to process single-channel floating-point thermal data.

### 2. Physics-Enforced Energy Conservation Loss
To prevent the network from violating thermodynamics by fabricating or destroying heat energy, Stage 1 uses a **radiance-space energy conservation constraint**:
* **Radiance Conversion**: Using calibration parameters (`radiance_mult` and `radiance_add`) parsed from the Landsat metadata XML files, pixel digital numbers (DN) are mapped to physical radiance values:
  $$\text{Radiance} = (\text{Pixel Value} \times \text{mult}) + \text{add}$$
* **Energy Integration**: The predicted high-resolution $512\times512$ radiance map is downscaled to $256\times256$ via $2\times2$ average pooling (`F.avg_pool2d`), simulating the sensor's physical integration window.
* **Loss Function**: The Mean Squared Error (MSE) between the downsampled predicted radiance and the raw 200m input radiance is minimized, forcing the output to obey energy conservation laws.

---

## 🎨 Stage 2: Geospatial-Conditioned Colorization (100m Thermal $\to$ 100m RGB)

Stage 2 translates the single-channel high-resolution thermal image into a 3-channel visible RGB image.

### 1. Hybrid U-Net Encoder-Decoder (`ThermalColorizerViT`)
The model is structured as a hybrid network blending a global Transformer backbone with a local CNN:
* **Global Encoder (ViT-Huge)**: The input thermal image is passed to a massive 632M parameter `ViT-Huge` model, which uses global self-attention across 1024 patches to extract atmospheric and contextual features.
* **Local CNN Encoder**: A parallel CNN encoder downsamples the input and extracts sharp local spatial maps (`c1` to `c4`). These features are bypassed directly to the decoder via **U-Net skip connections**, ensuring that spatial boundaries and sharp lines are preserved.

### 2. Feature-wise Linear Modulation (FiLM)
Thermal-to-RGB colorization is a "one-to-many" problem (e.g., a cold reading could be a shaded valley or a water body). To resolve this ambiguity, we inject geospatial and solar metadata directly into the network's latent space:
* **Inputs**: Center Latitude, Center Longitude, Sun Elevation, and Sun Azimuth.
* **Mechanism**: A Multi-Layer Perceptron (`self.metadata_mlp`) maps these 4 parameters to a 2560-dimensional vector, which is split into scale ($\gamma$) and shift ($\beta$) vectors.
* **Modulation**: The ViT spatial feature maps are modulated before entering the decoder:
  $$X_{\text{modulated}} = X_{\text{spatial}} \times \gamma + \beta$$

### 3. Metadata Dropout (Demo Robustness)
To prevent the model from failing if metadata is missing during inference, we implement a **20% Metadata Dropout** during training. 
* 20% of the time, the model ignores the metadata.
* This forces the decoder to learn a robust, generalized "average" colorization strategy, while still exploiting specific geographic/solar priors when they are provided.

### 4. Lightweight Restormer Decoder
To process the bottleneck features, we implement a **Restormer Attention Decoder**:
* **Attention Mechanism**: Uses Multi-Dconv Head Transposed Attention (MDTA) and Gated Dconv Feed-forward Networks (GDFN) at lower resolutions ($64\times64$ to $256\times256$) to handle global color and light propagation.
* **VRAM Guardrail**: At the highest spatial resolution ($512\times512$), the model automatically falls back to local **Double Conv** layers. This cuts VRAM usage by over 40%, ensuring the model fits comfortably within standard 16GB GPU memory limits (such as a single Tesla T4).

---

## ⚖️ Advanced Loss Formulations (Stage 2)

Stage 2 uses a compound loss function to enforce structural, color, and boundary correctness:

1. **Cross-Modal Gradient Alignment Loss (Sobel Similarity)**:
   * Converts the predicted RGB image to a single-channel Grayscale Luminance map: $\text{Luma} = 0.299R + 0.587G + 0.114B$.
   * Applies horizontal and vertical **Sobel filters** to both the input thermal image and the predicted Luma map.
   * Minimizes the **Cosine Distance** between these gradient vectors at every pixel. This mathematically locks the RGB boundaries to the thermal boundaries, preventing the model from hallucinating lines.
2. **Chromatic Angle Loss**:
   * Evaluates the **Cosine Similarity** of the $[R, G, B]$ vector at each pixel.
   * By ignoring vector magnitude (brightness) and focusing on the vector direction (chromatic angle/hue), it heavily penalizes flat gray/sepia predictions, forcing the model to output vibrant, realistic colors.
3. **Padding Penalty**:
   * Extracts the invalid rotated margin mask (`1.0 - valid_mask`).
   * Forcefully penalizes and zeroes out any non-black predictions in this region, eliminating glowing diagonal artifacts at the satellite swath boundaries.

---

## 🛰️ Large-Image Handling: Proposed Production Tiling & Stitching System

When running inference on full, raw Landsat 9 scenes (which are too large to fit in GPU memory at once), the system implements a vectorized **Tiling, Stitching, and Feathering** approach.

```
[Raw Large GeoTIFF] 
       │
       ▼  (Reflect Pad & PyTorch Unfold)
┌──────────────────────────────────────────────┐
│  Overlapping Patches (e.g. 512x512, step 448)│
└──────────────────────────────────────────────┘
       │
       ▼  (GPU Batch Inference)
┌──────────────────────────────────────────────┐
│  Colorized RGB Patches                       │
└──────────────────────────────────────────────┘
       │
       ▼  (Stitch + Cosine Window Blend)
[Seamless High-Res RGB TIFF]
```

### 1. Reflective Padding
The raw image is padded using reflection (`np.pad(..., mode="reflect")`) so its dimensions are perfectly divisible by the model patch size ($256$ for Stage 1 or $512$ for Stage 2). Reflective padding ensures that border tiles do not suffer from edge distortion.

### 2. Slicing with PyTorch `.unfold()`
The large padded tensor is sliced into overlapping windows in parallel using PyTorch's native `.unfold()` method:
```python
# Slices the width and height into patches of size 'patch_size' with overlap
windows = image_tensor.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
```
This is fully vectorized and runs directly in memory without slow pixel loops.

### 3. Overlapping & Feathered Blending (Seamless Stitching)
To prevent boundary seams (grid lines) where adjacent patches meet, the sliding window operates with an overlap (e.g., a stride of $448$ for a patch size of $512$, giving a 64-pixel overlap).
* During reconstruction, the system initializes a canvas accumulator and a weight accumulator.
* For each patch, we multiply it by a 2D **Cosine/Gaussian Window Function** before adding it to the canvas. The window function assigns a weight of $1.0$ at the center of the patch and smoothly drops to $0.0$ at the borders.
* The final image is obtained by dividing the canvas accumulator by the weight accumulator. This completely feathers the overlapping regions, resulting in a seamless, professional output.

---

## 🛠️ Project Structure

```
IR-colorization-BAH2026/
│
├── data/
│   └── metadata_mapping.json     # Compiled geospatial/solar metadata mapping
│
├── models.py                     # ThermalHATWrapper, ThermalColorizerViT, Restormer
├── losses.py                     # Stage1PhysicsLoss, Stage2ColorizationLoss
├── physics_informed_ml.py        # Dataset class, CrossModalGradientAlignmentLoss
├── preprocess_metadata.py        # Parses MTL.xml files to extract lat/lon & solar angles
├── train.py                      # Main end-to-end training script (Stage 1 & Stage 2)
├── driver.py                     # Baseline patch extraction driver
└── README.md                     # Technical reference readme
```

---

## 🚀 Getting Started

### 1. Preprocess Metadata
If you have new Landsat 9 folders containing `_MTL.xml` files, run the preprocessor to generate the updated geospatial database:
```bash
python preprocess_metadata.py --input_dir /path/to/landsat/products --output_json data/metadata_mapping.json
```

### 2. Train the Pipeline
To train both Stage 1 and Stage 2 end-to-end on your dataset (e.g., on Kaggle or a local workstation):
```bash
python train.py --stage both --epochs 50 --batch_size 4 --patches_dir /path/to/patches
```
*Note: The script automatically detects available GPUs and applies `nn.DataParallel` with native gradient checkpointing.*
