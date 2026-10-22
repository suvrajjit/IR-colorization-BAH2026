# IRGB: Infrared Image Colorization and Enhancement for Improved Object Interpretation in Satellite Imagery

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation and ablation suite for the research paper:  
**"IRGB: Infrared Image Colorization and Enhancement for Improved Object Interpretation in Satellite Imagery"**  
*Suvrajit Dey, Swamantak Roy, Sumit Kumar Tetarave, Satya Ranjan Dash, and Rishikesh Sahay*

---

## 📌 Abstract

Thermal infrared (TIR) satellite imagery is indispensable for monitoring thermal anomalies, wildfire tracking, urban heat-island analysis, and volcanic monitoring. However, its single-band grayscale representation and coarse spatial resolution limit object-level interpretability. 

**IRGB** is a two-stage, physics-informed, metadata-conditioned deep learning framework that enhances Landsat 9 Band 10 thermal data into high-resolution, photorealistic, and structurally faithful RGB visualizations:
1. **Stage 1 (Thermal Super-Resolution, 200m $\to$ 100m):** Employs a single-channel Hybrid Attention Transformer (HAT) constrained by a top-of-atmosphere radiance energy conservation loss ($\mathcal{L}_{phy}$) to prevent unphysical thermal artifacts.
2. **Stage 2 (Context-Conditioned Colorization, 100m TIR $\to$ 100m RGB):** Combines a Vision Transformer (ViT-Huge) global context encoder, a multi-scale CNN skip encoder, and a Restormer decoder, conditioned on solar geometry and geolocation metadata via Feature-wise Linear Modulation (FiLM).

```
               STAGE 1: Physics-Informed Super-Resolution
[Raw 200m TIR] ──► [Hybrid Attention Transformer (HAT)] ──► [Enhanced 100m TIR]
                           │
                           └── Constrained by Radiance Energy Loss (L_phy)

               STAGE 2: Geospatial-Conditioned Colorization
[Enhanced 100m TIR] ──► [ViT-Huge Global + Multi-Scale CNN Skips] ──► [Restormer Decoder] ──► [100m 3-Channel RGB]
                                      ▲                                      ▲
                                      │                                      │
                         [Solar & Coordinate Metadata] ──────────► [FiLM Modulation]
```

---

## 📊 Ablation Study (Paper Table 1)

Quantitative evaluation on held-out test scenes across India (scene-level disjoint split, seed 42):

| Stage | Variant | PSNR (dB) | SSIM | Rad. RMSE | Chrom. err. | Key Finding |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Full ($\mathcal{L}_{pixel} + \mathcal{L}_{phy} + \mathcal{L}_{ssim}$)** | **23.25 $\pm$ 10.79** | **0.800 $\pm$ 0.257** | **0.3446** | n/a | **Lowest physical energy error** |
| 1 | w/o $\mathcal{L}_{phy}$ | 23.23 $\pm$ 11.43 | 0.789 $\pm$ 0.271 | 0.3979 | n/a | Radiance RMSE increases by $+15.5\%$ |
| 1 | w/o $\mathcal{L}_{ssim}$ | 23.12 $\pm$ 10.53 | 0.795 $\pm$ 0.266 | 0.3283 | n/a | Structural similarity drops |
| 1 | Plain HAT replaced by CNN SR | 25.94 $\pm$ 9.34 | 0.852 $\pm$ 0.182 | 0.3530 | n/a | Pixel baseline without transformer attention |
| \hline | | | | | | |
| **2** | **Full IRGB** | **15.44 $\pm$ 3.65** | **0.623 $\pm$ 0.175** | n/a | **9.53$^\circ$ $\pm$ 5.42$^\circ$** | **Best color & structural balance** |
| 2 | w/o metadata (no FiLM) | 14.21 $\pm$ 3.44 | 0.588 $\pm$ 0.168 | n/a | 10.77$^\circ$ $\pm$ 6.07$^\circ$ | Color drift without solar context |
| 2 | Metadata concatenated to input | 10.85 $\pm$ 3.66 | 0.353 $\pm$ 0.207 | n/a | 22.21$^\circ$ $\pm$ 14.04$^\circ$ | Input channels corrupt spatial convolutions |
| 2 | w/o $\mathcal{L}_{grad}$ | 15.07 $\pm$ 2.75 | 0.625 $\pm$ 0.159 | n/a | 9.02$^\circ$ $\pm$ 3.48$^\circ$ | Boundary correlation degrades |
| 2 | w/o $\mathcal{L}_{chrom}$ | 14.38 $\pm$ 3.13 | 0.602 $\pm$ 0.187 | n/a | 13.93$^\circ$ $\pm$ 10.82$^\circ$ | $+46\%$ surge in angular chromatic error |
| 2 | w/o I-JEPA global branch | 15.52 $\pm$ 2.59 | 0.618 $\pm$ 0.158 | n/a | 9.08$^\circ$ $\pm$ 4.40$^\circ$ | Loss of global scene context |
| 2 | w/o CNN skip branch | 12.84 $\pm$ 2.82 | 0.377 $\pm$ 0.235 | n/a | 18.04$^\circ$ $\pm$ 11.09$^\circ$ | **Catastrophic structural collapse** |

---

## 🛠️ Repository Organization

The repository separates the core training and evaluation suite from data curation tools:

```
IRGB/
├── models.py               # Core architectures: ThermalHATWrapper, ThermalColorizerViT, Restormer
├── losses.py               # Physics energy loss, Chromaticity angular loss, Sobel alignment
├── physics_informed_ml.py  # IRGBDataset loader, Planck radiance conversion, split manager
├── train.py                # End-to-end training pipeline for Stage 1 & Stage 2
├── run_ablation.py         # Master runner for reproducing Table 1 ablations
├── run_stage1.py           # 1-command runner for Stage 1 experiments
├── run_stage2.py           # 1-command runner for Stage 2 experiments
├── show_results.py         # Formats ablation_results.json into Markdown and LaTeX tables
├── paper_insertions.tex    # Pre-compiled LaTeX table and interpretation paragraphs
├── PAPER_INSERTIONS.md     # Markdown reference for paper writing
│
├── data/
│   └── metadata_mapping.json # Radiometric calibration (mult/add), solar angles, lat/lon
│
├── dataset_prep/           # Raw satellite data preprocessing & patch tiling pipeline
│   ├── README.md           # Dataset preparation documentation
│   ├── driver.py           # Automated patch extraction driver
│   ├── preprocess_metadata.py # Extracts metadata from Landsat 9 MTL.xml
│   ├── create_output_zip.py   # Dataset archiving utility
│   └── scripts/            # Band composite, GDAL downscaling, and tiling scripts
│
├── utils/                  # Shared file utilities, logging, and visualizers
└── tests/                  # Unit tests (loss verification)
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/suvrajjit/IR-colorization-BAH2026.git
cd IR-colorization-BAH2026
pip install -r requirements.txt
```

### 2. Reproduce Table 1 Ablation Study

To reproduce the experiments on Kaggle or a local GPU workstation:

**Run Stage 1 Super-Resolution Suite (Rows 1–4):**
```bash
python run_stage1.py --epochs 25 --batch_size 4
```

**Run Stage 2 Colorization Suite (Rows 5–11):**
```bash
python run_stage2.py --epochs 5 --batch_size 4
```
*(Or evaluate individual variants: `--variant full`, `wo_film`, `concat_meta`, `wo_grad`, `wo_chrom`, `wo_vit`, `wo_skips`)*

**Display Table 1 & LaTeX Code:**
```bash
python show_results.py
```

### 3. Full End-to-End Model Training

To train the complete pipeline end-to-end:
```bash
# Stage 1: Super-Resolution (HAT)
python train.py --stage super_res --epochs 50 --batch_size 4 --patches_dir /path/to/patches

# Stage 2: Context-Conditioned Colorization (ViT + FiLM + Restormer)
python train.py --stage colorization --epochs 50 --batch_size 4 --patches_dir /path/to/patches
```

---

## 🛰️ Dataset Preparation

The dataset consists of 800 co-registered patch triplets (256×256 low-res TIR, 512×512 high-res TIR, 512×512 reference RGB) derived from Landsat 9 Collection 2 Level-2 products over the Indian subcontinent.

To prepare the dataset from raw Landsat 9 GeoTIFF files:
```bash
cd dataset_prep
python preprocess_metadata.py --input_dir /path/to/raw_landsat --output_json ../data/metadata_mapping.json
python driver.py
```
See [`dataset_prep/README.md`](dataset_prep/README.md) for detailed instructions.

---

## 📖 Citation

If you use this codebase or paper in your research, please cite:

```bibtex
@article{dey2026irgb,
  title={IRGB: Infrared Image Colorization and Enhancement for Improved Object Interpretation in Satellite Imagery},
  author={Dey, Suvrajit and Roy, Swamantak and Tetarave, Sumit Kumar and Dash, Satya Ranjan and Sahay, Rishikesh},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
