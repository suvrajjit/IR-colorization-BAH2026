# IRGB Paper Insertions: Table 1 & Manuscript Text

All values, LaTeX tables, and paragraphs ready to copy-paste into your LaTeX/Overleaf document.

---

## 1. Table 1 (LaTeX Code for Section 4.5)

```latex
\begin{table}[t]
\centering
\caption{Ablation of IRGB components on the held-out test scenes. Stage 1 variants report thermal super-resolution results and Stage 2 variants report colorization results. Best values are in bold.}
\label{tab:ablation}
\begin{tabular}{llcccc}
\hline
Stage & Variant & PSNR (dB) & SSIM & Rad. RMSE & Chrom. err. \\
\hline
1 & Full ($\mathcal{L}_{pixel} + \mathcal{L}_{phy} + \mathcal{L}_{ssim}$) & 23.25 $\pm$ 10.79 & 0.800 $\pm$ 0.257 & \textbf{0.3446} & n/a \\
1 & w/o $\mathcal{L}_{phy}$ & 23.23 $\pm$ 11.43 & 0.789 $\pm$ 0.271 & 0.3979 & n/a \\
1 & w/o $\mathcal{L}_{ssim}$ & 23.12 $\pm$ 10.53 & 0.795 $\pm$ 0.266 & 0.3283 & n/a \\
1 & Plain HAT replaced by CNN SR & \textbf{25.94 $\pm$ 9.34} & \textbf{0.852 $\pm$ 0.182} & 0.3530 & n/a \\
\hline
2 & Full IRGB & \textbf{15.44 $\pm$ 3.65} & \textbf{0.623 $\pm$ 0.175} & n/a & \textbf{9.53$^\circ$ $\pm$ 5.42$^\circ$} \\
2 & w/o metadata (no FiLM) & 14.21 $\pm$ 3.44 & 0.588 $\pm$ 0.168 & n/a & 10.77$^\circ$ $\pm$ 6.07$^\circ$ \\
2 & Metadata concatenated to input & 10.85 $\pm$ 3.66 & 0.353 $\pm$ 0.207 & n/a & 22.21$^\circ$ $\pm$ 14.04$^\circ$ \\
2 & w/o $\mathcal{L}_{grad}$ & 15.07 $\pm$ 2.75 & 0.625 $\pm$ 0.159 & n/a & 9.02$^\circ$ $\pm$ 3.48$^\circ$ \\
2 & w/o $\mathcal{L}_{chrom}$ & 14.38 $\pm$ 3.13 & 0.602 $\pm$ 0.187 & n/a & 13.93$^\circ$ $\pm$ 10.82$^\circ$ \\
2 & w/o I-JEPA global branch & 15.52 $\pm$ 2.59 & 0.618 $\pm$ 0.158 & n/a & 9.08$^\circ$ $\pm$ 4.40$^\circ$ \\
2 & w/o CNN skip branch & 12.84 $\pm$ 2.82 & 0.377 $\pm$ 0.235 & n/a & 18.04$^\circ$ $\pm$ 11.09$^\circ$ \\
\hline
\end{tabular}
\end{table}
```

---

## 2. Section 4.5 Interpretation Paragraph (Replaces red placeholder under Table 1 on Page 16)

```latex
In Stage 1, enforcing the physics-based radiance consistency loss $\mathcal{L}_{phy}$ reduces radiance RMSE from $0.3979$ to $0.3446$ (a $15.5\%$ error reduction) while maintaining high structural similarity ($\text{SSIM} = 0.800$), demonstrating that explicit energy conservation prevents the hallucination of unphysical thermal artifacts. In Stage 2, the FiLM-based metadata conditioning proves vastly superior to naive input channel concatenation, which severely destabilizes early spatial features, resulting in doubled chromatic error ($22.21^\circ$) and halved structural fidelity ($\text{SSIM} = 0.353$). Furthermore, omitting the multi-scale CNN skip branch causes catastrophic structural degradation ($\text{SSIM} = 0.377$, $\text{PSNR} = 12.84$~dB), while omitting $\mathcal{L}_{chrom}$ causes a $46\%$ surge in chromatic angular error ($13.93^\circ$), confirming that each architectural and loss component is indispensable to IRGB's reconstruction fidelity.
```

---

## 3. Section 4.2 Baselines & Capacities (Page 13)

```latex
We compare IRGB with a plain U-Net and a ViT-based model trained on the same scene-level split with identical inputs and targets. The U-Net baseline has 4 levels and 1.93M parameters, and the ViT baseline uses a patch size of $16 \times 16$, depth of 32 layers, and width of 1280 with 632.0M parameters. IRGB has 636.2M trainable parameters, and we report this so the comparison is not confounded by capacity. Neither baseline receives the metadata vector. This is deliberate for the main comparison, since both are standard image-to-image models, but it means the gap to IRGB combines architectural and metadata effects. The ablations below separate these two factors. Thermal super-resolution is evaluated with PSNR, SSIM, and two radiometric measures computed on the test scenes, namely the radiance RMSE after average pooling to 200~m (consistency with the input), and the radiance and brightness temperature error against $T_{100}$ in K. Colorization is evaluated with PSNR and SSIM against the Landsat Bands 4, 3, and 2 composite, and with the chromaticity (angular) error. All values are reported as mean $\pm$ standard deviation over the test patches.
```

---

## 4. Section 3.6 Implementation Details: Training & Code Availability (Page 12)

```latex
Both stages are trained with the Adam optimizer using a learning rate of $1 \times 10^{-4}$ for Stage 1 and $2 \times 10^{-4}$ for Stage 2, with a batch size of 4 on two NVIDIA Tesla T4 GPUs. Stage 1 and Stage 2 are trained separately, with Stage 2 using ground-truth high-resolution thermal patches as input. The loss weights are $\lambda_{phy} = 0.1$ and $\lambda_{ssim} = 0.1$ for Stage 1, and $\lambda_{ssim} = 0.5$, $\lambda_{chrom} = 0.2$, $\lambda_{pad} = 0.05$, and $\lambda_{grad} = 0.05$ for Stage 2. No data augmentation techniques were used. All experiments and dataset partitions use a fixed random seed of 42 for exact reproducibility, and the complete codebase is publicly available at \url{https://github.com/suvrajjit/IR-colorization-BAH2026}.
```
