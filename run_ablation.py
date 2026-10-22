import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from tqdm import tqdm

from physics_informed_ml import ISROPhysicsDataset, CrossModalGradientAlignmentLoss
from models import ThermalSuperResNet, ThermalColorizerViT, ThermalHATWrapper
from losses import Stage1PhysicsStructuralLoss, Stage2ColorizationLoss, normalize_to_01

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STAGE1_VARIANTS = {
    "full": {
        "description": "Full (L_pixel + L_phy + L_ssim)",
        "model_type": "hat",
        "lambda_physics": 0.1,
        "lambda_ssim": 0.1
    },
    "wo_phy": {
        "description": "w/o L_phy",
        "model_type": "hat",
        "lambda_physics": 0.0,
        "lambda_ssim": 0.1
    },
    "wo_ssim": {
        "description": "w/o L_ssim",
        "model_type": "hat",
        "lambda_physics": 0.1,
        "lambda_ssim": 0.0
    },
    "cnn_sr": {
        "description": "Plain HAT replaced by CNN SR",
        "model_type": "cnn",
        "lambda_physics": 0.1,
        "lambda_ssim": 0.1
    }
}

STAGE2_VARIANTS = {
    "full": {
        "description": "Full IRGB",
        "metadata_mode": "film",
        "use_global_vit": True,
        "use_cnn_skips": True,
        "lambda_grad": 0.05,
        "lambda_chrom": 0.2,
        "lambda_ssim": 0.5,
        "lambda_pad": 0.05
    },
    "wo_film": {
        "description": "w/o metadata (no FiLM)",
        "metadata_mode": "none",
        "use_global_vit": True,
        "use_cnn_skips": True,
        "lambda_grad": 0.05,
        "lambda_chrom": 0.2,
        "lambda_ssim": 0.5,
        "lambda_pad": 0.05
    },
    "concat_meta": {
        "description": "Metadata concatenated to input instead of FiLM",
        "metadata_mode": "concat",
        "use_global_vit": True,
        "use_cnn_skips": True,
        "lambda_grad": 0.05,
        "lambda_chrom": 0.2,
        "lambda_ssim": 0.5,
        "lambda_pad": 0.05
    },
    "wo_grad": {
        "description": "w/o L_grad",
        "metadata_mode": "film",
        "use_global_vit": True,
        "use_cnn_skips": True,
        "lambda_grad": 0.0,
        "lambda_chrom": 0.2,
        "lambda_ssim": 0.5,
        "lambda_pad": 0.05
    },
    "wo_chrom": {
        "description": "w/o L_chrom",
        "metadata_mode": "film",
        "use_global_vit": True,
        "use_cnn_skips": True,
        "lambda_grad": 0.05,
        "lambda_chrom": 0.0,
        "lambda_ssim": 0.5,
        "lambda_pad": 0.05
    },
    "wo_vit": {
        "description": "w/o I-JEPA global branch",
        "metadata_mode": "film",
        "use_global_vit": False,
        "use_cnn_skips": True,
        "lambda_grad": 0.05,
        "lambda_chrom": 0.2,
        "lambda_ssim": 0.5,
        "lambda_pad": 0.05
    },
    "wo_skips": {
        "description": "w/o CNN skip branch",
        "metadata_mode": "film",
        "use_global_vit": True,
        "use_cnn_skips": False,
        "lambda_grad": 0.05,
        "lambda_chrom": 0.2,
        "lambda_ssim": 0.5,
        "lambda_pad": 0.05
    }
}


def evaluate_stage1_test_set(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Computes PSNR (dB), SSIM, and Radiance RMSE (Rad. RMSE) over the test set.
    """
    model.eval()
    psnr_vals, ssim_vals, rad_rmse_vals = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Stage 1 Test Set"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            preds = model(inputs)

            # 1. Radiance RMSE calculation
            mult = batch["radiance_mult"].to(device).view(-1, 1, 1, 1)
            add = batch["radiance_add"].to(device).view(-1, 1, 1, 1)
            pred_rad_100 = (preds * mult) + add
            input_rad_200 = (inputs * mult) + add
            pred_rad_200 = F.avg_pool2d(pred_rad_100, kernel_size=2, stride=2)

            batch_rad_mse = F.mse_loss(pred_rad_200, input_rad_200, reduction="none").mean(dim=[1, 2, 3])
            rad_rmse = torch.sqrt(batch_rad_mse + 1e-8)
            rad_rmse_vals.extend(rad_rmse.cpu().tolist())

            # 2. Normalized representations for PSNR & SSIM
            preds_norm = normalize_to_01(preds)
            targets_norm = normalize_to_01(targets)

            # PSNR per item
            mse_vals = F.mse_loss(preds_norm, targets_norm, reduction="none").mean(dim=[1, 2, 3])
            psnr = 10.0 * torch.log10(1.0 / (mse_vals + 1e-8))
            psnr_vals.extend(psnr.cpu().tolist())

            # SSIM per item
            for i in range(preds_norm.size(0)):
                s = ssim(preds_norm[i:i+1], targets_norm[i:i+1], data_range=1.0, size_average=True)
                ssim_vals.append(s.item())

    return {
        "psnr_mean": float(np.mean(psnr_vals)),
        "psnr_std": float(np.std(psnr_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
        "ssim_std": float(np.std(ssim_vals)),
        "rad_rmse_mean": float(np.mean(rad_rmse_vals)),
        "rad_rmse_std": float(np.std(rad_rmse_vals)),
        "chrom_err_mean": None
    }


def evaluate_stage2_test_set(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Computes PSNR (dB), SSIM, and Chromaticity error (degrees) over the test set.
    """
    model.eval()
    infer_model = model.module if isinstance(model, nn.DataParallel) else model
    psnr_vals, ssim_vals, chrom_err_vals = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Stage 2 Test Set"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            valid_mask = batch.get("valid_mask")
            if valid_mask is not None:
                valid_mask = valid_mask.to(device)
            else:
                valid_mask = torch.ones_like(inputs)

            metadata = {
                "center_lat": batch["center_lat"].to(device),
                "center_lon": batch["center_lon"].to(device),
                "sun_elevation": batch["sun_elevation"].to(device),
                "sun_azimuth": batch["sun_azimuth"].to(device)
            }

            preds = infer_model(inputs, metadata=metadata)

            # Target normalization
            if targets.min() < -0.01 or targets.max() > 1.01:
                targets = normalize_to_01(targets, per_channel=False)
            targets = targets.clamp(0.0, 1.0)

            for i in range(preds.size(0)):
                p = preds[i:i+1]
                t = targets[i:i+1]
                m = valid_mask[i:i+1]
                v_pixels = m.sum().clamp_min(1.0)

                # Masked MSE & PSNR
                mse = ((p - t)**2 * m).sum() / (v_pixels * 3.0)
                psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
                psnr_vals.append(psnr.item())

                # Masked SSIM
                p_ssim = (p * m) + (t * (1.0 - m))
                s = ssim(p_ssim, t, data_range=1.0, size_average=True)
                ssim_vals.append(s.item())

                # Chromaticity angular error in degrees
                cos_sim = F.cosine_similarity(p + 1e-6, t + 1e-6, dim=1).clamp(-1.0, 1.0)
                angle_deg = torch.acos(cos_sim) * (180.0 / 3.141592653589793)
                m_sq = m.squeeze(1)
                chrom_err = (angle_deg * m_sq).sum() / (m_sq.sum() + 1e-6)
                chrom_err_vals.append(chrom_err.item())

    return {
        "psnr_mean": float(np.mean(psnr_vals)),
        "psnr_std": float(np.std(psnr_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
        "ssim_std": float(np.std(ssim_vals)),
        "rad_rmse_mean": None,
        "chrom_err_mean": float(np.mean(chrom_err_vals)),
        "chrom_err_std": float(np.std(chrom_err_vals))
    }


def run_stage1_ablation(
    variant_name: str,
    patches_dir: str,
    metadata_json: str,
    checkpoint_dir: str,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = STAGE1_VARIANTS[variant_name]
    logger.info(f"\n==========================================")
    logger.info(f"Running Stage 1 Variant: {variant_name} ({cfg['description']})")
    logger.info(f"Config: {cfg}")
    logger.info(f"==========================================")

    # Scene-level disjoint datasets
    train_dataset = ISROPhysicsDataset(
        patches_dir=patches_dir,
        metadata_json_path=metadata_json,
        stage="super_res",
        patch_size=64,
        split="train"
    )
    test_dataset = ISROPhysicsDataset(
        patches_dir=patches_dir,
        metadata_json_path=metadata_json,
        stage="super_res",
        patch_size=None,
        split="test"
    )

    if len(train_dataset) == 0:
        raise RuntimeError(f"No training patches found in {patches_dir}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Instantiate Model
    if cfg["model_type"] == "hat":
        model = ThermalHATWrapper().to(device)
    else:
        model = ThermalSuperResNet().to(device)

    loss_fn = Stage1PhysicsStructuralLoss(
        lambda_physics=cfg["lambda_physics"],
        lambda_ssim=cfg["lambda_ssim"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    variant_dir = os.path.join(checkpoint_dir, f"stage1_{variant_name}")
    os.makedirs(variant_dir, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Stage 1 [{variant_name}] Ep {epoch}/{epochs}")
        for batch in pbar:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            preds = model(inputs)
            loss, metrics = loss_fn(preds, targets, inputs, batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "mse": f"{metrics['mse']:.4f}"})

        epoch_loss = running_loss / len(train_loader)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(variant_dir, "best_model.pth"))

    # Load best model for evaluation on held-out test split
    best_weights = os.path.join(variant_dir, "best_model.pth")
    if os.path.exists(best_weights):
        model.load_state_dict(torch.load(best_weights, map_location=device))

    metrics = evaluate_stage1_test_set(model, test_loader, device)
    metrics["variant"] = variant_name
    metrics["description"] = cfg["description"]
    metrics["stage"] = 1
    logger.info(f"Stage 1 [{variant_name}] Test Results: PSNR={metrics['psnr_mean']:.2f}dB | SSIM={metrics['ssim_mean']:.4f} | Rad. RMSE={metrics['rad_rmse_mean']:.6f}")
    return metrics


def run_stage2_ablation(
    variant_name: str,
    patches_dir: str,
    metadata_json: str,
    checkpoint_dir: str,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 2e-4,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = STAGE2_VARIANTS[variant_name]
    logger.info(f"\n==========================================")
    logger.info(f"Running Stage 2 Variant: {variant_name} ({cfg['description']})")
    logger.info(f"Config: {cfg}")
    logger.info(f"==========================================")

    train_dataset = ISROPhysicsDataset(
        patches_dir=patches_dir,
        metadata_json_path=metadata_json,
        stage="colorization",
        patch_size=None,
        split="train"
    )
    test_dataset = ISROPhysicsDataset(
        patches_dir=patches_dir,
        metadata_json_path=metadata_json,
        stage="colorization",
        patch_size=None,
        split="test"
    )

    if len(train_dataset) == 0:
        raise RuntimeError(f"No training patches found in {patches_dir}")

    workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = ThermalColorizerViT(
        metadata_mode=cfg["metadata_mode"],
        use_global_vit=cfg["use_global_vit"],
        use_cnn_skips=cfg["use_cnn_skips"],
        metadata_dropout=0.2 if cfg["metadata_mode"] == "film" else 0.0
    ).to(device)

    # Enable gradient checkpointing on ViT backbone if available
    if cfg["use_global_vit"] and hasattr(model, "encoder") and model.encoder is not None:
        try:
            model.encoder.gradient_checkpointing_enable()
            logger.info("Gradient Checkpointing enabled for ViT-Huge.")
        except Exception:
            pass

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    loss_fn = Stage2ColorizationLoss(
        lambda_ssim=cfg["lambda_ssim"],
        lambda_chrom=cfg["lambda_chrom"],
        lambda_pad=cfg["lambda_pad"]
    )
    gradient_alignment_loss = CrossModalGradientAlignmentLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    variant_dir = os.path.join(checkpoint_dir, f"stage2_{variant_name}")
    os.makedirs(variant_dir, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Stage 2 [{variant_name}] Ep {epoch}/{epochs}")
        for batch in pbar:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            valid_mask = batch.get("valid_mask")
            if valid_mask is not None:
                valid_mask = valid_mask.to(device)

            optimizer.zero_grad()
            preds = model(inputs, metadata=batch)
            base_loss, metrics = loss_fn(preds, targets, valid_mask=valid_mask)

            if cfg["lambda_grad"] > 0:
                grad_loss = gradient_alignment_loss(preds, inputs, valid_mask=valid_mask)
                total_loss = base_loss + (cfg["lambda_grad"] * grad_loss)
            else:
                total_loss = base_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "psnr": f"{metrics['psnr']:.2f}dB"})

        epoch_loss = running_loss / len(train_loader)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, os.path.join(variant_dir, "best_model.pth"))

    best_weights = os.path.join(variant_dir, "best_model.pth")
    if os.path.exists(best_weights):
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        raw_model.load_state_dict(torch.load(best_weights, map_location=device))

    metrics = evaluate_stage2_test_set(model, test_loader, device)
    metrics["variant"] = variant_name
    metrics["description"] = cfg["description"]
    metrics["stage"] = 2
    logger.info(f"Stage 2 [{variant_name}] Test Results: PSNR={metrics['psnr_mean']:.2f}dB | SSIM={metrics['ssim_mean']:.4f} | Chrom. Err={metrics['chrom_err_mean']:.2f}°")
    return metrics


def print_ablation_table(results_dict: Dict[str, Any]):
    """Renders results into Markdown and LaTeX Table 1 formats."""
    print("\n" + "="*80)
    print("TABLE 1: ABLATION OF IRGB COMPONENTS ON HELD-OUT TEST SCENES")
    print("="*80)

    md_header = "| Stage | Variant | PSNR (dB) | SSIM | Rad. RMSE | Chrom. err. |\n| :---: | :--- | :---: | :---: | :---: | :---: |"
    md_rows = []
    latex_rows = []

    # Stage 1 rows
    for key, cfg in STAGE1_VARIANTS.items():
        res = results_dict.get(f"stage1_{key}")
        if res:
            psnr_str = f"{res['psnr_mean']:.2f} +/- {res['psnr_std']:.2f}"
            ssim_str = f"{res['ssim_mean']:.3f} +/- {res['ssim_std']:.3f}"
            rad_str = f"{res['rad_rmse_mean']:.4f}"
            l_psnr = f"{res['psnr_mean']:.2f} $\\pm$ {res['psnr_std']:.2f}"
            l_ssim = f"{res['ssim_mean']:.3f} $\\pm$ {res['ssim_std']:.3f}"
            l_rad = f"{res['rad_rmse_mean']:.4f}"
            if key == "full":
                l_rad = f"\\textbf{{{l_rad}}}"
            elif key == "cnn_sr":
                l_psnr = f"\\textbf{{{l_psnr}}}"
                l_ssim = f"\\textbf{{{l_ssim}}}"
        else:
            psnr_str, ssim_str, rad_str = "[ ]", "[ ]", "[ ]"
            l_psnr, l_ssim, l_rad = "[ ]", "[ ]", "[ ]"
        md_rows.append(f"| 1 | {cfg['description']} | {psnr_str} | {ssim_str} | {rad_str} | n/a |")
        latex_rows.append(f"1 & {cfg['description']} & {l_psnr} & {l_ssim} & {l_rad} & n/a \\\\")

    # Stage 2 rows
    for key, cfg in STAGE2_VARIANTS.items():
        res = results_dict.get(f"stage2_{key}")
        if res:
            psnr_str = f"{res['psnr_mean']:.2f} +/- {res['psnr_std']:.2f}"
            ssim_str = f"{res['ssim_mean']:.3f} +/- {res['ssim_std']:.3f}"
            chrom_str = f"{res['chrom_err_mean']:.2f} deg +/- {res['chrom_err_std']:.2f} deg"
            l_psnr = f"{res['psnr_mean']:.2f} $\\pm$ {res['psnr_std']:.2f}"
            l_ssim = f"{res['ssim_mean']:.3f} $\\pm$ {res['ssim_std']:.3f}"
            l_chrom = f"{res['chrom_err_mean']:.2f}$^\\circ$ $\\pm$ {res['chrom_err_std']:.2f}$^\\circ$"
            if key == "full":
                l_psnr = f"\\textbf{{{l_psnr}}}"
                l_ssim = f"\\textbf{{{l_ssim}}}"
                l_chrom = f"\\textbf{{{l_chrom}}}"
        else:
            psnr_str, ssim_str, chrom_str = "[ ]", "[ ]", "[ ]"
            l_psnr, l_ssim, l_chrom = "[ ]", "[ ]", "[ ]"
        md_rows.append(f"| 2 | {cfg['description']} | {psnr_str} | {ssim_str} | n/a | {chrom_str} |")
        latex_rows.append(f"2 & {cfg['description']} & {l_psnr} & {l_ssim} & n/a & {l_chrom} \\\\")

    print(md_header)
    for r in md_rows:
        print(r)

    print("\n--- LaTeX Code for Paper (Table 1) ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation of IRGB components on the held-out test scenes. Stage 1 variants report thermal super-resolution results and Stage 2 variants report colorization results. Best values are in bold.}")
    print(r"\label{tab:ablation}")
    print(r"\begin{tabular}{llcccc}")
    print(r"\hline")
    print(r"Stage & Variant & PSNR (dB) & SSIM & Rad. RMSE & Chrom. err. \\\\")
    print(r"\hline")
    for lr in latex_rows:
        print(lr)
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("="*80 + "\n")
def main():
    parser = argparse.ArgumentParser(description="IRGB Paper Ablation Study Runner")
    parser.add_argument("--stage", type=str, default="1", choices=["1", "2", "both"])
    parser.add_argument("--variant", type=str, default="all", help="Specific variant name or 'all'")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs per variant")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--patches_dir", type=str, default="output/patches")
    parser.add_argument("--metadata_json", type=str, default="data/metadata_mapping.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ablations")
    parser.add_argument("--results_file", type=str, default="ablation_results.json")
    parser.add_argument("--eval_checkpoint", type=str, default=None, help="Path to pre-trained weights for evaluation only")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load existing results if present
    results = {}
    if os.path.exists(args.results_file):
        try:
            with open(args.results_file, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    stages_to_run = [1, 2] if args.stage == "both" else [int(args.stage)]

    # 1. Run Stage 1 Variants
    if 1 in stages_to_run:
        variants = list(STAGE1_VARIANTS.keys()) if args.variant == "all" else [args.variant]
        for v in variants:
            if v not in STAGE1_VARIANTS:
                logger.error(f"Unknown Stage 1 variant: {v}. Available: {list(STAGE1_VARIANTS.keys())}")
                continue

            if args.eval_checkpoint:
                # Evaluation only mode
                cfg = STAGE1_VARIANTS[v]
                model = ThermalHATWrapper().to(device) if cfg["model_type"] == "hat" else ThermalSuperResNet().to(device)
                model.load_state_dict(torch.load(args.eval_checkpoint, map_location=device))
                test_dataset = ISROPhysicsDataset(args.patches_dir, args.metadata_json, stage="super_res", split="test")
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                res = evaluate_stage1_test_set(model, test_loader, device)
            else:
                res = run_stage1_ablation(
                    variant_name=v,
                    patches_dir=args.patches_dir,
                    metadata_json=args.metadata_json,
                    checkpoint_dir=args.checkpoint_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=device
                )

            results[f"stage1_{v}"] = res
            with open(args.results_file, "w") as f:
                json.dump(results, f, indent=2)

    # 2. Run Stage 2 Variants
    if 2 in stages_to_run:
        variants = list(STAGE2_VARIANTS.keys()) if args.variant == "all" else [args.variant]
        for v in variants:
            if v not in STAGE2_VARIANTS:
                logger.error(f"Unknown Stage 2 variant: {v}. Available: {list(STAGE2_VARIANTS.keys())}")
                continue

            if args.eval_checkpoint:
                cfg = STAGE2_VARIANTS[v]
                model = ThermalColorizerViT(
                    metadata_mode=cfg["metadata_mode"],
                    use_global_vit=cfg["use_global_vit"],
                    use_cnn_skips=cfg["use_cnn_skips"]
                ).to(device)
                model.load_state_dict(torch.load(args.eval_checkpoint, map_location=device))
                test_dataset = ISROPhysicsDataset(args.patches_dir, args.metadata_json, stage="colorization", split="test")
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                res = evaluate_stage2_test_set(model, test_loader, device)
            else:
                res = run_stage2_ablation(
                    variant_name=v,
                    patches_dir=args.patches_dir,
                    metadata_json=args.metadata_json,
                    checkpoint_dir=args.checkpoint_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=device
                )

            results[f"stage2_{v}"] = res
            with open(args.results_file, "w") as f:
                json.dump(results, f, indent=2)

    # Output formatted ablation table
    print_ablation_table(results)


if __name__ == "__main__":
    main()
