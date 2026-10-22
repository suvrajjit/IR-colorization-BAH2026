import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Optional
from tqdm import tqdm

from physics_informed_ml import ISROPhysicsDataset, CrossModalGradientAlignmentLoss
from models import ThermalSuperResNet, ThermalColorizerViT, ThermalHATWrapper
from losses import Stage1PhysicsStructuralLoss, Stage2ColorizationLoss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_comparison_plot(
    epoch: int,
    stage: str,
    input_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    save_dir: str
):
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Detach and convert to numpy
    input_np = input_tensor.detach().cpu().numpy()
    pred_np = pred_tensor.detach().cpu().numpy()
    target_np = target_tensor.detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    try:
        from utils.visualization import percentile_stretch
    except ImportError:
        def percentile_stretch(img, low=2, high=98):
            if img.ndim == 3:
                stretched = np.zeros_like(img)
                for i in range(img.shape[-1]):
                    stretched[..., i] = percentile_stretch(img[..., i], low, high)
                return stretched.astype(np.uint8)
            low_val = np.percentile(img, low)
            high_val = np.percentile(img, high)
            stretched = np.clip(img, low_val, high_val)
            stretched = (stretched - low_val) * 255.0 / (high_val - low_val + 1e-5)
            return stretched.astype(np.uint8)
            
    if stage == "super_res":
        inp = input_np[0]
        pred = pred_np[0]
        tgt = target_np[0]
        
        # Use target's min/max to lock the color scale across all three plots
        vmin = tgt.min()
        vmax = tgt.max()
        
        im0 = axes[0].imshow(inp, cmap='inferno', vmin=vmin, vmax=vmax)
        axes[0].set_title("Input (Low-Res 200m)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        im1 = axes[1].imshow(pred, cmap='inferno', vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Prediction (Super-Res) - Ep {epoch}")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        im2 = axes[2].imshow(tgt, cmap='inferno', vmin=vmin, vmax=vmax)
        axes[2].set_title("Target (High-Res 100m)")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
    else:  # stage == "colorization"
        inp = input_np[0]
        pred = np.transpose(pred_np, (1, 2, 0))
        tgt = np.transpose(target_np, (1, 2, 0))
        
        def render_rgb(img):
            if np.nanmin(img) >= -0.01 and np.nanmax(img) <= 1.01:
                return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            return percentile_stretch(img)

        pred_rendered = render_rgb(pred)
        tgt_rendered = render_rgb(tgt)
        
        im0 = axes[0].imshow(inp, cmap='inferno')
        axes[0].set_title("Input (Thermal 100m)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        axes[1].imshow(pred_rendered)
        axes[1].set_title(f"Prediction (Colorized) - Ep {epoch}")
        
        axes[2].imshow(tgt_rendered)
        axes[2].set_title("Target (Ground Truth RGB)")
        
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{stage}_epoch_{epoch}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info(f"Saved side-by-side comparison plot to: {plot_path}")
    
    # Try inline display in Jupyter/Colab notebooks
    try:
        from IPython.display import display, Image
        display(Image(filename=plot_path))
    except Exception:
        pass


def train_super_res(
    patches_dir: str, 
    metadata_json_path: str, 
    checkpoint_dir: str, 
    epochs: int, 
    batch_size: int, 
    device: torch.device,
    model_type: str = "hat",
    lambda_physics: float = 0.1,
    lambda_ssim: float = 0.1,
    lr: float = 1e-4,
    split: str = "train"
):
    logger.info(f"Starting Stage 1 ({model_type.upper()}) Training Loop on split='{split}'")
    
    # 1. Dataset & DataLoader (train with patch size of 64 to avoid CUDA OOM)
    train_dataset = ISROPhysicsDataset(
        patches_dir=patches_dir, 
        metadata_json_path=metadata_json_path, 
        stage="super_res",
        patch_size=64,
        split=split
    )
    if len(train_dataset) == 0:
        logger.error("No training samples found for Super-Resolution. Skipping Stage 1.")
        return

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Separate full-resolution dataset for validation plotting
    val_dataset = ISROPhysicsDataset(
        patches_dir=patches_dir, 
        metadata_json_path=metadata_json_path, 
        stage="super_res",
        patch_size=None,
        split="val" if split == "train" else "all"
    )

    # 2. Model, Loss, Optimizer
    if model_type.lower() == "cnn":
        model = ThermalSuperResNet().to(device)
    else:
        model = ThermalHATWrapper().to(device)

    loss_fn = Stage1PhysicsStructuralLoss(lambda_physics=lambda_physics, lambda_ssim=lambda_ssim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_mse = 0.0
        running_ssim = 0.0
        running_physics = 0.0
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Stage 1 - Epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar):
            torch.cuda.empty_cache() # Prevent VRAM fragmentation
            # Unpack values and move tensors to target device
            inputs = batch["input"].to(device)     # (B, 1, 256, 256)
            targets = batch["target"].to(device)   # (B, 1, 512, 512)

            # Zero gradient buffer
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)                # (B, 1, 512, 512)

            # Compute loss
            loss, metrics = loss_fn(
                pred_100m=outputs, 
                gt_100m=targets, 
                input_200m=inputs, 
                batch_constants=batch
            )

            # Backward pass & step
            loss.backward()
            optimizer.step()

            running_loss += metrics["total"]
            running_mse += metrics["mse"]
            running_ssim += metrics["ssim"]
            running_physics += metrics["physics"]

            # Update progress bar
            pbar.set_postfix({'loss': f'{metrics["total"]:.4f}', 'mse': f'{metrics["mse"]:.4f}'})

        n_batches = len(dataloader)
        epoch_loss = running_loss / n_batches
        logger.info(f"Super-Res Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.6f} | MSE: {running_mse/n_batches:.6f} | SSIM: {running_ssim/n_batches:.4f} | Physics: {running_physics/n_batches:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = os.path.join(checkpoint_dir, "super_res_model_best.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"*** New best Stage 1 model found (Loss: {best_loss:.6f})! Saved to super_res_model_best.pth ***")

        # Save side-by-side comparison for this epoch (random sample)
        if len(val_dataset) > 0:
            import random
            rand_idx = random.randint(0, len(val_dataset) - 1)
            val_sample = val_dataset[rand_idx]
            val_input = val_sample["input"].unsqueeze(0).to(device)
            val_target = val_sample["target"].unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                val_pred = model(val_input)
            save_comparison_plot(
                epoch=epoch,
                stage="super_res",
                input_tensor=val_input[0],
                pred_tensor=val_pred[0],
                target_tensor=val_target[0],
                save_dir=os.path.join(checkpoint_dir, "comparisons")
            )

        # VRAM Cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 4. Save Weights (Last epoch)
    model_path = os.path.join(checkpoint_dir, "super_res_model_last.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Stage 1 last epoch weights successfully saved to: {model_path}\n")


def train_colorizer(
    patches_dir: str, 
    metadata_json_path: str, 
    checkpoint_dir: str, 
    epochs: int, 
    batch_size: int, 
    device: torch.device,
    metadata_mode: str = "film",
    use_global_vit: bool = True,
    use_cnn_skips: bool = True,
    lambda_ssim: float = 0.5,
    lambda_chrom: float = 0.2,
    lambda_pad: float = 0.05,
    lambda_grad: float = 0.05,
    lr: float = 2e-4,
    split: str = "train"
):
    logger.info(f"Starting Stage 2 Colorization Training (metadata_mode='{metadata_mode}', vit={use_global_vit}, skips={use_cnn_skips}) on split='{split}'")

    # 1. Dataset & DataLoader
    dataset = ISROPhysicsDataset(
        patches_dir=patches_dir, 
        metadata_json_path=metadata_json_path, 
        stage="colorization",
        split=split
    )
    if len(dataset) == 0:
        logger.error("No training samples found for Colorization. Skipping Stage 2.")
        return
    # 2. Model, Loss, Optimizer
    gpu_count = torch.cuda.device_count()
    logger.info(f"Using user-provided batch size of {batch_size} across {gpu_count} GPU(s).")
    
    import os
    workers = min(4, os.cpu_count() or 1)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=workers,
        pin_memory=True
    )

    model = ThermalColorizerViT(
        metadata_mode=metadata_mode,
        use_global_vit=use_global_vit,
        use_cnn_skips=use_cnn_skips,
        metadata_dropout=0.2 if metadata_mode == "film" else 0.0
    ).to(device)
    
    # Enable activation offloading (Gradient Checkpointing) to prevent OOM
    if use_global_vit and hasattr(model, "encoder") and model.encoder is not None:
        try:
            model.encoder.gradient_checkpointing_enable()
            logger.info("Gradient Checkpointing enabled for ViT-Huge.")
        except Exception:
            pass

    if gpu_count > 1:
        logger.info(f"Enabling nn.DataParallel across {gpu_count} GPUs!")
        model = nn.DataParallel(model)

    loss_fn = Stage2ColorizationLoss(lambda_ssim=lambda_ssim, lambda_chrom=lambda_chrom, lambda_pad=lambda_pad)
    gradient_alignment_loss = CrossModalGradientAlignmentLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_mse = 0.0
        running_ssim = 0.0
        running_psnr = 0.0
        running_color = 0.0
        running_grad_loss = 0.0
        running_loss = 0.0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Stage 2 - Epoch {epoch}/{epochs}")
        for batch_idx, batch in pbar:
            # Unpack values and move tensors to target device
            inputs = batch["input"].to(device)     # (B, 1, 512, 512)
            targets = batch["target"].to(device)   # (B, 3, 512, 512)
            valid_mask = batch.get("valid_mask")
            if valid_mask is not None:
                valid_mask = valid_mask.to(device)

            # Zero gradient
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, metadata=batch)                # (B, 3, 512, 512)

            # Compute loss
            base_loss, metrics = loss_fn(outputs, targets, valid_mask=valid_mask)
            
            if lambda_grad > 0:
                grad_loss = gradient_alignment_loss(outputs, inputs, valid_mask=valid_mask)
                loss = base_loss + (lambda_grad * grad_loss)
                metrics["grad_loss"] = grad_loss.item()
            else:
                loss = base_loss
                metrics["grad_loss"] = 0.0
            
            metrics["total"] = loss.item()

            # Backward pass & step
            loss.backward()
            optimizer.step()

            running_loss += metrics["total"]
            running_mse += metrics["mse"]
            running_ssim += metrics["ssim"]
            running_psnr += metrics["psnr"]
            running_color += metrics["color"]
            running_grad_loss += metrics["grad_loss"]

            # Update progress bar
            pbar.set_postfix({'loss': f'{metrics["total"]:.4f}', 'psnr': f'{metrics["psnr"]:.2f}dB'})

        n_batches = len(dataloader)
        epoch_loss = running_loss / n_batches
        epoch_grad_loss = running_grad_loss / n_batches
        logger.info(f"Colorizer Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.6f} | Grad: {epoch_grad_loss:.6f} | MSE: {metrics['mse']:.6f} | Color: {metrics['color']:.4f} | SSIM: {metrics['ssim']:.4f} | PSNR: {metrics['psnr']:.2f}dB | Valid: {metrics['valid_ratio']:.2%}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = os.path.join(checkpoint_dir, "colorizer_model_best.pth")
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, best_model_path)
            logger.info(f"*** New best Stage 2 model found (Loss: {best_loss:.6f})! Saved to colorizer_model_best.pth ***")

        # Save side-by-side comparison for this epoch (random sample)
        if len(dataset) > 0:
            import random
            rand_idx = random.randint(0, len(dataset) - 1)
            val_sample = dataset[rand_idx]
            val_input = val_sample["input"].unsqueeze(0).to(device)
            val_target = val_sample["target"].unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                # Unwrap the DataParallel module for single-image inference to prevent NCCL CUDA crashes
                infer_model = model.module if isinstance(model, nn.DataParallel) else model
                
                # Extract metadata for this single sample and add batch dimension
                val_metadata = {
                    "center_lat": val_sample["center_lat"].unsqueeze(0).to(device),
                    "center_lon": val_sample["center_lon"].unsqueeze(0).to(device),
                    "sun_elevation": val_sample["sun_elevation"].unsqueeze(0).to(device),
                    "sun_azimuth": val_sample["sun_azimuth"].unsqueeze(0).to(device)
                }
                
                val_pred = infer_model(val_input, metadata=val_metadata)
            save_comparison_plot(
                epoch=epoch,
                stage="colorization",
                input_tensor=val_input[0],
                pred_tensor=val_pred[0],
                target_tensor=val_target[0],
                save_dir=os.path.join(checkpoint_dir, "comparisons")
            )

        # VRAM Cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 4. Save Weights (Last epoch)
    model_path = os.path.join(checkpoint_dir, "colorizer_model_last.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Stage 2 last epoch weights successfully saved to: {model_path}\n")


def train_pipeline(
    patches_dir: str, 
    metadata_json_path: str, 
    epochs: int = 5, 
    batch_size: int = 4,
    stages: list = [1, 2],
    stage1_arch: str = "hat",
    lambda_physics: float = 0.1,
    lambda_ssim_sr: float = 0.1,
    metadata_mode: str = "film",
    use_global_vit: bool = True,
    use_cnn_skips: bool = True,
    lambda_grad: float = 0.05,
    lambda_chrom: float = 0.2,
    split: str = "train"
):
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Selected Device: {device}")
    if device.type == "cuda":
        logger.info(f"Active GPU Name: {torch.cuda.get_device_name(0)}")

    # Ensure checkpoint output directory exists
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Walkthrough absolute path validations
    patches_dir = os.path.abspath(patches_dir)
    metadata_json_path = os.path.abspath(metadata_json_path)

    logger.info(f"Resolved Patches Dir: {patches_dir}")
    logger.info(f"Resolved Metadata JSON Path: {metadata_json_path}")
    logger.info(f"Resolved Checkpoints Dir: {checkpoint_dir}")

    # Part A: Train Super-Resolution model
    if 1 in stages:
        train_super_res(
            patches_dir=patches_dir,
            metadata_json_path=metadata_json_path,
            checkpoint_dir=checkpoint_dir,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            model_type=stage1_arch,
            lambda_physics=lambda_physics,
            lambda_ssim=lambda_ssim_sr,
            split=split
        )

    # Part B: Train Colorization model
    if 2 in stages:
        train_colorizer(
            patches_dir=patches_dir,
            metadata_json_path=metadata_json_path,
            checkpoint_dir=checkpoint_dir,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            metadata_mode=metadata_mode,
            use_global_vit=use_global_vit,
            use_cnn_skips=use_cnn_skips,
            lambda_grad=lambda_grad,
            lambda_chrom=lambda_chrom,
            split=split
        )

    logger.info("End-to-end training pipeline finished successfully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train IRGB Physics-Informed Satellite Pipeline")
    parser.add_argument("--stage", type=str, default="both", choices=["1", "2", "both"], help="Which stage to train")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    
    # Stage 1 architecture & loss options
    parser.add_argument("--stage1_arch", type=str, default="hat", choices=["hat", "cnn"], help="Stage 1 architecture")
    parser.add_argument("--lambda_phy", type=float, default=0.1, help="Stage 1 physics energy conservation loss weight")
    parser.add_argument("--lambda_ssim_sr", type=float, default=0.1, help="Stage 1 SSIM loss weight")
    
    # Stage 2 architecture & loss options
    parser.add_argument("--metadata_mode", type=str, default="film", choices=["film", "concat", "none"], help="Stage 2 metadata conditioning mode")
    parser.add_argument("--no_vit", action="store_true", help="Disable ViT global branch (w/o I-JEPA global branch)")
    parser.add_argument("--no_skips", action="store_true", help="Disable CNN skip connections (w/o CNN skip branch)")
    parser.add_argument("--lambda_grad", type=float, default=0.05, help="Stage 2 gradient alignment loss weight")
    parser.add_argument("--lambda_chrom", type=float, default=0.2, help="Stage 2 chromaticity loss weight")
    
    # Data options
    parser.add_argument("--split", type=str, default="train", choices=["train", "all"], help="Dataset split for training")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_patches = os.path.join(base_dir, "output", "patches")
    default_metadata = os.path.join(base_dir, "data", "metadata_mapping.json")

    parser.add_argument("--patches_dir", type=str, default=default_patches, help="Path to patches directory")
    parser.add_argument("--metadata_json_path", type=str, default=default_metadata, help="Path to metadata JSON file")
    args = parser.parse_args()

    stages = [1, 2] if args.stage == "both" else [int(args.stage)]

    train_pipeline(
        patches_dir=args.patches_dir,
        metadata_json_path=args.metadata_json_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stages=stages,
        stage1_arch=args.stage1_arch,
        lambda_physics=args.lambda_phy,
        lambda_ssim_sr=args.lambda_ssim_sr,
        metadata_mode=args.metadata_mode,
        use_global_vit=not args.no_vit,
        use_cnn_skips=not args.no_skips,
        lambda_grad=args.lambda_grad,
        lambda_chrom=args.lambda_chrom,
        split=args.split
    )

