import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from typing import Dict, Tuple

def normalize_to_01(tensor: torch.Tensor, eps: float = 1e-8, per_channel: bool = False) -> torch.Tensor:
    """
    Dynamically min-max normalizes each item in the batch to [0.0, 1.0] range.
    Input shape: (B, C, H, W)
    """
    B = tensor.size(0)
    if per_channel:
        C = tensor.size(1)
        flat_tensor = tensor.view(B, C, -1)
        mins = flat_tensor.min(dim=2, keepdim=True)[0].view(B, C, 1, 1)
        maxs = flat_tensor.max(dim=2, keepdim=True)[0].view(B, C, 1, 1)
    else:
        flat_tensor = tensor.view(B, -1)
        mins = flat_tensor.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        maxs = flat_tensor.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    return (tensor - mins) / (maxs - mins + eps)

class Stage1PhysicsStructuralLoss(nn.Module):
    """
    Stage 1 Loss: Combines Pixel MSE, Physics-Enforced Energy Conservation, and SSIM.
    Supports ablating physics loss (lambda_physics=0.0) or SSIM loss (lambda_ssim=0.0).
    """
    def __init__(self, lambda_physics: float = 0.1, lambda_ssim: float = 0.1):
        super(Stage1PhysicsStructuralLoss, self).__init__()
        self.lambda_physics = lambda_physics
        self.lambda_ssim = lambda_ssim

    def forward(self, pred_100m: torch.Tensor, gt_100m: torch.Tensor, 
                input_200m: torch.Tensor, batch_constants: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Pixel MSE
        pixel_loss = F.mse_loss(pred_100m, gt_100m)

        # 2. Physics-Enforced Energy Conservation Constraint
        device = pred_100m.device
        mult = batch_constants["radiance_mult"].to(device).view(-1, 1, 1, 1)
        add = batch_constants["radiance_add"].to(device).view(-1, 1, 1, 1)

        pred_radiance_100m = (pred_100m * mult) + add
        input_radiance_200m = (input_200m * mult) + add

        pred_radiance_200m = F.avg_pool2d(pred_radiance_100m, kernel_size=2, stride=2)
        physics_loss = F.mse_loss(pred_radiance_200m, input_radiance_200m)
        rad_rmse = torch.sqrt(physics_loss + 1e-8)

        # 3. Structural Loss (SSIM)
        # Normalize to [0, 1] for stable SSIM and PSNR computation
        pred_norm = normalize_to_01(pred_100m)
        gt_norm = normalize_to_01(gt_100m)
        
        # ssim expects (B, C, H, W) with values in [0, 1]
        ssim_val = ssim(pred_norm, gt_norm, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val

        # PSNR on normalized [0, 1] thermal representation
        mse_norm = F.mse_loss(pred_norm, gt_norm)
        psnr_val = 10.0 * torch.log10(1.0 / (mse_norm + 1e-8))

        # 4. Total Loss
        total_loss = pixel_loss + (self.lambda_physics * physics_loss) + (self.lambda_ssim * ssim_loss)

        metrics = {
            "mse": pixel_loss.item(),
            "physics": physics_loss.item(),
            "rad_rmse": rad_rmse.item(),
            "ssim": ssim_val.item(),
            "psnr": psnr_val.item(),
            "total": total_loss.item()
        }
        return total_loss, metrics

class Stage2ColorizationLoss(nn.Module):
    """
    Stage 2 Loss: Combines Pixel L1, SSIM, Chromatic Angle, and Padding penalty.
    Supports ablating chromatic angle loss (lambda_chrom=0.0) or SSIM loss (lambda_ssim=0.0).
    """
    def __init__(
        self, 
        lambda_ssim: float = 0.5, 
        lambda_chrom: float = 0.2, 
        lambda_pad: float = 0.05,
        lambda_psnr: float = 0.0
    ):
        super(Stage2ColorizationLoss, self).__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_chrom = lambda_chrom
        self.lambda_pad = lambda_pad
        self.lambda_psnr = lambda_psnr

    def _prepare_target(self, gt_rgb: torch.Tensor) -> torch.Tensor:
        # New datasets provide normalized targets. Keep compatibility with older
        # raw-DN tensors by normalizing globally across all RGB channels.
        # CRITICAL: We must NOT normalize per channel, otherwise we destroy the R:G:B color ratios!
        if gt_rgb.detach().min().item() >= -0.01 and gt_rgb.detach().max().item() <= 1.01:
            return gt_rgb.clamp(0.0, 1.0)
        return normalize_to_01(gt_rgb, per_channel=False).clamp(0.0, 1.0)

    def forward(self, pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, valid_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Assume pred_rgb is already bounded to [0, 1] via Sigmoid
        gt_norm = self._prepare_target(gt_rgb)

        if valid_mask is None:
            valid_mask = torch.ones_like(pred_rgb[:, :1, :, :])
        elif valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        valid_mask = valid_mask.to(device=pred_rgb.device, dtype=pred_rgb.dtype)
        valid_pixels = valid_mask.sum().clamp_min(1.0)

        # 1. Pixel L1 Loss (masked so no-data borders do not teach black/gray)
        l1_map = F.l1_loss(pred_rgb, gt_norm, reduction="none")
        pixel_loss = (l1_map * valid_mask).sum() / (valid_pixels * pred_rgb.size(1))

        # 2. Structural Loss (SSIM)
        pred_for_ssim = (pred_rgb * valid_mask) + (gt_norm.detach() * (1.0 - valid_mask))
        gt_for_ssim = gt_norm
        ssim_val = ssim(pred_for_ssim, gt_for_ssim, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val

        # 3. Chromatic Angle Loss (Forces vibrant colors over grey)
        # Cosine similarity explicitly forces the RGB vector angles to match, ignoring magnitude (brightness)
        # This heavily penalizes predicting 'grey' (which is just a fixed vector [c, c, c])
        cos_sim = F.cosine_similarity(pred_rgb + 1e-6, gt_norm + 1e-6, dim=1)
        color_map = 1.0 - cos_sim
        color_loss = (color_map * valid_mask.squeeze(1)).sum() / valid_pixels

        # Angular Chromaticity Error in degrees for paper evaluation
        cos_sim_clamped = cos_sim.clamp(-1.0, 1.0)
        angle_deg = torch.acos(cos_sim_clamped) * (180.0 / 3.141592653589793)
        chrom_err_deg = (angle_deg * valid_mask.squeeze(1)).sum() / valid_pixels
        
        # 4. Padding Penalty (Keep black regions black so they don't hallucinate glowing blobs)
        padding_mask = 1.0 - valid_mask
        padding_pixels = padding_mask.sum().clamp_min(1.0)
        padding_loss = (pred_rgb * padding_mask).abs().sum() / padding_pixels
        
        # 5. Total Loss
        total_loss = pixel_loss + (self.lambda_ssim * ssim_loss) + (self.lambda_chrom * color_loss) + (self.lambda_pad * padding_loss)

        # 6. PSNR Metric calculation for logging
        mse_map = F.mse_loss(pred_rgb, gt_norm, reduction="none")
        mse_for_psnr = (mse_map * valid_mask).sum() / (valid_pixels * pred_rgb.size(1))
        psnr_val = 10 * torch.log10(1.0 / (mse_for_psnr + 1e-8))

        metrics = {
            "l1": pixel_loss.item(),
            "mse": mse_for_psnr.item(),
            "ssim": ssim_val.item(),
            "psnr": psnr_val.item(),
            "color": color_loss.item(),
            "chrom_err_deg": chrom_err_deg.item(),
            "valid_ratio": valid_mask.mean().item(),
            "total": total_loss.item()
        }
        return total_loss, metrics
