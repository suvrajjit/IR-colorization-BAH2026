import os
import json
import logging
from typing import Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _robust_normalize_2d(array: np.ndarray, valid_mask: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """
    Robustly maps a single-channel image to [0, 1] using only valid pixels.
    This keeps no-data borders from dominating the range.
    """
    array = array.astype(np.float32, copy=False)
    finite_mask = np.isfinite(array)
    mask = valid_mask & finite_mask
    values = array[mask]

    if values.size < 16:
        values = array[finite_mask]
    if values.size == 0:
        return np.zeros_like(array, dtype=np.float32)

    lo = np.percentile(values, low)
    hi = np.percentile(values, high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(values.min())
        hi = float(values.max())

    normalized = (array - lo) / (hi - lo + 1e-6)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)

def _robust_normalize_chw(array: np.ndarray, valid_mask: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """
    Global robust normalization for RGB targets in channel-first format (C, H, W).
    CRITICAL: We calculate a single global min/max across ALL 3 channels.
    Normalizing channels independently destroys the physical R:G:B ratios, which artificially flattens the image into grayscale/sepia!
    """
    array = array.astype(np.float32, copy=False)
    finite_mask = np.isfinite(array)
    mask = valid_mask[None, ...] & finite_mask
    
    values = array[mask]
    if values.size < 16:
        values = array[finite_mask]
    if values.size == 0:
        return np.zeros_like(array, dtype=np.float32)
        
    lo = np.percentile(values, low)
    hi = np.percentile(values, high)
    
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(values.min())
        hi = float(values.max())
        
    normalized = (array - lo) / (hi - lo + 1e-6)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)

# Module-level cache to prevent repeated slow directory scans over Kaggle network mounts
_CACHED_SAMPLES: Dict[str, list] = {}

class ISROPhysicsDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Landsat 9 image patches.
    Supports Stage 1 (Super-Resolution) and Stage 2 (Colorization) processing,
    integrating centralized metadata for physics-informed machine learning.
    """
    def __init__(
        self, 
        patches_dir: str, 
        metadata_json_path: str, 
        stage: str = "super_res", 
        patch_size: Optional[int] = None,
        split: str = "all",
        split_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        split_seed: int = 42
    ):
        """
        Args:
            patches_dir: Path to directory containing sample folders (flat or nested).
            metadata_json_path: Path to the metadata mapping JSON file.
            stage: Pipeline stage, either "super_res" or "colorization".
            patch_size: Optional size to randomly crop input patches during training.
            split: "train", "val", "test", or "all" for scene-level disjoint evaluation.
            split_ratio: (train_ratio, val_ratio, test_ratio).
            split_seed: Random seed for deterministic scene partitioning.
        """
        assert stage in ["super_res", "colorization"], "stage must be either 'super_res' or 'colorization'"
        assert split in ["train", "val", "test", "all"], "split must be 'train', 'val', 'test', or 'all'"
        self.patches_dir = patches_dir
        self.stage = stage
        self.patch_size = patch_size
        self.split = split

        # Load metadata mapping JSON
        self.metadata = {}
        if os.path.exists(metadata_json_path):
            try:
                with open(metadata_json_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata)} products.")
            except Exception as e:
                logger.error(f"Failed to load metadata JSON: {e}")
        else:
            logger.warning(f"Metadata JSON not found at {metadata_json_path}. Using fallback defaults.")

        # Fast 2-level directory scan to avoid slow network FUSE os.walk on cloud/Kaggle mounts
        global _CACHED_SAMPLES
        if patches_dir in _CACHED_SAMPLES:
            all_samples = _CACHED_SAMPLES[patches_dir]
        else:
            all_samples = []
            try:
                for scene in sorted(os.listdir(patches_dir)):
                    scene_path = os.path.join(patches_dir, scene)
                    if os.path.isdir(scene_path):
                        for sample in sorted(os.listdir(scene_path)):
                            sample_path = os.path.join(scene_path, sample)
                            if os.path.isdir(sample_path):
                                all_samples.append(sample_path)
            except Exception as e:
                logger.warning(f"Fast directory scan encountered an error: {e}. Falling back to os.walk.")
                all_samples = []

            # Fallback only if direct scan found nothing
            if len(all_samples) == 0:
                for root_dir, _, files in os.walk(patches_dir):
                    if "tir_200m.npy" in files or "tir_100m_512.npy" in files:
                        all_samples.append(root_dir)

            all_samples.sort()
            _CACHED_SAMPLES[patches_dir] = all_samples

        if split != "all" and len(all_samples) > 0:
            # Group samples by scene/product to guarantee zero spatial leakage across tiles
            scene_to_samples: Dict[str, list] = {}
            for s in all_samples:
                norm = os.path.abspath(s).replace("\\", "/")
                parts = norm.split('/')
                # If last part is sample_xxx, the scene is the parent folder
                scene_id = parts[-2] if len(parts) >= 2 and parts[-1].startswith("sample_") else parts[-1]
                scene_to_samples.setdefault(scene_id, []).append(s)

            unique_scenes = sorted(list(scene_to_samples.keys()))
            rng = np.random.RandomState(split_seed)
            shuffled_scenes = unique_scenes.copy()
            rng.shuffle(shuffled_scenes)

            n_scenes = len(shuffled_scenes)
            train_cut = int(round(split_ratio[0] * n_scenes))
            val_cut = train_cut + int(round(split_ratio[1] * n_scenes))

            if split == "train":
                selected_scenes = set(shuffled_scenes[:train_cut])
            elif split == "val":
                selected_scenes = set(shuffled_scenes[train_cut:val_cut])
            elif split == "test":
                selected_scenes = set(shuffled_scenes[val_cut:])

            self.sample_paths = []
            for sc in selected_scenes:
                self.sample_paths.extend(scene_to_samples[sc])
            self.sample_paths.sort()
            logger.info(f"Split '{split}' loaded {len(self.sample_paths)} patches from {len(selected_scenes)}/{n_scenes} scenes.")
        else:
            self.sample_paths = all_samples
            logger.info(f"Initialized dataset with {len(self.sample_paths)} samples for stage '{stage}' (split: {split}).")

    def _get_metadata(self, sample_path: str) -> Dict[str, Any]:
        """
        Dynamically looks up metadata variables corresponding to the patch sample.
        If a lookup fails, falls back gracefully to standard Landsat 9 Band 10 defaults.
        """
        # Fallback defaults (Landsat 9 Band 10 defaults)
        defaults = {
            "radiance_mult": 0.00038,
            "radiance_add": 0.1,
            "k1": 799.0284,
            "k2": 1329.2405,
            "doy": 1,
            "fractional_hour": 12.0,
            "sun_azimuth": 0.0,
            "sun_elevation": 0.0,
            "center_lat": 0.0,
            "center_lon": 0.0
        }

        if not self.metadata:
            return defaults

        # Normalize paths for platform independence
        normalized_path = os.path.abspath(sample_path).replace("\\", "/")
        path_parts = normalized_path.split('/')

        # Look for partial matches in the metadata keys
        for key in self.metadata:
            # Match by directory substring
            if key in normalized_path:
                return self.metadata[key]

            # Match by path-row and date code combination
            key_parts = key.split('_')
            if len(key_parts) >= 5:
                path_row = key_parts[2]
                date_str = key_parts[3]
                if any(path_row in p and date_str in p for p in path_parts):
                    return self.metadata[key]

        # Check if any path directory part exactly matches a key
        for part in path_parts:
            if part in self.metadata:
                return self.metadata[part]

        return defaults

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_path = self.sample_paths[idx]
        meta = self._get_metadata(sample_path)

        if self.stage == "super_res":
            # Input: 200m TIR (256x256), Target: 100m TIR (512x512)
            input_arr = np.load(os.path.join(sample_path, "tir_200m.npy"))
            target_arr = np.load(os.path.join(sample_path, "tir_100m_512.npy"))

            # Squeeze any existing 3rd dimension if it is of size 1 (e.g. (1, H, W) or (H, W, 1))
            if input_arr.ndim == 3:
                if input_arr.shape[0] == 1:
                    input_arr = input_arr[0]
                elif input_arr.shape[-1] == 1:
                    input_arr = input_arr[..., 0]

            if target_arr.ndim == 3:
                if target_arr.shape[0] == 1:
                    target_arr = target_arr[0]
                elif target_arr.shape[-1] == 1:
                    target_arr = target_arr[..., 0]

            # Random crop for training if patch_size is specified
            if self.patch_size is not None:
                h_in, w_in = input_arr.shape
                if h_in >= self.patch_size and w_in >= self.patch_size:
                    y = np.random.randint(0, h_in - self.patch_size + 1)
                    x = np.random.randint(0, w_in - self.patch_size + 1)
                    input_arr = input_arr[y:y+self.patch_size, x:x+self.patch_size]
                    
                    # Target is 2x resolution
                    y_tgt = y * 2
                    x_tgt = x * 2
                    patch_size_tgt = self.patch_size * 2
                    target_arr = target_arr[y_tgt:y_tgt+patch_size_tgt, x_tgt:x_tgt+patch_size_tgt]

            # Convert to float32 and add channel dimension: (1, H, W)
            input_tensor = torch.from_numpy(input_arr.copy()).float().unsqueeze(0)
            target_tensor = torch.from_numpy(target_arr.copy()).float().unsqueeze(0)

        else:  # self.stage == "colorization"
            # Input: 100m TIR (512x512), Target: 100m RGB (512x512x3)
            input_arr = np.load(os.path.join(sample_path, "tir_100m_512.npy"))
            target_arr = np.load(os.path.join(sample_path, "rgb_100m_512.npy"))

            # Squeeze TIR input if it has channel dimension
            if input_arr.ndim == 3:
                if input_arr.shape[0] == 1:
                    input_arr = input_arr[0]
                elif input_arr.shape[-1] == 1:
                    input_arr = input_arr[..., 0]

            # Permute RGB array to channel-first format (C, H, W) if channel-last
            if target_arr.ndim == 3:
                if target_arr.shape[0] != 3 and target_arr.shape[-1] == 3:
                    target_arr = np.moveaxis(target_arr, -1, 0).copy()
            else:
                raise ValueError(f"RGB array must have 3 dimensions, got {target_arr.ndim} in {sample_path}")

            # No-data areas in Landsat patches are stored as zeros. Keep a spatial
            # mask so the loss does not train on black borders or missing imagery.
            valid_mask = (
                np.isfinite(input_arr)
                & (input_arr > 0)
                & np.all(np.isfinite(target_arr), axis=0)
                & np.any(target_arr > 0, axis=0)
            )

            # Random crop for training if patch_size is specified
            if self.patch_size is not None:
                h_in, w_in = input_arr.shape
                if h_in >= self.patch_size and w_in >= self.patch_size:
                    y = np.random.randint(0, h_in - self.patch_size + 1)
                    x = np.random.randint(0, w_in - self.patch_size + 1)
                    input_arr = input_arr[y:y+self.patch_size, x:x+self.patch_size]
                    target_arr = target_arr[:, y:y+self.patch_size, x:x+self.patch_size]
                    valid_mask = valid_mask[y:y+self.patch_size, x:x+self.patch_size]

            input_arr = _robust_normalize_2d(input_arr, valid_mask)
            target_arr = _robust_normalize_chw(target_arr, valid_mask)

            # Convert input TIR to float32 and add channel dimension: (1, H, W)
            input_tensor = torch.from_numpy(input_arr.copy()).float().unsqueeze(0)
            target_tensor = torch.from_numpy(target_arr.copy()).float()
            valid_mask_tensor = torch.from_numpy(valid_mask.astype(np.float32, copy=True)).unsqueeze(0)

        # Build output batch items
        item = {
            "input": input_tensor,
            "target": target_tensor,
            "radiance_mult": torch.tensor(meta.get("radiance_mult", 0.00038), dtype=torch.float32),
            "radiance_add": torch.tensor(meta.get("radiance_add", 0.1), dtype=torch.float32),
            "k1": torch.tensor(meta.get("k1", 799.0284), dtype=torch.float32),
            "k2": torch.tensor(meta.get("k2", 1329.2405), dtype=torch.float32),
            "doy": torch.tensor(meta.get("doy", 1), dtype=torch.int32),
            "fractional_hour": torch.tensor(meta.get("fractional_hour", 12.0), dtype=torch.float32),
            "sun_azimuth": torch.tensor(meta.get("sun_azimuth", 0.0), dtype=torch.float32),
            "sun_elevation": torch.tensor(meta.get("sun_elevation", 0.0), dtype=torch.float32),
            "center_lat": torch.tensor(meta.get("center_lat", 0.0), dtype=torch.float32),
            "center_lon": torch.tensor(meta.get("center_lon", 0.0), dtype=torch.float32)
        }
        if self.stage == "colorization":
            item["valid_mask"] = valid_mask_tensor
        return item


class PhysicsInformedLoss(nn.Module):
    """
    Custom Physics-Informed Loss Function for Stage 1 (Super-Resolution).
    Combines standard target MSE loss with an energy conservation constraint
    enforced by downsampling Top of Atmosphere (TOA) Spectral Radiance.
    """
    def __init__(self, lambda_physics: float = 0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.lambda_physics = lambda_physics

    def forward(self, pred_100m: torch.Tensor, gt_100m: torch.Tensor, 
                input_200m: torch.Tensor, batch_constants: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            pred_100m: Model's predicted 100m resolution TIR map (B, 1, 512, 512).
            gt_100m: Ground truth 100m resolution TIR target map (B, 1, 512, 512).
            input_200m: Low-res 200m resolution input TIR map (B, 1, 256, 256).
            batch_constants: Dictionary containing batches of physical scaling variables.
        """
        # 1. Standard reconstruction loss (MSE) at 100m resolution
        pixel_loss = F.mse_loss(pred_100m, gt_100m)

        # 2. Physics-Enforced Energy Conservation Constraint
        # Extract and reshape scaling parameters to match (B, 1, 1, 1) dimensions
        device = pred_100m.device
        mult = batch_constants["radiance_mult"].to(device).view(-1, 1, 1, 1)
        add = batch_constants["radiance_add"].to(device).view(-1, 1, 1, 1)

        # Convert DN values to TOA Spectral Radiance (L_lambda)
        # Radiance = (DN * radiance_mult) + radiance_add
        pred_radiance_100m = (pred_100m * mult) + add
        input_radiance_200m = (input_200m * mult) + add

        # Downsample the predicted 100m radiance back to 200m resolution footprint (2x2 average pooling)
        pred_radiance_200m = F.avg_pool2d(pred_radiance_100m, kernel_size=2, stride=2)

        # Energy conservation loss (MSE on the downsampled 200m radiance footprint)
        physics_loss = F.mse_loss(pred_radiance_200m, input_radiance_200m)

        # 3. Total Weighted Loss
        total_loss = pixel_loss + self.lambda_physics * physics_loss

        return total_loss


class CrossModalGradientAlignmentLoss(nn.Module):
    """
    Structural Boundary Alignment Loss using Sobel filters.
    Computes the cosine similarity of spatial gradients between the 1-channel TIR input
    and the predicted 3-channel RGB output to ensure structural and topographical boundaries align perfectly.
    """
    def __init__(self):
        super(CrossModalGradientAlignmentLoss, self).__init__()
        
        # Define 3x3 Sobel kernels
        sobel_x = torch.tensor([[-1.,  0.,  1.],
                                [-2.,  0.,  2.],
                                [-1.,  0.,  1.]], dtype=torch.float32)
        
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32)
        
        # Reshape to (out_channels, in_channels, H, W)
        # We'll apply this independently to each channel using groups=in_channels
        # So shape is (1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def _compute_gradients(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the x and y gradients for an image tensor (B, C, H, W).
        Applies the Sobel filter independently to each channel.
        """
        B, C, H, W = img.shape
        
        # Duplicate kernels for each channel
        weight_x = self.sobel_x.repeat(C, 1, 1, 1)
        weight_y = self.sobel_y.repeat(C, 1, 1, 1)
        
        # Apply conv2d with groups=C to apply the filter independently to each channel
        grad_x = F.conv2d(img, weight_x, padding=1, groups=C)
        grad_y = F.conv2d(img, weight_y, padding=1, groups=C)
        
        return grad_x, grad_y

    def forward(self, pred_rgb: torch.Tensor, input_tir: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred_rgb: Model's predicted 3-channel RGB image (B, 3, H, W)
            input_tir: High-res 1-channel TIR image (B, 1, H, W)
        """
        # --- THE FIX ---
        # If we force R, G, and B spatial gradients to independently match the thermal image,
        # the only mathematical solution is to make R=G=B (which equals grey/sepia!).
        # Instead, we convert RGB to a single Luminance (grayscale) channel.
        # This forces the *brightness structure* to match the thermal edges, but leaves the *colors* completely free to diverge!
        pred_luma = 0.299 * pred_rgb[:, 0:1, :, :] + 0.587 * pred_rgb[:, 1:2, :, :] + 0.114 * pred_rgb[:, 2:3, :, :]
        
        # 1. Compute gradients on 1-channel tensors
        u_x, u_y = self._compute_gradients(input_tir) # Shape: (B, 1, H, W)
        v_x, v_y = self._compute_gradients(pred_luma) # Shape: (B, 1, H, W)
        
        # 2. Compute Cosine Similarity of the structural gradients
        mag_u = torch.sqrt(u_x**2 + u_y**2 + 1e-8)
        mag_v = torch.sqrt(v_x**2 + v_y**2 + 1e-8)
        
        dot_product = (u_x * v_x) + (u_y * v_y)
        cos_theta = dot_product / (mag_u * mag_v)

        loss_map = 1.0 - cos_theta
        if valid_mask is not None:
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)
            valid_mask = valid_mask.to(device=loss_map.device, dtype=loss_map.dtype)
            loss = (loss_map * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        else:
            loss = loss_map.mean()
        
        return loss

# --- Unit Test & Verification block ---
if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    patches_dir = os.path.join(base_dir, "output", "patches")
    metadata_json = os.path.join(base_dir, "data", "metadata_mapping.json")

    logger.info("=== Testing ISROPhysicsDataset in 'super_res' mode ===")
    try:
        sr_dataset = ISROPhysicsDataset(
            patches_dir=patches_dir,
            metadata_json_path=metadata_json,
            stage="super_res"
        )
        
        if len(sr_dataset) > 0:
            sample = sr_dataset[0]
            logger.info(f"Successfully loaded sample 0.")
            logger.info(f"Input shape: {sample['input'].shape} (expected: [1, 256, 256])")
            logger.info(f"Target shape: {sample['target'].shape} (expected: [1, 512, 512])")
            logger.info(f"Metadata constants: radiance_mult={sample['radiance_mult'].item():.6f}, "
                        f"radiance_add={sample['radiance_add'].item():.6f}, doy={sample['doy'].item()}")
            
            # Setup DataLoader and Test Loss Function
            sr_loader = DataLoader(sr_dataset, batch_size=2, shuffle=True)
            batch = next(iter(sr_loader))
            logger.info("Successfully fetched a batch from DataLoader.")
            logger.info(f"Batch input shape: {batch['input'].shape}")
            logger.info(f"Batch target shape: {batch['target'].shape}")

            # Instantiate and test loss function
            loss_fn = PhysicsInformedLoss(lambda_physics=0.5)
            # Create dummy model predictions: (B, 1, 512, 512)
            mock_preds = batch['target'] + torch.randn_like(batch['target']) * 10.0
            
            loss_val = loss_fn(
                pred_100m=mock_preds,
                gt_100m=batch['target'],
                input_200m=batch['input'],
                batch_constants=batch
            )
            logger.info(f"PhysicsInformedLoss executed successfully. Loss value: {loss_val.item():.4f}")
        else:
            logger.warning("No samples found. Please check patches_dir.")
            
    except Exception as e:
        logger.error(f"Error during 'super_res' dataset testing: {e}", exc_info=True)

    logger.info("\n=== Testing ISROPhysicsDataset in 'colorization' mode ===")
    try:
        color_dataset = ISROPhysicsDataset(
            patches_dir=patches_dir,
            metadata_json_path=metadata_json,
            stage="colorization"
        )
        
        if len(color_dataset) > 0:
            sample = color_dataset[0]
            logger.info(f"Successfully loaded sample 0.")
            logger.info(f"Input shape: {sample['input'].shape} (expected: [1, 512, 512])")
            logger.info(f"Target shape: {sample['target'].shape} (expected: [3, 512, 512])")
        else:
            logger.warning("No samples found. Please check patches_dir.")
            
    except Exception as e:
        logger.error(f"Error during 'colorization' dataset testing: {e}", exc_info=True)
