import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Residual block with two convolutional layers."""
    def __init__(self, channels: int):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))

class ThermalSuperResNet(nn.Module):
    """
    Stage 1 Model: Thermal Super-Resolution Network.
    Upscales a low-resolution single-channel thermal patch by a factor of 2.
    Input shape:  (B, 1, 256, 256)
    Output shape: (B, 1, 512, 512)
    """
    def __init__(self, num_res_blocks: int = 4):
        super(ThermalSuperResNet, self).__init__()
        # Initial feature extraction layer
        self.conv_init = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu_init = nn.ReLU(inplace=True)
        
        # Residual blocks backbone
        res_blocks = [ResBlock(64) for _ in range(num_res_blocks)]
        self.res_backbone = nn.Sequential(*res_blocks)
        
        # Sub-pixel upscaling layer
        # PixelShuffle(2) expects input channels = output channels * (2^2)
        # For single channel output, input channels = 1 * 4 = 4
        self.conv_upscale = nn.Conv2d(64, 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 1, 256, 256)
        out = self.relu_init(self.conv_init(x)) # Shape: (B, 64, 256, 256)
        out = self.res_backbone(out)            # Shape: (B, 64, 256, 256)
        out = self.conv_upscale(out)            # Shape: (B, 4, 256, 256)
        out = self.pixel_shuffle(out)           # Shape: (B, 1, 512, 512)
        return F.relu(out)


class DoubleConv(nn.Module):
    """Helper module with double convolution layers for U-Net."""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class ThermalColorizerNet(nn.Module):
    """
    Stage 2 Model: Cross-Modal Colorization Network.
    Translates a high-resolution single-channel thermal map into a 3-channel RGB image.
    Uses a symmetrical U-Net architecture with skip connections.
    Input shape:  (B, 1, 512, 512)
    Output shape: (B, 3, 512, 512)
    """
    def __init__(self):
        super(ThermalColorizerNet, self).__init__()
        # Encoder (Downsampling) path
        self.inc = DoubleConv(1, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Decoder (Upsampling) path with skip connection handling
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128) # Cat with down2 (128 channels) -> 256 input
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)  # Cat with down1 (64 channels) -> 128 input
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)   # Cat with inc (32 channels) -> 64 input
        
        # Out projection layer (3 output channels for RGB)
        self.outc = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 1, 512, 512)
        x1 = self.inc(x)         # (B, 32, 512, 512)
        x2 = self.down1(x1)      # (B, 64, 256, 256)
        x3 = self.down2(x2)      # (B, 128, 128, 128)
        x4 = self.down3(x3)      # (B, 256, 64, 64)

        # Decode & skip connections
        u1 = self.up1(x4)        # (B, 128, 128, 128)
        u1 = torch.cat([u1, x3], dim=1) # (B, 256, 128, 128)
        u1 = self.conv_up1(u1)   # (B, 128, 128, 128)

        u2 = self.up2(u1)        # (B, 64, 256, 256)
        u2 = torch.cat([u2, x2], dim=1) # (B, 128, 256, 256)
        u2 = self.conv_up2(u2)   # (B, 64, 256, 256)

        u3 = self.up3(u2)        # (B, 32, 512, 512)
        u3 = torch.cat([u3, x1], dim=1) # (B, 64, 512, 512)
        u3 = self.conv_up3(u3)   # (B, 32, 512, 512)

        logits = self.outc(u3)   # (B, 3, 512, 512)
        return torch.sigmoid(logits)


class ThermalHATWrapper(nn.Module):
    """
    Wrapper for the Hybrid Attention Transformer (HAT) to integrate with our pipeline.
    Enforces a ReLU activation at the output to ensure mathematically valid DN predictions (>=0).
    """
    def __init__(self):
        super(ThermalHATWrapper, self).__init__()
        
        import sys
        import os
        
        # Dynamically add the HAT repository directory to sys.path
        # Checking multiple possible paths to handle both local and Colab environments
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        possible_hat_dirs = [
            os.path.join(current_dir, 'HAT'),         # If uploaded directly alongside models.py
            os.path.join(parent_dir, 'HAT'),          # Local Windows structure
            '/content/HAT',                           # If cloned directly in Colab
            '/kaggle/working/HAT'                     # If cloned directly in Kaggle
        ]
        
        hat_dir = None
        for d in possible_hat_dirs:
            if os.path.exists(d) and os.path.isdir(d):
                hat_dir = d
                break
                
        if hat_dir is None:
            raise ImportError("Could not find the HAT repository. Please ensure it is cloned as 'HAT' in the project directory.")
            
        if hat_dir not in sys.path:
            sys.path.insert(0, hat_dir)
            
        # Monkey-patch torchvision to fix compatibility issues with basicsr==1.3.4.9
        # In newer torchvision versions, functional_tensor was removed.
        try:
            import torchvision.transforms.functional_tensor
        except ImportError:
            import torchvision.transforms.functional as F_vision
            sys.modules['torchvision.transforms.functional_tensor'] = F_vision
            
        try:
            from hat.archs.hat_arch import HAT
        except ImportError as e:
            raise ImportError(f"Could not import HAT from {hat_dir}. Error: {e}")
            
        # Initialize a "Small" HAT variant to save VRAM and train faster on Colab
        # A standard HAT uses depths=(6,6,6,6,6,6) and embed_dim=180. We scale it down slightly.
        self.hat = HAT(
            upscale=2,
            in_chans=1,
            img_size=64,                 # Matches training patch size (multiple of window_size=16)
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            mlp_ratio=2.0,
            depths=(6, 6, 6, 6),
            num_heads=(6, 6, 6, 6),
            embed_dim=96,                # Reduced from 144 to 96
            resi_connection='1conv',
            upsampler='pixelshuffle',
            use_checkpoint=True          # Highly critical for avoiding OOM!
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.hat(x)
        # Bounding output to strictly non-negative values for uncalibrated DNs
        return F.relu(out)

class ThermalColorizerViT(nn.Module):
    """
    Stage 2 Model: Cross-Modal Colorization Network using ViT-Huge Encoder and Restormer Decoder.
    Translates a high-resolution single-channel thermal map into a 3-channel RGB image.
    Supports Table 1 ablation configurations:
      - metadata_mode: "film" (default), "concat" (concatenated to input), or "none"
      - use_global_vit: True (default) or False (w/o I-JEPA global branch)
      - use_cnn_skips: True (default) or False (w/o CNN skip branch)
      - metadata_dropout: float (default 0.2)
    Input shape:  (B, 1, 512, 512)
    Output shape: (B, 3, 512, 512)
    """
    def __init__(
        self, 
        patch_size: int = 16, 
        embed_dim: int = 1280,
        metadata_mode: str = "film",
        use_global_vit: bool = True,
        use_cnn_skips: bool = True,
        metadata_dropout: float = 0.2
    ):
        super(ThermalColorizerViT, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.metadata_mode = metadata_mode.lower()
        self.use_global_vit = use_global_vit
        self.use_cnn_skips = use_cnn_skips
        self.metadata_dropout = metadata_dropout
        
        import sys
        import os
        
        # Dynamically add the ijepa repository directory to sys.path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        possible_ijepa_dirs = [
            os.path.join(current_dir, 'ijepa'),
            os.path.join(parent_dir, 'ijepa'),
            '/content/ijepa',
            '/kaggle/working/ijepa'
        ]
        
        ijepa_dir = None
        for d in possible_ijepa_dirs:
            if os.path.exists(d) and os.path.isdir(d):
                ijepa_dir = d
                break
                
        if ijepa_dir is not None and ijepa_dir not in sys.path:
            sys.path.insert(0, ijepa_dir)
            
        if self.use_global_vit:
            try:
                from transformers import ViTModel, ViTConfig
            except ImportError as e:
                raise ImportError(f"Could not import transformers. Error: {e}")
                
            # Instantiate HF Model from scratch (No Pretrained Weights)
            # Using the ViT-Huge architectural dimensions, initializing randomly for thermal training.
            config = ViTConfig(
                image_size=512,
                patch_size=16,
                num_channels=3,
                hidden_size=1280,
                num_hidden_layers=32,
                num_attention_heads=16,
                intermediate_size=5120
            )
            self.encoder = ViTModel(config, add_pooling_layer=False)
            self.bottleneck_proj = nn.Conv2d(self.embed_dim, 256, kernel_size=1)
        else:
            self.encoder = None
            # CNN bottleneck takes deepest c4 (256, 64, 64) -> (256, 32, 32)
            self.cnn_bottleneck = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # CNN Encoder for Skip Connections (High-resolution details)
        self.inc = DoubleConv(1, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Metadata handling modules
        if self.metadata_mode == "concat":
            # Concat 4-channel broadcast metadata to 1-channel thermal input -> 5 channels projected back to 1
            self.meta_input_proj = nn.Conv2d(1 + 4, 1, kernel_size=1)
        elif self.metadata_mode == "film":
            # FiLM Conditioning MLP for geospatial metadata
            self.metadata_mlp = nn.Sequential(
                nn.Linear(4, 128),
                nn.ReLU(),
                nn.Linear(128, self.embed_dim * 2)
            )
        
        # --- Restormer Decoder Import ---
        possible_restormer_dirs = [
            os.path.join(current_dir, 'Restormer'),
            os.path.join(parent_dir, 'Restormer'),
            '/kaggle/working/Restormer',
            '/content/Restormer',
            'd:\\Projects\\ISRO\\Restormer'
        ]
        restormer_dir = None
        for d in possible_restormer_dirs:
            if os.path.exists(d) and os.path.isdir(d):
                restormer_dir = d
                break
            
        if restormer_dir is not None:
            # Add the archs folder directly to sys.path to avoid basicsr's messy __init__.py dependencies
            archs_dir = os.path.join(restormer_dir, "basicsr", "models", "archs")
            if archs_dir not in sys.path and os.path.exists(archs_dir):
                sys.path.insert(0, archs_dir)
            if restormer_dir not in sys.path:
                sys.path.insert(0, restormer_dir)
            
        try:
            from restormer_arch import TransformerBlock, Upsample
        except ImportError as e:
            print(f"WARNING: Restormer failed to import. Error: {e}")
            print("Falling back to CNN.")
            TransformerBlock = None
            
        self.use_restormer = TransformerBlock is not None
        
        if self.use_restormer:
            # Restormer Configurations
            num_heads = [1, 2, 4, 8]
            expansion = 2.66
            LayerNorm_type = 'WithBias'
            
            # Stage 1: 32x32 -> 64x64
            self.up0 = Upsample(256) # outputs 128
            self.reduce0 = nn.Conv2d(128 + 256, 128, kernel_size=1)
            # Lightweight: 3 blocks
            self.dec0 = nn.Sequential(*[TransformerBlock(dim=128, num_heads=num_heads[3], ffn_expansion_factor=expansion, bias=False, LayerNorm_type=LayerNorm_type) for _ in range(3)])
            
            # Stage 2: 64x64 -> 128x128
            self.up1 = Upsample(128) # outputs 64
            self.reduce1 = nn.Conv2d(64 + 128, 64, kernel_size=1)
            # Lightweight: 2 blocks
            self.dec1 = nn.Sequential(*[TransformerBlock(dim=64, num_heads=num_heads[2], ffn_expansion_factor=expansion, bias=False, LayerNorm_type=LayerNorm_type) for _ in range(2)])
            
            # Stage 3: 128x128 -> 256x256
            self.up2 = Upsample(64) # outputs 32
            self.reduce2 = nn.Conv2d(32 + 64, 32, kernel_size=1)
            # Lightweight: 1 block instead of 2 at 256x256
            self.dec2 = nn.Sequential(*[TransformerBlock(dim=32, num_heads=num_heads[1], ffn_expansion_factor=expansion, bias=False, LayerNorm_type=LayerNorm_type) for _ in range(1)])
            
            # Stage 4: 256x256 -> 512x512
            self.up3 = Upsample(32) # outputs 16
            self.reduce3 = nn.Conv2d(16 + 32, 16, kernel_size=1)
            # Lightweight: Replace Transformer block with DoubleConv at 512x512 to prevent OOM
            self.dec3 = DoubleConv(16, 16)
            
            self.outc = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        else:
            # Fallback CNN Decoder
            self.up0 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
            self.conv_up0 = DoubleConv(256 + 256, 128)
            
            self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
            self.conv_up1 = DoubleConv(128 + 128, 64)
            
            self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
            self.conv_up2 = DoubleConv(64 + 64, 32)
            
            self.up3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
            self.conv_up3 = DoubleConv(32 + 32, 32)
            
            self.outc = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x: torch.Tensor, metadata=None) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # If metadata is concatenated to input (Table 1: Metadata concatenated to input instead of FiLM)
        if self.metadata_mode == "concat" and metadata is not None:
            metadata_vec = torch.cat([
                metadata["center_lat"].unsqueeze(1),
                metadata["center_lon"].unsqueeze(1),
                metadata["sun_elevation"].unsqueeze(1),
                metadata["sun_azimuth"].unsqueeze(1)
            ], dim=1).float().to(x.device)
            meta_spatial = metadata_vec.view(B, 4, 1, 1).expand(B, 4, H, W)
            x = self.meta_input_proj(torch.cat([x, meta_spatial], dim=1))

        # 1. CNN Encoder Path
        c1 = self.inc(x)      # (B, 32, 512, 512)
        c2 = self.down1(c1)   # (B, 64, 256, 256)
        c3 = self.down2(c2)   # (B, 128, 128, 128)
        c4 = self.down3(c3)   # (B, 256, 64, 64)
        c4_deepest = c4

        # Table 1: w/o CNN skip branch
        if not self.use_cnn_skips:
            c1 = torch.zeros_like(c1)
            c2 = torch.zeros_like(c2)
            c3 = torch.zeros_like(c3)
            c4 = torch.zeros_like(c4)

        # 2. Global Path (ViT vs CNN bottleneck)
        if self.use_global_vit:
            x_3ch = x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x
            vit_outputs = self.encoder(x_3ch) 
            x_encoded = vit_outputs.last_hidden_state # (B, 1025, 1280)
            
            x_spatial_tokens = x_encoded[:, 1:, :] # (B, 1024, 1280)
            grid_h, grid_w = H // self.patch_size, W // self.patch_size
            x_spatial = x_spatial_tokens.transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w)
            
            # FiLM Modulation (Table 1: Full IRGB uses FiLM, w/o metadata disables FiLM)
            if self.metadata_mode == "film" and metadata is not None:
                apply_metadata = True
                if self.training and torch.rand(1).item() < self.metadata_dropout:
                    apply_metadata = False
                    
                if apply_metadata:
                    metadata_vec = torch.cat([
                        metadata["center_lat"].unsqueeze(1),
                        metadata["center_lon"].unsqueeze(1),
                        metadata["sun_elevation"].unsqueeze(1),
                        metadata["sun_azimuth"].unsqueeze(1)
                    ], dim=1).float().to(x.device)
                    gamma_beta = self.metadata_mlp(metadata_vec)
                    gamma, beta = gamma_beta.chunk(2, dim=1)
                    x_spatial = x_spatial * gamma.view(B, -1, 1, 1) + beta.view(B, -1, 1, 1)
            
            bottleneck = self.bottleneck_proj(x_spatial) # (B, 256, 32, 32)
        else:
            # Table 1: w/o I-JEPA global branch (pure CNN bottleneck)
            bottleneck = self.cnn_bottleneck(c4_deepest) # (B, 256, 32, 32)
        
        # 3. Decoder Path
        if self.use_restormer:
            d1 = self.up0(bottleneck)
            d1 = torch.cat([c4, d1], dim=1)
            d1 = self.dec0(self.reduce0(d1))
            
            d2 = self.up1(d1)
            d2 = torch.cat([c3, d2], dim=1)
            d2 = self.dec1(self.reduce1(d2))
            
            d3 = self.up2(d2)
            d3 = torch.cat([c2, d3], dim=1)
            d3 = self.dec2(self.reduce2(d3))
            
            d4 = self.up3(d3)
            d4 = torch.cat([c1, d4], dim=1)
            d4 = self.dec3(self.reduce3(d4))
        else:
            d1 = self.conv_up0(torch.cat([c4, self.up0(bottleneck)], dim=1))
            d2 = self.conv_up1(torch.cat([c3, self.up1(d1)], dim=1))
            d3 = self.conv_up2(torch.cat([c2, self.up2(d2)], dim=1))
            d4 = self.conv_up3(torch.cat([c1, self.up3(d3)], dim=1))
            
        logits = self.outc(d4)
        return torch.sigmoid(logits)
