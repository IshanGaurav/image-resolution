"""
SRCNN (Super-Resolution Convolutional Neural Network) model.

Reference:  Dong et al., "Image Super-Resolution Using Deep Convolutional
            Networks", IEEE TPAMI 2016.

This implementation uses random weights by default (no pretrained checkpoint
is loaded). To use a trained model, call ``model.load_weights(path)``.
"""

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from .base_model import BaseModel


class SRCNNNet(nn.Module):
    """
    Three-layer CNN following the classic architecture:
        1. Patch extraction & representation  (9×9, 1 → 64)
        2. Non-linear mapping                (1×1 or 5x5, 64 → 32)
        3. Reconstruction                    (5×5, 32 → 1)

    Operates on a single-channel (luminance / Y) input.
    """

    def __init__(self, map_kernel: int = 1) -> None:
        super().__init__()
        map_pad = map_kernel // 2
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=map_kernel, padding=map_pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) float32 tensor, pixel values in [0, 1].
        Returns:
            (B, 1, H, W) tensor.
        """
        return self.layers(x)


class SRCNNModel(BaseModel):
    """
    Wrapper that makes :class:`SRCNNNet` usable through the common
    :class:`BaseModel` interface.

    Pipeline:
        1. Bicubic-upscale the LR image to the target resolution.
        2. Convert to YCbCr; extract the Y channel.
        3. Run the CNN on the Y channel.
        4. Recombine with Cb/Cr and convert back to RGB uint8.
    """

    def __init__(self, scale_factor: int = 4) -> None:
        self.scale_factor = scale_factor
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.net = SRCNNNet().to(self.device)
        self.net.eval()

    # ---- optional: load a trained checkpoint -------------------------
    def load_weights(self, path: str) -> None:
        """Load a ``.pth`` or ``.pth.tar`` state dict into the network."""
        import os
        if not os.path.isabs(path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base_dir, path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"No weights file found at {path}")

        state = torch.load(path, map_location=self.device)
        
        # Handle Lornatang or other implementations that wrap state_dict
        if 'state_dict' in state:
            state = state['state_dict']
            
        # Detect and map Lornatang's keys (features.0, map.0, reconstruction)
        if 'features.0.weight' in state:
            new_state = {}
            new_state['layers.0.weight'] = state['features.0.weight']
            new_state['layers.0.bias'] = state['features.0.bias']
            new_state['layers.2.weight'] = state['map.0.weight']
            new_state['layers.2.bias'] = state['map.0.bias']
            new_state['layers.4.weight'] = state['reconstruction.weight']
            new_state['layers.4.bias'] = state['reconstruction.bias']
            state = new_state
            
        # Auto-detect map_kernel if the loaded model uses 9-5-5 instead of 9-1-5
        if 'layers.2.weight' in state:
            map_kernel_size = state['layers.2.weight'].shape[-1]
            if map_kernel_size != self.net.layers[2].kernel_size[0]:
                self.net = SRCNNNet(map_kernel=map_kernel_size).to(self.device)

        self.net.load_state_dict(state)
        self.net.eval()

    # ---- BaseModel interface -----------------------------------------
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Super-resolve *image* by ``self.scale_factor``.

        Args:
            image: LR image, np.ndarray (H, W, 3) uint8 RGB.
        Returns:
            SR image, np.ndarray (H*scale, W*scale, 3) uint8 RGB.
        """
        # 1. Accept input image and determine target size
        h, w = image.shape[:2]
        target_size = (w * self.scale_factor, h * self.scale_factor)

        # 2. Convert to YCbCr using PIL
        pil_img = Image.fromarray(image, mode="RGB")
        pil_ycbcr = pil_img.convert("YCbCr")
        
        # Split into individual channels
        y, cb, cr = pil_ycbcr.split()

        # 3. Extract Y channel, bicubic-upscale it, and pass through SRCNN
        y_upscaled = y.resize(target_size, Image.BICUBIC)
        y_np = np.array(y_upscaled, dtype=np.float32) / 255.0  # normalize
        
        y_tensor = (
            torch.from_numpy(y_np)
            .unsqueeze(0)            # (1, H, W)
            .unsqueeze(0)            # (1, 1, H, W)
            .to(self.device)
        )

        with torch.no_grad():
            y_pred = self.net(y_tensor)  # (1, 1, H, W)

        # 5. Convert output Y tensor back to uint8 numpy array
        y_pred_np = y_pred.squeeze().cpu().numpy()  # (H, W)
        y_pred_np = np.clip(y_pred_np * 255.0, 0, 255).astype(np.uint8)
        y_out = Image.fromarray(y_pred_np, mode="L")

        # 4. Upscale Cb and Cr using standard Bicubic
        cb_out = cb.resize(target_size, Image.BICUBIC)
        cr_out = cr.resize(target_size, Image.BICUBIC)

        # 6. Merge the new Y channel with upscaled Cb and Cr channels
        merged_ycbcr = Image.merge("YCbCr", (y_out, cb_out, cr_out))

        # 7. Convert properly to RGB and return as NumPy array
        pil_out = merged_ycbcr.convert("RGB")
        return np.array(pil_out, dtype=np.uint8)
