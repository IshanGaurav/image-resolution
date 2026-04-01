"""
VDSR (Very Deep Super-Resolution) model.

Reference:  Kim et al., "Accurate Image Super-Resolution Using Very Deep
            Convolutional Networks", CVPR 2016.

This implementation uses a 20-layer deep network with residual learning.
To use a trained model, call ``model.load_weights(path)``.
"""

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from .base_model import BaseModel


class VDSRNet(nn.Module):
    """
    20-layer CNN following the VDSR architecture:
        1. Input layer: Conv(3x3, 1 -> 64) + ReLU
        2. Intermediate layers: 18 layers of Conv(3x3, 64 -> 64) + ReLU
        3. Output layer: Conv(3x3, 64 -> 1)
        4. Global residual connection: output = input + network_output
    """

    def __init__(self, num_layers: int = 20) -> None:
        super().__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
            
        # Output layer
        layers.append(nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) float32 tensor, pixel values in [0, 1].
        Returns:
            (B, 1, H, W) tensor.
        """
        residual = x
        out = self.layers(x)
        return torch.add(out, residual)


class VDSRModel(BaseModel):
    """
    Wrapper that makes :class:`VDSRNet` usable through the common
    :class:`BaseModel` interface.

    Pipeline:
        1. Bicubic-upscale the LR image to the target resolution.
        2. Convert to YCbCr; extract the Y channel.
        3. Run the VDSR on the Y channel (learning the residual).
        4. Recombine with Cb/Cr and convert back to RGB uint8.
    """

    def __init__(self, scale_factor: int = 4) -> None:
        self.scale_factor = scale_factor
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.net = VDSRNet().to(self.device)
        self.net.eval()

    def load_weights(self, path: str) -> None:
        """Load a ``.pth`` or ``.pt`` state dict into the network."""
        import os
        import sys

        if not os.path.isabs(path):
            # If relative, look relative to the script's directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base_dir, path)
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"No weights file found at {path}")

        # PyTorch 2.6 blocks globals dynamically and this older checkpoint requires 'vdsr.Net'
        # We must explicitly alias the current module (models.vdsr) to 'vdsr' in sys.modules
        if 'vdsr' not in sys.modules:
            sys.modules['vdsr'] = sys.modules[__name__]
        if not hasattr(sys.modules['vdsr'], 'Net'):
            setattr(sys.modules['vdsr'], 'Net', VDSRNet)
            
        # Register safe global for PyTorch 2.6+ pickler
        if hasattr(torch.serialization, 'add_safe_globals'):
            try:
                torch.serialization.add_safe_globals([sys.modules['vdsr'].Net])
            except Exception:
                pass

        state = torch.load(path, map_location=self.device, weights_only=False)

        
        # Determine if it's a full model object or a nested state dict
        if hasattr(state, 'state_dict') and callable(getattr(state, 'state_dict')):
            state = state.state_dict()
        elif isinstance(state, dict):
            if 'state_dict' in state:
                state = state['state_dict']
            elif 'model' in state:
                state = state['model']
        elif hasattr(state, '__dict__'):
            state = state.__dict__
        
        # Map keys if necessary (e.g., if the checkpoint was trained with DataParallel)
        new_state = {}
        for k, v in state.items():
            name = k.replace('module.', '') # remove `module.` prefix
            
            # Map twtygqyy's implementation keys to our Sequential layers
            if name == 'input.weight':
                name = 'layers.0.weight'
            elif name == 'output.weight':
                name = 'layers.38.weight'
            elif name.startswith('residual_layer.'):
                # residual_layer.i.conv.weight -> layers.2*(i+1).weight
                parts = name.split('.')
                idx = int(parts[1])
                param_type = parts[3] # weight or bias
                new_idx = 2 * (idx + 1)
                name = f'layers.{new_idx}.{param_type}'
            
            new_state[name] = v
            
        self.net.load_state_dict(new_state)
        self.net.eval()

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

        # 3. Extract Y channel, bicubic-upscale it
        y_upscaled = y.resize(target_size, Image.BICUBIC)
        y_np = np.array(y_upscaled, dtype=np.float32) / 255.0  # normalize
        
        y_tensor = (
            torch.from_numpy(y_np)
            .unsqueeze(0)            # (1, H, W)
            .unsqueeze(0)            # (1, 1, H, W)
            .to(self.device)
        )

        # 4. Run through VDSR
        with torch.no_grad():
            y_pred = self.net(y_tensor)  # (1, 1, H, W)

        # 5. Convert output Y tensor back to uint8 numpy array
        y_pred_np = y_pred.squeeze().cpu().numpy()  # (H, W)
        y_pred_np = np.clip(y_pred_np * 255.0, 0, 255).astype(np.uint8)
        y_out = Image.fromarray(y_pred_np, mode="L")

        # 6. Upscale Cb and Cr using standard Bicubic
        cb_out = cb.resize(target_size, Image.BICUBIC)
        cr_out = cr.resize(target_size, Image.BICUBIC)

        # 7. Merge the new Y channel with upscaled Cb and Cr channels
        merged_ycbcr = Image.merge("YCbCr", (y_out, cb_out, cr_out))

        # 8. Convert properly to RGB and return as NumPy array
        pil_out = merged_ycbcr.convert("RGB")
        return np.array(pil_out, dtype=np.uint8)
