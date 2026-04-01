import torch
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from super_resolution_project.models.vdsr import VDSRModel

def test_loading():
    weights_path = os.path.join("super_resolution_project", "weights", "model_epoch_50.pth")
    
    print(f"Attempting to load weights from: {weights_path}")
    
    if not os.path.exists(weights_path):
        print("Error: Weights file not found!")
        return

    try:
        model = VDSRModel(scale_factor=4)
        model.load_weights(weights_path)
        print("Success: VDSRModel initialized and weights loaded correctly!")
    except Exception as e:
        print(f"Error loading weights: {e}")

if __name__ == "__main__":
    test_loading()
