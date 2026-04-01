# This file exists solely to satisfy PyTorch's pickle unpickler for the legacy VDSR checkpoint.
# The checkpoint was originally saved with a global reference to `vdsr.Net`.
# By putting this in the root (which is in sys.path during Streamlit execution),
# Python's `__import__('vdsr')` succeeds naturally and maps precisely to the real implementation.

from models.vdsr import VDSRNet as Net
