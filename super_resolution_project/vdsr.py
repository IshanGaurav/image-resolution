import torch.nn as nn

# The original twtygqyy checkpoint serialized internal class references natively.
# We recreate the exact empty husks of these classes so the Python unpickler
# can successfully reconstruct the object graph before diving into state_dict().

class Conv_ReLU_Block(nn.Module):
    pass

class Net(nn.Module):
    pass

