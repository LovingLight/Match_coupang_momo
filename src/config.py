import torch
import os
import random
import numpy as np

def setTorchSeed(seed=0):
    # Set seeds for reproducibility across multiple runs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Ensure consistent behavior in PyTorch (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

# Seed for initializing randomness
seed = 2024
setTorchSeed(seed)

# Determine if CUDA is available and set device accordingly
string_device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(string_device)

# parameters
max_length = 128
doc_stride = 64
hidden = 768
pad_on_right = True
ignore_index = -100