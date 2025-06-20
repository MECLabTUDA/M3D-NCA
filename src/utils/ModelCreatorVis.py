from typing import Type, Dict, Any, TYPE_CHECKING, List, Union, TypeVar, Optional, Callable
from torch.utils.data import Dataset
import torch
import torchio
import numpy as np

import torch.nn as nn
import os
from os.path import join
import copy
"""
Annotation for typechecking; mostly deprecated.
"""

class Extended_Visualization_Model_Annotations(nn.Module):
    if TYPE_CHECKING:
        def __init__(self, channel_n, fire_rate, device, instrumentation_function: Callable[[np.ndarray, int], bool] = None, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, groups=False):
            
            pass
            
        def set_state_dict(self, dict: Dict[int, np.ndarray]):
            pass

        def export_state_dict(self) -> Dict[int, np.ndarray]:
            pass
        
        def set_instrumentation_function(self, function: Callable[[np.ndarray, int], bool]):
            pass

        def forward(self, x, steps=10, fire_rate=0.5):
            r"""Forward function applies update function step times leaving input channels unchanged
                #Args:
                    x: image
                    steps: number of steps to run update
                    fire_rate: random activation rate of each cell
            """
            pass






     
