from src.models.Model_BasicNCA3D import BasicNCA3D
import torch
import numpy as np
from typing import Callable, Dict

from abc import ABC
"""
Abstract interface for model, largely deprecated
"""
class VisualizationModel(ABC):
    def set_state_dict(self, dict: Dict[int, np.ndarray]):
        raise NotImplementedError()

    def export_state_dict(self) -> Dict[int, np.ndarray]:
        raise NotImplementedError()
    
    def set_instrumentation_function(self, function: Callable[[np.ndarray, int], bool]):
        raise NotImplementedError()
    