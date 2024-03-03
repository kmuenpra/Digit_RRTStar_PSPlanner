from abc import ABCMeta, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseModel(metaclass=ABCMeta):

    @abstractmethod
    def add_data(self, x_new: np.ndarray, y_new: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def optimize(self, *args, **kwargs) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
