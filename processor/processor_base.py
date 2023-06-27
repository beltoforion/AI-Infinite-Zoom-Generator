from abc import ABC, abstractmethod
import numpy as np


class ProcessorBase(ABC):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name
        
    @abstractmethod
    def process(image : np.array) -> np.array:
        pass