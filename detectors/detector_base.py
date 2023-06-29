from abc import ABC, abstractmethod


class DetectorBase(ABC):
    def __init__(self, name):
        self.__name = name

    @property
    def name(self):
        return self.__name
   
    @abstractmethod
    def search(self, file):
        pass