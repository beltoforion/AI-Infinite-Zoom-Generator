import cv2

from abc import ABC, abstractmethod


class DetectorBase(ABC):
    def __init__(self, name):
        self.__name = name
        self.__pattern = None

    def load(self, file):
        self.__pattern = cv2.imread(file)
        self.__width, self.__height, _ = self.__pattern.shape
        self.after_load(file)

    @property
    def name(self):
        return self.__name

    @property
    def image(self):
        return self.__pattern

    @abstractmethod
    def after_load(self, file):
        pass

    @abstractmethod
    def search(self, file):
        pass