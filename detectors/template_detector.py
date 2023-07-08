import numpy as np
import cv2

from detectors.detector_base import DetectorBase


class TemplateDetector(DetectorBase):
    def __init__(self, threshold = 0.7, max_num = -100, method = cv2.TM_CCORR_NORMED):
        super(TemplateDetector, self).__init__("TemplateDetector")       
        
        if method in [cv2.TM_CCOEFF, cv2.TM_CCORR, cv2.TM_SQDIFF]:
            raise Exception("serch requires a normalized algorithm!")

        self.__method = method
        self.__threshold = threshold
        self.__max_num = max_num

    @property
    def threshold(self):
        return self.__threshold

    @threshold.setter
    def threshold(self, value):
        self.__threshold = value

    @property
    def pattern(self):
        return self.__pattern

    @pattern.setter
    def pattern(self, pat):
        self.__pattern = pat
        self.__height, self.__width = self.__pattern.shape[:2]

    @property
    def max_num(self):
        return self.__max_num

    @max_num.setter
    def max_num(self, value):
        self.__max_num = value

    def load(self, file):
        self.__pattern = cv2.imread(file)
        self.__height, self.__width = self.__pattern.shape[:2]


    def search(self, image : np.array, threshold : float = None):
        if image is None:
            raise Exception('Image is null!')

        if self.__method in [cv2.TM_CCOEFF, cv2.TM_CCORR, cv2.TM_SQDIFF]:
            raise Exception("serch requires a normalized algorithm!")

        num_channels = len(image.shape)
        if num_channels==2:
            self.__pattern = cv2.cvtColor(self.__pattern, cv2.COLOR_BGR2GRAY)

        if image.dtype.name == 'float32' and self.__pattern.dtype.name == 'uint8':
            self.__pattern = np.float32(self.__pattern) 
            self.__pattern = self.__pattern / 255.0

        res = cv2.matchTemplate(image, self.__pattern, self.__method)
        if self.__method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            res = 1 - res

        result_copy = res.copy()

        if threshold is None:
            threshold = self.__threshold

        img_height, img_width = image.shape[:2]

        max_val = 1
        rects = []

        ct = 0
        while max_val > threshold:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            ct += 1

            if ct > self.__max_num:
                break

            x, y = max_loc
            if max_val > threshold:
                h1 = np.clip(max_loc[1] - self.__height//2, 0, img_height)
                h2 = np.clip(max_loc[1] + self.__height//2 + 1, 0, img_height)

                w1 = np.clip(max_loc[0] - self.__width//2, 0, img_width)
                w2 = np.clip(max_loc[0] + self.__width//2 + 1, 0, img_width)
                res[h1:h2, w1:w2] = 0   

                # note: The size of the match image is smaller by the size of the pattern.
                # therefor pattern size/2 needs to be added.
                rects.append((int(x + self.__width//2), int(y + self.__height//2), self.__width, self.__height, max_val, 0))

        return np.array(rects), result_copy