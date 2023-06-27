from pathlib import Path
from exceptions.argument_exception import ArgumentException

from detectors.template_detector import *

import helper.image_helper as ih

import cv2
import math

class InfiniZoomParameter:
    def __init__(self):
        self.__zoom_steps = 100

    @property
    def zoom_steps(self):
        return self.__zoom_steps
    
    @zoom_steps.setter
    def zoom_steps(self, steps: int):
        if steps<1:
            raise ArgumentException("Range error: steps must be greater than 0")
        
        self.__zoom_steps = steps

    @property
    def input_path(self):
        return self.__input_path
    
    @input_path.setter
    def input_path(self, path : Path):
        self.__input_path = path        

    @property
    def output_file(self):
        return self.__output_file
    
    @output_file.setter
    def output_file(self, file):
        self.__output_file = file


class InfiniZoom:
    def __init__(self, param : InfiniZoomParameter):
        self.__param = param
        self.__image_list = []
        self.__video_writer = None
#        self.__width = 0
#        self.__height = 0

    def load_images(self):
        if not self.__param.input_path.exists():
            raise Exception("input path does not exist")
        
        self.__image_list = ih.read_images_folder(self.__param.input_path)

    def process(self):
        if len(self.__image_list)==0:
            raise Exception("process failed: Image list is empty!")
        
        h, w = self.__image_list[0].shape[:2]

        self.__video_writer = cv2.VideoWriter("video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 60, (w, h))

        for i in range(len(self.__image_list)-1):
            img1 = self.__image_list[i]
            img2 = self.__image_list[i+1]

            self.zoom_in(self.__video_writer, img1, img2)

        self.__video_writer.release()


    def zoom_in(self, video_writer, imgCurr, imgNext):
        steps = self.__param.zoom_steps

        # imgNext has exactly a quarter the size of img
        h, w = imgCurr.shape[:2]
        cx = w // 2
        cy = h // 2

        step_size_log = math.exp(math.log(2)/steps)

        zf = 1
        img_curr = imgCurr.copy()
        img_next = imgNext.copy()
#        cv2.imshow("Image", img_curr)
#        cv2.waitKey()

#        cv2.imshow("Image", img_next)
#        cv2.waitKey()

        for i in range(0, steps):
            zf = step_size_log**i
            
            mtx_curr = cv2.getRotationMatrix2D((cx, cy), 0, zf)
            img_curr = cv2.warpAffine(imgCurr, mtx_curr, (w, h))

            ww = round(w * (zf/2.0))
            hh = round(h * (zf/2.0))
            mtx_next = cv2.getRotationMatrix2D((cx, cy), 0, zf/2.0)

            img_next = imgNext.copy()
            img_next = cv2.warpAffine(img_next, mtx_next, (w, h))

            # Before copying throw away 25% of the outter image
            ww = int(ww*0.6)
            hh = int(hh*0.6)
            img_next = ih.crop_image(img_next, (ww, hh))

            # The second image may not be perfectly centered. We need to determine 
            # image offset to compensate
            if i == 0:
                detector = TemplateDetector(threshold=0.1, max_num=1)
                detector.load('./images/pattern4.png')  

            hs = hh//2
            ws = ww//2
            img_curr[cy-hs:cy-hs+hh, cx-ws:cx-ws+ww] = img_next
    #        print(f'zoom={zf}; s1={ww}x{hh}; s2={h}x{w}\r\n')

            video_writer.write(img_curr)
            cv2.imshow("Image", img_curr)
            cv2.waitKey(50)

        cv2.destroyAllWindows()



