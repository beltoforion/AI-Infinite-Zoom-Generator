from pathlib import Path
from exceptions.argument_exception import ArgumentException
from detectors.template_detector import *

import helper.image_helper as ih
import cv2
import math


class InfiniZoomParameter:
    def __init__(self):
        self.__zoom_steps = 100
        self.__reverse = False
        self.__auto_sort = False
        self.__zoom_image_crop = 0.8

    @property
    def zoom_image_crop(self):
        return self.__zoom_image_crop
    
    @zoom_image_crop.setter
    def zoom_image_crop(self, crop: float):
        self.__zoom_image_crop = crop

    @property
    def reverse(self):
        return self.__reverse
    
    @reverse.setter
    def reverse(self, stat: bool):
        self.__reverse = stat

    @property
    def auto_sort(self):
        return self.__auto_sort
    
    @auto_sort.setter
    def auto_sort(self, stat: bool):
        self.__auto_sort = stat

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

    def __load_images(self):
        if not self.__param.input_path.exists():
            raise Exception("input path does not exist")
        
        print(f'Reading images form "{str(self.__param.input_path)}"')
        self.__image_list = ih.read_images_folder(self.__param.input_path)

    def __auto_sort(self):
        print(f'Figuring out image order.')

        for i in range(0, len(self.__image_list)):
            for j in range(0, len(self.__image_list)):
                if i>=j:
                    continue
                
                img1 = self.__image_list[i].copy()
                img2 = self.__image_list[j].copy()
                if img1.shape != img2.shape:
                    raise Exception("Auto sort failed: Inconsistent image sizes!")

                h, w = img1.shape[:2]
                w = int(w * self.__param.zoom_image_crop)
                h = int(h * self.__param.zoom_image_crop)
                img2 = ih.crop_image(img2, (w, h))

                print(f'Correlating image {i} with image {j}.')


    def process(self):
        self.__load_images()

        if len(self.__image_list)==0:
            raise Exception("Processing failed: Image list is empty!")

        if self.__param.auto_sort:
            self.__auto_sort()

        h, w = self.__image_list[0].shape[:2]

        video_w = int(w * self.__param.zoom_image_crop)
        video_h = int(h * self.__param.zoom_image_crop)
        self.__video_writer = cv2.VideoWriter(self.__param.output_file, cv2.VideoWriter_fourcc(*'mp4v'), 60, (video_w, video_h))

        for i in range(len(self.__image_list)-1):
            img1 = self.__image_list[i]
            img2 = self.__image_list[i+1]

            self.zoom_in(self.__video_writer, img1, img2, video_w, video_h)

        cv2.destroyAllWindows()
        self.__video_writer.release()


    def zoom_in(self, video_writer, imgCurr, imgNext, video_w, video_h):
        steps = self.__param.zoom_steps

        # imgNext has exactly a quarter the size of img
        h, w = imgCurr.shape[:2]
        cx = w // 2
        cy = h // 2

        step_size_log = math.exp(math.log(2)/steps)

        zf = 1
        img_curr = imgCurr.copy()
        img_next = imgNext.copy()

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
            ww = int(ww*0.8)
            hh = int(hh*0.8)
            img_next = ih.crop_image(img_next, (ww, hh))

            if i==0:
                # The second image may not be perfectly centered. We need to determine 
                # image offset to compensate
                detector = TemplateDetector(threshold=0.3, max_num=1)
                detector.pattern = img_next
                result = detector.search(img_curr)
                if len(result)==0:
                    raise Exception("Cannot match image to precursor!")

                bx, by, bw, bh, score = result[0, :5]
            
                # compute initial misalignment of the second image. Ideally the second image 
                # should be centered. We have to gradually eliminate the misalignment in the
                # successive zoom steps so that it is zero when switching to the next image.
                ma_x = int(cx - bx)
                ma_y = int(cy - by)
                print(f'image misalignment: x={ma_x:.2f}; y={ma_y:.2f}')
                
                # How much do we need to compensate for each step?
                ma_comp_x = ma_x / steps
                ma_comp_y = ma_y / steps

            # Add the smaller image into the larger one but shift it to 
            # compensate the misalignment. Problem is that when it is maximized
            # it would be shifted off center. We need to fix that later.
            hs = hh//2 + int(ma_y)
            ws = ww//2 + int(ma_x)
            img_curr[cy-hs:cy-hs+hh, cx-ws:cx-ws+ww] = img_next

            #opacity = max(0, min(zf-1, 1))
            #img_curr = ih.overlay_images(img_curr, img_next, (int(-ma_x), int(-ma_y)), 'center', opacity)

            # finally we have to gradually shift the resulting image back because the
            # next frame should again be close to the center and the misalignment compensation
            # brought us away. So we gradually shift the combined image back so that the center
            # position remains in the center.
            ox = ma_comp_x * i
            oy = ma_comp_y * i
            mtx_shift = np.float32([[1, 0, ox], [0, 1, oy]])
            img_curr = cv2.warpAffine(img_curr, mtx_shift, (img_curr.shape[1], img_curr.shape[0]))

#            cv2.imshow("Image", img_curr)
#                cv2.waitKey()

#                cv2.imshow("Image", img_next)
#                cv2.waitKey()

#            cv2.rectangle(img_curr, (int(bx-bw//2), int(by-bh//2)), (int(bx+bw//2), int(by+bh//2)), (0,0,255), 1)

            #print(f'bx={bx:.2f}; by={by:.2f}; score={score:.2f}; zoom={zf}; s1={ww}x{hh}; s2={h}x{w}')

            # final crop, ther may be some inconsiostencies at the boundaries
            img_curr = ih.crop_image(img_curr, (video_w, video_h))

            video_writer.write(img_curr)
            cv2.imshow("Image", img_curr)
            cv2.waitKey(10)



