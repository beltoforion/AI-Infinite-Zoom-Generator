import numpy as np
import pathlib
import rawpy
import cv2
from typing import Union

from processor.processor_base import *


def crop_image(image, new_size):
    """ Crop an image. 
     
        Crop is done centric.  
    """
    h, w = image.shape[:2]
    cx = w //2 
    cy = h //2
    
    start_x = cx - new_size[0]//2
    start_y = cy - new_size[1]//2
    end_x = start_x + new_size[0]
    end_y = start_y + new_size[1]
    cropped_image = image[start_y:end_y, start_x:end_x]
    
    return cropped_image


def read_image(file : str, processor : Union[ProcessorBase, list] = None):
    """ Read an image in raw or jpeg.
        The image will be preprocesses with a list of processors.
    """

    ext = pathlib.Path(file).suffix
    if ext.lower()=='.cr2':
        image : np.array = rawpy.imread(file).postprocess(output_bps=16) 
        image = np.float32(image) # image.astype(np.float32)
        image = image / 65535.0
    else:
        image : np.array = cv2.imread(file)

    original_image = image.copy()

    if type(processor) is list:
        for p in processor:
            image = p.process(image)
    elif isinstance(processor, ProcessorBase):
        image = processor.process(image)
    elif processor is None:
        pass

        return image, original_image


def read_images_folder(path : pathlib.Path):
    pathlist = path.glob('**/*.*')

    images = []

    for path in pathlist:
        if not path.suffix.lower() in ['.png', '.jpg']:
            continue

        img_orig, _ = read_image(str(path), None)
        if img_orig is not None:
            images.append(img_orig)

    if len(images)>0:
        first_image_shape = images[0].shape
        if not all(image.shape == first_image_shape for image in images):
            raise Exception("read_images_folder: inconsistent image sizes")

    return np.array(images)