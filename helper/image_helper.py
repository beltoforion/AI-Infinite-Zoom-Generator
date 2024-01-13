import numpy as np
import pathlib
import cv2
from typing import Union

from processor.processor_base import *


def crop_image(image, crop_size):
    """ Crop an image. 
     
        Crop is done centric.  
    """
    h, w = image.shape[:2]
    cx = w //2 
    cy = h //2
    
    start_x = cx - crop_size[0]//2
    start_y = cy - crop_size[1]//2
    
    end_x = start_x + crop_size[0]
    end_y = start_y + crop_size[1]
    
    cropped_image = image[start_y:end_y, start_x:end_x]
    
    return cropped_image


def draw_cross(image, center, size, color=(0, 255, 0), thickness=2):
    """
    Draw a cross on the given image at the specified center coordinates with the given size, color, and thickness.
    
    Parameters:
        image (numpy.ndarray): The image on which to draw the cross.
        center (tuple): The center coordinates of the cross in (x, y) format.
        size (int): The size of the cross.
        color (tuple): The color of the cross in BGR format. Default is green (0, 255, 0).
        thickness (int): The thickness of the cross lines. Default is 2.
    """
    x, y = center
    half_size = size // 2

    cv2.line(image, (x - half_size, y), (x + half_size, y), color, thickness)
    cv2.line(image, (x, y - half_size), (x, y + half_size), color, thickness)


def read_image(file : str, processor : Union[ProcessorBase, list] = None):
    """ Read an image in raw or jpeg.
        The image will be preprocesses with a list of processors.
    """

    ext = pathlib.Path(file).suffix
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
    pathlist = sorted(path.glob('*.*'))

    images = []

    for path in pathlist:
        if not path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            continue

        img_orig, _ = read_image(str(path), None)
        if img_orig is not None:
            images.append(img_orig)

    if len(images)>0:
        first_image_shape = images[0].shape
        if not all(image.shape == first_image_shape for image in images):
            raise Exception(f"Reading images failed because not all images in the folder have equal size! Expected image size is {first_image_shape[:2]}.")

    return np.array(images)


def create_radial_mask(h, w, inner_radius_fraction=0.4, outer_radius_fraction=1.0):
    center = (int(w / 2), int(h / 2))
    inner_radius = int(min(center[0], center[1]) * inner_radius_fraction)
    outer_radius = int(min(center[0], center[1]) * outer_radius_fraction)

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = np.zeros((h, w))
    mask[dist_from_center <= inner_radius] = 1.0
    mask[dist_from_center >= outer_radius] = 0.0

    transition = np.logical_and(dist_from_center > inner_radius, dist_from_center < outer_radius)
    mask[transition] = (outer_radius - dist_from_center[transition]) / (outer_radius - inner_radius)

    return mask

def overlay_images(background, foreground, position, relative_to='corner', opacity=1):
    # Get the dimensions of the foreground image
    fh, fw, _ = foreground.shape

    # Create an alpha mask of the same size as the foreground image
    mask = create_radial_mask(fh, fw, 0.4, 1 + opacity)

    # Convert foreground to float and normalize
    foreground = foreground.astype(float) / 255

    # Create a 4-channel image (RGB + alpha) for the foreground
    foreground_alpha = np.dstack([foreground, mask])

    # Get the position
    x, y = position

    # If position is relative to the center, adjust the position
    if relative_to == 'center':
        y = background.shape[0]//2 - fh//2 + y
        x = background.shape[1]//2 - fw//2 + x

    # Calculate the overlay region
    overlay_x_start = max(x, 0)
    overlay_y_start = max(y, 0)
    overlay_x_end = min(x+fw, background.shape[1])
    overlay_y_end = min(y+fh, background.shape[0])

    # Calculate the region of the foreground to be overlayed
    foreground_x_start = max(0, -x)
    foreground_y_start = max(0, -y)
    foreground_x_end = min(fw, overlay_x_end - x)
    foreground_y_end = min(fh, overlay_y_end - y)

    # Prepare the overlay with the correct opacity
    foreground_region = foreground_alpha[foreground_y_start:foreground_y_end, foreground_x_start:foreground_x_end]
    background_region = background[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end] / 255

    overlay = (foreground_region[..., :3] * foreground_region[..., 3:4] * opacity +
               background_region * (1 - foreground_region[..., 3:4] * opacity)) * 255

    # Overlay the appropriately sized and positioned region of the foreground onto the background
    background[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end] = overlay.astype(np.uint8)

    return background
