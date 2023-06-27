import os
import cv2
import numpy as np
import math

def read_images_to_array(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            if img is not None:
                images.append(img)
    return np.array(images)


def crop_image(image, new_size):
    h, w = image.shape[:2]
    cx = w //2 
    cy = h //2
    
    start_x = cx - new_size[0]//2
    start_y = cy - new_size[1]//2
    end_x = start_x + new_size[0]
    end_y = start_y + new_size[1]
    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image

def zoom_in(video_writer, imgCurr, imgNext, steps):
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
        ww = int(ww*0.6)
        hh = int(hh*0.6)
        img_next = crop_image(img_next, (ww, hh))

        hs = hh//2
        ws = ww//2
        img_curr[cy-hs:cy-hs+hh, cx-ws:cx-ws+ww] = img_next
#        print(f'zoom={zf}; s1={ww}x{hh}; s2={h}x{w}\r\n')

        video_writer.write(img_curr)
        cv2.imshow("Image", img_curr)
        cv2.waitKey(50)


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, 'input/nostradamus')

    # Replace 'folder_path' with the path to your folder containing PNG files
    images = read_images_to_array(folder_path)

    h, w = images[0].shape[:2]
    video_writer = cv2.VideoWriter("video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 60, (w, h))

    for i in range(len(images)-1):
        imgBig = images[i]
        imgSmall = images[i+1]

        zoom_in(video_writer, imgBig, imgSmall, 70)
    
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
