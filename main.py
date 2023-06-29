import os
import infini_zoom as izoom

from pathlib import Path


def infini_zoom():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, 'input/pirat')
    #folder_path = os.path.join(current_dir, 'input/street')

    param = izoom.InfiniZoomParameter()
    param.reverse = False
    param.auto_sort = True
    param.zoom_image_crop = 0.8
    param.zoom_steps = 100
    param.input_path = Path(folder_path)
    param.output_file = "video2.mp4"

    iz = izoom.InfiniZoom(param)
    iz.process()


def main():
    infini_zoom()

if __name__ == "__main__":
    main()
