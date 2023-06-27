import os

import infini_zoom as izoom

from pathlib import Path


def infini_zoom():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, 'input/nostradamus')

    param = izoom.InfiniZoomParameter()
    param.zoom_steps = 100
    param.input_path = Path(folder_path)
    param.output_file = "video.mp4"

    iz = izoom.InfiniZoom(param)
    iz.load_images()
    iz.process()
    

def main():
    infini_zoom()

if __name__ == "__main__":
    main()
