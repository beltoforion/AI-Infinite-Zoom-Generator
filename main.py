import os
import infini_zoom as izoom
import argparse

from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='AI Outpainting Zoom Generator - Turn an AI generated image series into an animation')
    parser.add_argument('-zf', '--ZoomFactor', dest='zoom_factor', help='The outpainting zoom factor set up when creating the image sequence.', required=False, type=float, default=2)
    parser.add_argument('-zs', '--ZoomSteps', dest='zoom_steps', help='The number of zoom steps to be generated between two successive images.', required=False, type=int, default=100)
    parser.add_argument('-zc', '--ZoomCrop', dest='zoom_crop', help='Set the crop factor of each zoom steps followup image. This is helpfull to hide image varyations on the edges.', required=False, type=float, default=0.8)
    parser.add_argument('-o', '--Output', dest='output', help='Name of output file including mp4 extension.', required=False, type=str, default='output.mp4')
    parser.add_argument('-i', '--Input', dest='input_folder', help='Path to to folder containing input images.', required=True, type=str)
    parser.add_argument('-as', '--AutoSort', dest='auto_sort', help='Input images are unsorted, automatically sort them.', required=False, action='store_true', default=False)
    parser.add_argument('-rev', '--Reverse', dest='reverse', help='Reverse the output video.', required=False, action='store_true', default=False)
    
    args = parser.parse_args()

    print('\r\n')
    print('AI Outpainting Zoom Video Generator')
    print('-----------------------------------')
    print(f' - input folder: "{args.input_folder}"')
    print(f' - output file: "{args.output}"')
    print(f' - zoom factor: {args.zoom_factor}')
    print(f' - zoom steps: {args.zoom_steps}')
    print(f' - zoom crop: {args.zoom_crop}')
    print(f' - auto sort: {args.auto_sort}')
    print(f' - reverse: {args.reverse}')

    current_dir = os.path.dirname(os.path.abspath(__file__))

    param = izoom.InfiniZoomParameter()
    param.reverse = False
    param.auto_sort = args.auto_sort
    param.zoom_image_crop = args.zoom_crop
    param.zoom_steps = args.zoom_steps
    param.zoom_factor = args.zoom_factor  # The zoom factor used by midjourney
    param.input_path = Path(args.input_folder)
    param.output_file = args.output       # name of the output video file

    iz = izoom.InfiniZoom(param)
    iz.process()

if __name__ == "__main__":
    main()
