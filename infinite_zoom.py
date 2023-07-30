import infinite_zoom_impl as izoom
import argparse
import pathlib


from pathlib import Path

def valid_crop_range(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{x} is not a floating-point literal')

    if x < 0.1 or x > 0.95:
        raise argparse.ArgumentTypeError(f'{x:0.2f} not in range [0.1, 0.95]')

    return x


def main():
    parser = argparse.ArgumentParser(description='AI Outpainting Zoom Generator - Turn an AI generated image series into an animation')
    parser.add_argument('-zf', '--ZoomFactor', dest='zoom_factor', help='The outpainting zoom factor set up when creating the image sequence.', required=False, type=float, default=2)
    parser.add_argument('-zs', '--ZoomSteps', dest='zoom_steps', help='The number of zoom steps to be generated between two successive images.', required=False, type=int, default=100)
    parser.add_argument('-zc', '--ZoomCrop', dest='zoom_crop', help='Set the crop factor of each zoom steps followup image. This is helpfull to hide image varyations on the edges.', required=False, type=valid_crop_range, default=0.8)
    parser.add_argument('-o', '--Output', dest='output', help='Name of output file or folder. If this is a folder name the output will consist of the frames. If it is not a folder name it is assumed to be the name of the mp4 output file.', required=False, type=str, default='output.mp4')
    parser.add_argument('-i', '--Input', dest='input_folder', help='Path to to folder containing input images.', required=True, type=str)
    parser.add_argument('-as', '--AutoSort', dest='auto_sort', help='Input images are unsorted, automatically sort them.', required=False, action='store_true', default=False)
    parser.add_argument('-dbg', '--Debug', dest='debug', help='Enable debug aides', required=False, action='store_true', default=False)    
    parser.add_argument('-d', '--Delay', dest='delay', help='Start/Stop delay in seconds', required=False, type=float, default=0.0)    
    parser.add_argument('-rev', '--Reverse', dest='reverse', help='Reverse the output video.', required=False, action='store_true', default=False)
    parser.add_argument('-fps', '--FramesPerSecond', dest='fps', help='Frames per second in the output video.', required=False, type=float, default=60.0)    
    
    args = parser.parse_args()

    print('\r\n')
    print('AI Outpainting Zoom Video Generator')
    print('-----------------------------------')
    print(f' - input folder: "{args.input_folder}"')
    print(f' - output: "{args.output}"')
    print(f' - fps: {args.fps}')
    print(f' - zoom factor: {args.zoom_factor}')
    print(f' - zoom steps: {args.zoom_steps}')
    print(f' - zoom crop: {args.zoom_crop}')
    print(f' - auto sort: {args.auto_sort}')
    print(f' - debug aides: {args.debug}')    
    print(f' - delay: {args.delay}')        
    print(f' - reverse: {args.reverse}')

    param = izoom.InfiniZoomParameter()
    param.reverse = args.reverse
    param.auto_sort = args.auto_sort
    param.debug_mode = args.debug
    param.zoom_image_crop = args.zoom_crop
    param.zoom_steps = args.zoom_steps
    param.zoom_factor = args.zoom_factor  # The zoom factor used by midjourney
    param.input_path = Path(args.input_folder)
    param.delay = args.delay
    param.fps = args.fps

    # when no extension is present we assume it is an output folder. In this
    # case no video is created and all frames are saved into the output folder
    output_ext = pathlib.Path(args.output).suffix
    if output_ext == '':
        if not pathlib.Path(args.output).is_dir():
            print(f"\nCreating frame output folder: {args.output}")
            pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
            
        param.output_frames = True
        param.output_file = None
        param.output_folder = args.output
    else:
        param.output_frames = False
        param.output_file = args.output       # name of the output video file
        param.output_folder = None

    iz = izoom.InfiniZoom(param)
    iz.process()

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(f'\nError: {ex}')
