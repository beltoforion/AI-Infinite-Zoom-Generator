import os
import infini_zoom as izoom
import argparse

from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='AI Outpainting Zoom Generator - Turn an AI generated image series into an animation')
    parser.add_argument("-zf", "--ZoomFactor", dest="zoom_factor", help='The outpainting zoom factor set up when creating the image sequence.', required=False, type=float, default=2)
    parser.add_argument("-zs", "--ZoomSteps", dest="zoom_steps", help='The number of zoom steps to be generated between two successive images.', required=False, type=float, default=100)
    parser.add_argument("-zc", "--ZoomCrop", dest="zoom_crop", help='Set the crop factor of each zoom steps followup image. This is helpfull to hide image varyations on the edges.', required=False, type=float, default=0.8)
    parser.add_argument("-o", dest="output", help='Name of output file including mp4 extension.', required=False, type=str, default='output.mp4')
#    parser.add_argument("-tmpl", "--Template", dest="template", help='A template with a file where the created HTML fragment shall be inserted', required=True, type=str)    
#    parser.add_argument("-v", dest="verbose", action='store_true', help='Output prompts', required=False)    
#    parser.add_argument("-s", dest="style", help='Writing Style', required=False)        
    args = parser.parse_args()

    print('\r\n')
    print('AI Outpainting Zoom Generator')
    print('----------------------------------')
    print(f' - zoom factor: {args.zoom_factor}')
    print(f' - number of zoom steps: {args.zoom_steps}')    
    print(f' - zoom crop: {args.zoom_crop}')    
    print(f' - output file: {args.output}')    

    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, 'input/pirat')
    #folder_path = os.path.join(current_dir, 'input/street')

    param = izoom.InfiniZoomParameter()
    param.reverse = False
    param.auto_sort = True
    param.zoom_image_crop = args.zoom_crop
    param.zoom_steps = args.zoom_steps
    param.zoom_factor = args.zoom_factor               # The zoom factor used by midjourney
    param.input_path = Path(folder_path)
    param.output_file = args.output    # name of the output video file

    iz = izoom.InfiniZoom(param)
    iz.process()

if __name__ == "__main__":
    main()
