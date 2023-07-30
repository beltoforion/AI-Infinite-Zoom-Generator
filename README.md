# Creating an "Inifinite Zoom" from AI-Outpainted images
The Python command line script published here can turn a series of AI-generated images into a zoom animation. For more details have a look at my web page were I explain the inner workings in detail:

* In German: https://beltoforion.de/de/infinite_zoom/index.php
* In English: https://beltoforion.de/en/infinite_zoom/index.php

Here is an example video created by the script:

https://github.com/beltoforion/ai_ever_zoom/assets/2202567/78bcbe99-8dbb-48d7-88bf-f8f400ed10c9

## What is AI Outpainting?
Outpainting is a technique where zoom out of an image by a certain factor while letting a generative AI 
fill in the newly created empty edge. By giving the AI new prompts, you can control the evolution of the scene 
as you zoom out.

![outpainting_example](https://github.com/beltoforion/ai_ever_zoom/assets/2202567/206d4f06-6a9b-4b9b-8377-131a319d2457)

AI outpainting requires the use of a generative AI for images and can be done with Midjourney, Dall-E or Photoshop 
(generative AI currently only in beta). I have only tested this command line script on Midjourney images as they are 
the easiest to create. In principle, this program will work with any outpainted image set.

## Preparing the images
Before you start you need a set of outpainted ai images. Copy this set in a separate folder in the "input" folder. It is best 
to order the images in the folder by giving them sequential names (i.e. "frame_01.png", "frame_02.png", ..., "frame_10.png").

Alternatively you can use the "-as" option to let the script find out the image order for you but this will take some time as
each image is matched against all other images to figure out their relations automatically.

* Create a set of outpainted images with the generative AI of your choice.
  + The first image is the innermost image of the series.
  + The Image series must be zoomed with respect to the center
  + The entire image series must use the same zoom factor (i.e. 2x)
* Rename and order the image sequence by giving them sequential names. (i.e. "frame_01.png", "frame_02.png")

## Usage

You need python to execute this script. Put your input images into a folder and then run the script on the content of this folder.

```python
python ./infinite_zoom.py -zf 2 -zs 100 -zc 0.8 -i ./samples_ps -o video.mp4
```
or an example to dump the frames without creating a video file:

```python
python3 ./infinite_zoom.py -as -i ./sample_fairytale -o myframes/
```


## Command Line Options

<b>-zf</b><br/> Zoom factor used for creating the outpinted image sequence. For image sequences created by Midjourney use either "2" or "1.33". (Midjourney incorrectly states that it low zoom level is 1.5 but it is actually just 1.33)
<br/><br/>
<b>-zs</b><br/> Number of zoom steps for each image
<br/><br/>
<b>-zc</b><br/> Crop zoomed images by this factor. Midjourney takes some liberties in modyfing the edge regions between zoom steps. They may not match perfectly.
<br/><br/>
<b>-i</b><br/> Path to folder with input images.
<br/><br/>
<b>-o</b><br/> Name of the output folder or file. Must either be a valid file name with an mp4 extension or a folder name. If no extension is given it is assumed to be a folder name and the output will consist of the frame dump instead of a single video file.
<br/><br/>
<b>-as</b><br/> Automatically sort input images. If you use this option you can name the images arbitrarily. The script will figure out the right order.
<br/><br/>
<b>-dbg</b><br/> Show debug overlays.
<br/><br/>
<b>-rev</b><br/> Reverse the video. This will create a zoom out effect.
<br/><br/>
<b>-fps</b><br/> Set the target framerate of the output video.





