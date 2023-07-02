# Generating Zoom Animations from AI Outpainting
This python program can turn a bunch of AI generated images into a zoom animation. Currently it is expecting Midjourney images as input that were
created using the 2x zoom option. 

Please note: This is project a work in progress!

https://github.com/beltoforion/ai_ever_zoom/assets/2202567/78bcbe99-8dbb-48d7-88bf-f8f400ed10c9

## What is AI Outpainting?
Before you can use this script you have to create a set outpainted ai images. Outpainting is a technique were you
create an image with a generative AI and then you zoom this image out by a given zoom factor and let the AI fill
the newly created empty border. By also changing the prompt you can create scene transitions whilst zooming out.

AI outpainting requires the use of a generative AI for images and can be done with Midjourney, by Dall-E or with 
Photoshop (generative AI currently only in the beta version). I only ever tested this command line script on midjourney 
images since they are easiest to create. In principle this program will work with any outpainted image set.

## Preparing the images
* Create a set of outpainted images with the generative AI of your choice.
* The Image series must be zoomed with respect to the center
* The entire image series must use the same zoom factor (i.e. 2x)

