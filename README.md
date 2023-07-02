# Generating Zoom Animations from an AI Outpainting image series
This python program can turn a series of AI generated images into a zoom animation. It is taking an image sequence as input that was created
with a generative AI using a constant zoom factor. You have probably already seen outpainted ai zoom sequences as their base images are now 
quite easy to create since [Midjourney introduces the zoom out feature](https://docs.midjourney.com/docs/zoom-out) in Version 5.

https://github.com/beltoforion/ai_ever_zoom/assets/2202567/78bcbe99-8dbb-48d7-88bf-f8f400ed10c9

## What is AI Outpainting?
Outpainting is a technique where you create an image with a generative AI and then zoom it out by a certain factor while letting the AI 
fill in the newly created empty edge. By also changing the prompt, you can create scene transitions as you zoom out.

![outpainting_example](https://github.com/beltoforion/ai_ever_zoom/assets/2202567/206d4f06-6a9b-4b9b-8377-131a319d2457)

AI outpainting requires the use of a generative AI for images and can be done with Midjourney, Dall-E or Photoshop 
(generative AI currently only in beta). I have only tested this command line script on Midjourney images as they are 
the easiest to create. In principle, this program will work with any outpainted image set.

## Preparing the images
Before you start you need a set of outpainted ai images. Copy this set in a separate folder in the "input" folder.

* Create a set of outpainted images with the generative AI of your choice. The first image is the innermost image of the series.
* The Image series must be zoomed with respect to the center
* The entire image series must use the same zoom factor (i.e. 2x)


