# PlateApi-Pub

An ALPR for Persian plates. Using YOLO-v5 and a custom dataset consisting of 1000+ images of different license plates, including special plates.

## General Info

The detection is done in two steps; first, detecting the license plate in the input image and then identifying characters on the plate. Once the plate has been detected in the input image, the part of the image that matches the bounding box is cropped. The cropped portion is then sent to the second network for character detection. Both networks are YOLO-based.

## Steps of the Project

1. Collecting images (Including special plate colors and characters)

2. Labeling images for the first part

	- Using [LabelImg](https://github.com/heartexlabs/labelImg)

	- Drawing a box around the plates in each image

	- Dataset-1

3. Cropping specified plates from images

4. Labeling characters from cropped plate images

	- Using [LabelImg](https://github.com/heartexlabs/labelImg)

	- Dataset-2

5. Increase the number of images using various methods of image augmentation

	- Using [YOLO-Image-Augmentation](https://github.com/RastinS/YOLO-Image-Augmentation)

	- Increasing the number of images by a factor of 16

	- On both Dataset-1 and Dataset-2

6. Training two independent networks on Dataset-1 and Dataset-2

7. Implementing a pipeline using Python with a single image as input and a list of plate characters as output

8. Implementing frame-fetching and detection of video sources

9. Django API implementation using the Django framework

10. Making the project Docker-based so that it can be easily installed and maintained

