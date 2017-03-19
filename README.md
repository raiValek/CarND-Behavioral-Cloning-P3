# CarND-Behavioral-Cloning-P3
Udacity Self-Driving Car Nanodegree - Project 3 - Behavioral Cloning

## Introduction
The objective of this project is to train a Convolutional Neural Network (CNN) in such a way, that it is able to drive a car on a test track autonomously. The training data can be obtained by driving the car manually on the test track and recording simultaneously images from three cameras on the front of the car as well as telemetry data like throttle, speed and the steering angle. For this purpose Udacity provides a simulator capable of recording training data and receiving data for autonomous driving.

In this project we will focus on an end-to-end approach where the CNN will compute steering angles directly from the images of the center camera, without the use of the other cameras nor other telemetry data. However this data can be used for training purposes. A similar approach was used by NVIDIA in this article.

## Collecting Training Data
The simulator gives the option to drive the car by keyboard, mouse or game pad input. After some test laps with the keyboard, it became apparent that it is pretty hard to gather accurate data with this method. Since it is only possible to give binary commands with the arrow keys, the steering angles in a curve appeared to be scattered while it should rise and fall smoothly. Therefore I have used the analog stick of a Xbox 360 controller. High quality training data with good driving habits is obviously crucial for good training results, since the CNN will try to copy the behavior as good as it can.

In this project good driving habits are represented by staying always at the center of the road. However, if there are no errors in the data, the CNN does not ever learn how to recover from bad situations, like driving to close to the edge. I have to address this by driving close to the edge on purpose, but only recording the moment of recovery. That worked surprisingly well at slow speed (10 mph). The model was able to drive the entire track and stay centered autonomously. Unfortunately, with rising speed it became very unstable and the car left the track.

## Data preprocessing
### Shifting Images
Inspired by this [blog post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.xgnblq2nv) I have tried an different approach. Instead of recording recovery maneuvers manually, it is probably better to augment the good data to create the wished behavior. To achieve this all images will be shifted with equal probability to the or to the right in a range of plus and minus 50 pixels. For every shift the steering in angle will be increased and decreased accordingly with a value of 0.4 at the maximum shift of 50 pixels.

img		corr
SHIFT1.jpg =>	-0.1673
SHIFT2.jpg =>	-0.2283
SHIFT3.jpg =>	0.2404
SHIFT4.jpg =>	0.1216

Another good side effect is to avoid the bias of driving staight. Since we are driving always in the center of the road, most of our steering angles will be near zero. Very small steering angles dominate the data.

HISTOGRAM

By shifting every image ramdomly and changing the steering accordingly we get a much bigger variaty of steering angles in the data.

HISTOGRAM DANACH

### Changing the Brightness
To deal with shadows an street surfaces with different color intensities, the brightness of each image will be randomly adjusted with a factor in a range between 0.3 and 1.3.

### Using the side-view Cameras
Since we get images of three cameras in training mode but we use only the center camera for autonomous driving, we can use the side-view cameras to increase the collected data by treating them like center camera images. If the image recorded by one of the side-view cameras would be the center, the car would be to far on the left or the right. So we have to add or subtract an specific factor from the current recorded angle to point the car to the center of the street again. This approach helps massively to center the car again. A correcting factor of 0.25 led to a good performance.

### Cropping
Since the cameras does not see only the street but also the sky, the surroundings and the hood of the car, each image will be cropped. One reason for this is of course performance, training the model and predicting an angle goes much faster with smaller images. Another reason is not to confuse the CNN with unnecessary information. The goal is to get general driving model and the only parameter has to be the road in front of the car and not a lamp post next to a specific curve. Therefore the images will be cropped 64 px from above and 23 pixel from below.

### FLipping
Since the test track is rougly a circle one steering direction will dominate and bias the model. A simple approach to avoid this and to get even more training data, is by copying and flipping each data.

## Architecture
My architecture consists of four convolutional layers with increasing number of filters, each followed by a RELU activation layer and a Maxpooling layer. The following for Fully Connected layers are seperated by Dropout layers to prevent overfitting, which is a perpetual danger in this project.

ARCHITCTURE
