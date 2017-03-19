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

![shift1](https://github.com/raiValek/CarND-Behavioral-Cloning-P3/blob/master/img/shift1.jpg)

Angle Correction: -0.1673

![shift2](https://github.com/raiValek/CarND-Behavioral-Cloning-P3/blob/master/img/shift2.jpg)

Angle Correction: -0.2283

![shift3](https://github.com/raiValek/CarND-Behavioral-Cloning-P3/blob/master/img/shift3.jpg)

Angle Correction: 0.2404

![shift4](https://github.com/raiValek/CarND-Behavioral-Cloning-P3/blob/master/img/shift4.jpg)

Angle Correction: 0.1216

Another good side effect is to avoid the bias of driving staight. Since we are driving always in the center of the road, most of our steering angles will be near zero. Very small steering angles dominate the data.

![Angles in recorded Data](https://github.com/raiValek/CarND-Behavioral-Cloning-P3/blob/master/img/org_data.png)

By shifting every image ramdomly and changing the steering accordingly we get a much bigger variaty of steering angles in the data.

![Angles in after Augmentations](https://github.com/raiValek/CarND-Behavioral-Cloning-P3/blob/master/img/hist_aug_data.png)

### Changing the Brightness
To deal with shadows an street surfaces with different color intensities, the brightness of each image will be randomly adjusted with a factor in a range between 0.3 and 1.3.

Examples with changed brightness

![bright1](https://github.com/raiValek/CarND-Behavioral-Cloning-P3/blob/master/img/bright1.jpg)

![bright2](https://github.com/raiValek/CarND-Behavioral-Cloning-P3/blob/master/img/bright2.jpg)

![bright3](https://github.com/raiValek/CarND-Behavioral-Cloning-P3/blob/master/img/bright3.jpg)

![bright4](https://github.com/raiValek/CarND-Behavioral-Cloning-P3/blob/master/img/bright4.jpg)

### Using the side-view Cameras
Since we get images of three cameras in training mode but we use only the center camera for autonomous driving, we can use the side-view cameras to increase the collected data by treating them like center camera images. If the image recorded by one of the side-view cameras would be the center, the car would be to far on the left or the right. So we have to add or subtract an specific factor from the current recorded angle to point the car to the center of the street again. This approach helps massively to center the car again. A correcting factor of 0.25 led to a good performance.

### Cropping
Since the cameras does not see only the street but also the sky, the surroundings and the hood of the car, each image will be cropped. One reason for this is of course performance, training the model and predicting an angle goes much faster with smaller images. Another reason is not to confuse the CNN with unnecessary information. The goal is to get general driving model and the only parameter has to be the road in front of the car and not a lamp post next to a specific curve. Therefore the images will be cropped 64 px from above and 23 pixel from below.

![not_cropped](https://github.com/raiValek/CarND-Behavioral-Cloning-P3/blob/master/img/not_cropped.jpg)

Original Image Size

![cropped](https://github.com/raiValek/CarND-Behavioral-Cloning-P3/blob/master/img/cropped.jpg)

Cropped Image

### FLipping
Since the test track is rougly a circle one steering direction will dominate and bias the model. A simple approach to avoid this and to get even more training data, is by copying and flipping each data.

## Architecture
My architecture consists of four convolutional layers with increasing number of filters, each followed by a RELU activation layer and a Maxpooling layer. The following for Fully Connected layers are seperated by Dropout layers to prevent overfitting, which is a perpetual danger in this project.
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 64, 64, 16)    1216        lambda_1[0][0]                   
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 64, 64, 16)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 32, 32, 16)    0           activation_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 32, 32, 32)    4640        maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 32, 32, 32)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 16, 16, 32)    0           activation_2[0][0]               
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 16, 16, 64)    18496       maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 16, 16, 64)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 8, 8, 64)      0           activation_3[0][0]               
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 8, 8, 128)     73856       maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 8, 8, 128)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 4, 4, 128)     0           activation_4[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2048)          0           maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 2048)          0           flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 512)           1049088     dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 512)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 256)           131328      dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 256)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 64)            16448       dropout_3[0][0]                  
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 64)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 16)            1040        dropout_4[0][0]                  
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 16)            0           dense_4[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             17          dropout_5[0][0]                  
====================================================================================================
Total params: 1,296,129
Trainable params: 1,296,129
Non-trainable params: 0
____________________________________________________________________________________________________
```
