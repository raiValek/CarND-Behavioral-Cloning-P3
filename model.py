# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.utils 
from random import shuffle
from sklearn.model_selection import train_test_split
from scipy.ndimage import imread
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.optimizers import Adam

CROP_ABOVE = 64
CROP_BEYOND = 23
SIDE_CAMERA_CORRECTION = 0.25

def change_brightness(img):
    """Changes the brightness of the given image randomly
    """ 
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    brightness = np.random.uniform(low=0.3, high=1.3)
    img_hsv[:,:,2] = img_hsv[:,:,2] * brightness
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return img

def change_steering(image, steer):
    """Shifts the image right or left and changes the steering accordingly
    """
    rows,cols,ch = image.shape
    shift = np.random.uniform(low=-50.0, high=50.0)
    new_angle = steer + shift/50.0*.4
    shift_matrix = np.float32([[1,0,shift],[0,1,0]])
    new_image = cv2.warpAffine(image, shift_matrix, (cols,rows))
 
    return new_image, new_angle

def crop_resize(img):
    """Crops the the image above and beyond the visible street and resizes it to (64, 64)
    """
    img = img[CROP_ABOVE:-CROP_BEYOND,:,:]
    img = cv2.resize(img, (64, 64), cv2.INTER_AREA)
    return img

def create_model():
    """Returns the model of the CNN
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x-127.5)/127.5, input_shape=(64, 64, 3)))
    model.add(Convolution2D(16, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Dropout(0.5))
    model.add(Dense(16))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    adam=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer=adam)

    return model

def get_training_data():
    """Loads all image paths and angles out of "./data/driving_log.csv
    """
    
    # open the csv file
    lines = []
    with open('./data/driving_log.csv') as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)

    # load all image paths and steering angles
    image_paths = []
    angles = []
    for line in lines:
        # get the angle
        angle = float(line[3])
        # get image paths
        for i in range(3):
            filename = line[i].split('/')[-1]
            image_paths.append('data/IMG/' + filename)
            # augment the angle if camera is a side camera
            if i == 1:
                angle += SIDE_CAMERA_CORRECTION
            elif i == 2:
                angle -= SIDE_CAMERA_CORRECTION

            angles.append(angle)
               
    return image_paths, angles

def create_generator(X_train, y_train, batch_size=128, valid=False):
    """Returns a python generator for training or validation data
    """

    while 1:
        # for every batch take half original data and half flipped data and augment 'em all
        for i in range(0, len(X_train), int(batch_size/2)):
            y_batch_angles = y_train[i:i+int(batch_size/2)]
            X_batch_paths = X_train[i:i+int(batch_size/2)]
            
            X_batch = []
            y_batch = []
            for path, angle in zip(X_batch_paths, y_batch_angles):               
                original_img = imread(path)
                original_angle = angle            

                flipped_img = np.fliplr(original_img)
                flipped_angle = -angle
                
                if not valid:
                    original_img = change_brightness(original_img)
                    flipped_img = change_brightness(flipped_img)

                    original_img, original_angle = change_steering(original_img, original_angle)
                    flipped_img, flipped_angle = change_steering(flipped_img, flipped_angle)

                original_img = crop_resize(original_img)
                flipped_img = crop_resize(flipped_img)

                X_batch.append(original_img)
                y_batch.append(original_angle)
                
                X_batch.append(flipped_img)
                y_batch.append(flipped_angle)

            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)

            yield sklearn.utils.shuffle(X_batch, y_batch)

def main():
    paths, angles = get_training_data()

    train_paths, valid_paths, train_angles, valid_angles = train_test_split(paths, angles, test_size=0.1)

    # create generators for training and validation data
    train_generator = create_generator(train_paths, train_angles)
    valid_generator = create_generator(valid_paths, valid_angles, valid=True)

    model = create_model()

    # train model with four times the count of the original captured data per epoch
    model.fit_generator(train_generator, samples_per_epoch=len(train_paths)*4, validation_data=valid_generator, nb_val_samples=len(valid_paths)*4, nb_epoch=20)

    # save model in current directory
    model.save('model.h5')

if __name__ == '__main__':
    main()
