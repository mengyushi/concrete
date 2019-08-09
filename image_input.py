import numpy as np
import h5py
import scipy
from scipy import ndimage

def imput_training_set():
    num_px = 64

    num_true = 70
    num_false = 30

    train_x = np.zeros((num_px*num_px*3,num_true+num_false))
    train_y = np.zeros((num_true+num_false,1))

    for n in range(1,num_true+1):
        my_image = "my_image_"+ str(n) +".jpg" # change this to the name of your image file
        my_label = 1 # the true class of your image


        fname = "images_training/" + my_image
        image = np.array(ndimage.imread(fname, flatten=False))
        my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))


        for i in range(num_px*num_px*3):
            train_x[i-1][n-1] = my_image[i-1][0]

        train_y[n-1][0] = my_label


    for n in range(num_true+1,num_true+num_false+1):
        my_image = "my_image_non_"+ str(n-num_true) +".jpg" # change this to the name of your image file
        my_label = 0 # the true class of your image

        fname = "images_training/" + my_image
        image = np.array(ndimage.imread(fname, flatten=False))
        my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))


        for i in range(num_px*num_px*3):
            train_x[i-1][n-1] = my_image[i-1][0]

        train_y[n-1][0] = my_label

        return train_x,train_y.T

def imput_testing_set():
    num_px = 64
    num_true = 10
    num_false = 10

    testing_x = np.zeros((num_px*num_px*3,num_true+num_false))
    testing_y = np.zeros((num_true+num_false,1))

    for n in range(1,num_true+1):
        my_image = "my_image_"+ str(n) +".jpg" # change this to the name of your image file
        my_label = 1 # the true class of your image


        fname = "images_testing/" + my_image
        image = np.array(ndimage.imread(fname, flatten=False))
        my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))


        for i in range(num_px*num_px*3):
            testing_x[i-1][n-1] = my_image[i-1][0]

        testing_y[n-1][0] = my_label


    for n in range(num_true+1,num_true+num_false+1):
        my_image = "my_image_non_"+ str(n-num_true) +".jpg" # change this to the name of your image file
        my_label = 0 # the true class of your image

        fname = "images_testing/" + my_image
        image = np.array(ndimage.imread(fname, flatten=False))
        my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))


        for i in range(num_px*num_px*3):
            testing_x[i-1][n-1] = my_image[i-1][0]

        testing_y[n-1][0] = my_label

        return testing_x,testing_y.T
