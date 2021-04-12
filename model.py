import argparse
import sys

import tensorflow as tf
import keras
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from keras.layers.core import Lambda
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np
from keras import backend as K
import pandas as pd
import cv2
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random, os
from sklearn.model_selection import train_test_split

import config
import generator

L1_layer = Lambda(lambda tensor: K.abs(tensor[0] - tensor[1]))


def initialize_weights(shape, dtype=None):
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def initialize_bias(shape, dtype=None):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def small_vgg(input_shape):
    input1 = Input(input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input1)

    # Block 1
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten()(x)
    x = Dense(512)(x)

    return Model(input1, x)


def create_model(input_shape):
    # Vytvorenie malej siete pre siam
    convnet = small_vgg(input_shape)

    # Vytvorenie vstupov
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Auto 1
    encoded_l = convnet(left_input)
    # Auto 2
    encoded_r = convnet(right_input)

    L1_distance = L1_layer([encoded_l, encoded_r])
    x = Dense(1024)(L1_distance)
    x = Dropout(0.2)(x)
    x = Dense(512)(x)
    x = Dropout(0.2)(x)
    x = Dense(256)(x)
    x = Dropout(0.2)(x)
    x = Activation('relu')(x)

    prediction = Dense(1, activation='sigmoid')(x)
    # optimizer = Adam(0.001, decay=2.5e-4)
    optimizer = SGD(learning_rate=0.0001, momentum=0.4)

    model = Model(inputs=[left_input, right_input], outputs=prediction)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())
    return model


def parseArgs():
    parser = argparse.ArgumentParser(description='Directory with captured samples')
    parser.add_argument('-d', action='store', dest='directory',
                        help='Directory to captures', default="")
    parser.add_argument('-c', action='store', dest='checkpoint',
                        help='Checkpoint file', default=None)

    return parser.parse_args()


def my_evaluate(model, img_car1, img_car2):
    width, height, rest = config.INPUT_SHAPE

    car1 = cv2.imread(img_car1)
    car1 = cv2.resize(car1, (width, height))

    car2 = cv2.imread(img_car2)
    car2 = cv2.resize(car2, (width, height))

    resized = [car1, car2]

    out = list(model.predict(resized))

    print(out)


def main():

    arguments = parseArgs()

    # tensorflow devices (GPU) print
    # print(device_lib.list_local_devices())

    model = create_model(input_shape=config.INPUT_SHAPE)

    if (arguments.checkpoint != None):
        checkpoint = arguments.checkpoint
        print("Using checkpoint", checkpoint)
        if (not os.path.exists(checkpoint)):
            print("Checkpoint nenajdeny")
            exit(1)
        model.load_weights(checkpoint)
    else:
        print("Start training ")
        print("Loading files...")
        
        arg_dir = arguments.directory
        if(not arguments.directory == ""):
            arg_dir = arguments.directory+"/"

        pathA = arg_dir+"capt/A"
        pathB = arg_dir+"capt/B"
        dataset = [x for x in os.listdir(pathA)
                   if os.path.isfile(os.path.join(pathA, x))]

        datasetB = [x for x in os.listdir(pathB)
                    if os.path.isfile(os.path.join(pathB, x))]

        print("DONE")

        checkpoint_path = os.getcwd() + "/checkpoint"
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        filepath = checkpoint_path + "/weights-improvement-epoch-{epoch:02d}-val-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        trainGeneratorOut = generator.create_pair(config.BATCH_SIZE, dataset, datasetB)
        validGeneratorOut = generator.create_pair(config.BATCH_SIZE, dataset, datasetB)
        # pairTrain, labelTrain = trainGeneratorOut
        # pairTest, labelTest = validGeneratorOut

        # opt = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

        # history = model.fit(
        #     [pairTrain[:, 0], pairTrain[:, 1]], labelTrain,
        #     validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
        #     batch_size=config.BATCH_SIZE,
        #     epochs=config.EPOCHS,
        #     callbacks=callbacks_list)

        history = model.fit(
            trainGeneratorOut,
            validation_data=validGeneratorOut,
            steps_per_epoch=100,
            epochs=config.EPOCHS,
            validation_steps=20,
            callbacks=callbacks_list)

        # my_evaluate(model, img_car1, img_car2)


if __name__ == "__main__":
    main()
