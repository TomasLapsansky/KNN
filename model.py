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

INPUT_SHAPE = (128, 128, 3)

L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

def get_pair(camera, id):
    try:
        df = pd.read_csv("pairs/ground_truth_shot_" + str(camera) + ".csv")
        options = [str(camera) + "A/" + str(id)]
        pair = str(df[df.camA.str.startswith(tuple(options))]['camB'])
        return pair.split("_")[0].split('"')[1]
    except:
        return "none/none"


def create_pair(images, batch_size, positive, dataset, datasetB):
    output = []

    pairImg = []
    pairLab = []
    labels = []
    pathA = "capt/A"
    pathB = "capt/B"
    width, height, rest = INPUT_SHAPE

    i = 0
    while (i < batch_size):
        print(i, batch_size, end='\r')
        
        image = random.choice(dataset)
        if (image[1] == "B"):
            print("Nenasiel som A")
            continue

        imagelist = image.split("_")
        pair = get_pair(int(imagelist[0][0]), int(imagelist[2]))
        if(pair=="none/none"):
            continue
        
        pair = pair.split("/")
        prefix = pair[0][0] + "B_id_" + pair[1]
        set_list = []
        #print("Mam obrazok A",i)
        label = None
        
        if (random.choice([True, False])):
            pat = re.compile(r"^"+prefix)
            dir_list = os.listdir(pathB)
            list_name =  [i for i in dir_list if pat.match(i)]          
            
            for file in list_name:
                
                if file.startswith(prefix):
                    set_list.append(file)
                else:
                    print("Nemam B")

            if (set_list == []):
                continue

            img1 = image
            img2 = random.choice(set_list)
            label = [1]
            #print("Mam obrazok B",i)

        else:

            img1 = image
            img2 = random.choice(datasetB)
            label = [0]
            #print("Mam obrazok B",i)

        img1 = cv2.imread(pathA + "/" + img1)
        img1 = cv2.resize(img1, (width, height))

        img2 = cv2.imread(pathB + "/" + img2)
        img2 = cv2.resize(img2, (width, height))

       #plt.figure()

       # f, axarr = plt.subplots(2,1) 

# use the created array to output your multiple images. In this case I have stacked 4 images vertically
       # axarr[0].imshow(img1,cmap="hot")
       # axarr[1].imshow(img2,cmap="hot")
       # print(label)
       # plt.show()

        labels.append(label)
        output.append([img1, img2])

        i = i + 1

    return (np.array(output), np.array(labels))


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

    return Model(input1,x)


def create_model(input_shape=(128, 128, 3)):
    
    # Vytvorenie malej siete pre siam 
    convnet = small_vgg(input_shape)

    # Vytvorenie vstupov
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    #Auto 1
    encoded_l = convnet(left_input)
    #Auto 2
    encoded_r = convnet(right_input)


    L1_distance = L1_layer([encoded_l, encoded_r])
    x = Dense(1024)(L1_distance)
    x = Dropout(0.2)(x)
    x = Dense(512)(x)
    x = Dropout(0.2)(x)
    x = Dense(256)(x)
    x = Dropout(0.2)(x)
    x = Activation('relu')(x)

    prediction = Dense(1,activation='softmax')(x)
    optimizer = Adam(0.001, decay=2.5e-4)

    model = Model(inputs=[left_input,right_input],outputs=prediction)
    model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])

    print(model.summary())
    return model


def main():
    EPOCHS = 100
    BATCH_SIZE = 16
    SPE = 100
    no_output = 2

    print(device_lib.list_local_devices())

    print("Nacitavam subory...")
    pathA = "capt/A"
    pathB = "capt/B"
    dataset=[x for x in os.listdir(pathA)
                               if os.path.isfile(os.path.join(pathA, x))]  

    datasetB = [x for x in os.listdir(pathB)
                                  if os.path.isfile(os.path.join(pathB, x))]
     
    print("DONE")

    traingeneratorOut = create_pair(1, 2000, True, dataset, datasetB)
    validgeneratorOut = create_pair(1, 1000, True, dataset, datasetB)
    pairTrain, labelTrain = traingeneratorOut
    pairTest, labelTest = validgeneratorOut
    model = create_model(input_shape=INPUT_SHAPE)

    #opt = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
    
    history = model.fit(
        [pairTrain[:, 0], pairTrain[:, 1]], labelTrain, 
        validation_data=([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:]),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS)


if __name__ == "__main__":
    main()
