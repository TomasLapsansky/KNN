import argparse
import sys

import tensorflow as tf
import keras
from keras.applications.resnet import ResNet50
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.optimizers import Adam



from keras.preprocessing.image import ImageDataGenerator

import xml.etree.ElementTree as ET


import cv2
import matplotlib.pyplot as plt

import pandas as pd

import os

import config
import generator

SN = 3
PN = 24
identity_num = 751


# def create_model_old(input_shape):
#     # Vytvorenie malej siete pre siam
#
#     # Vytvorenie vstupov
#     left_input = Input(input_shape)
#     right_input = Input(input_shape)
#
#     # Auto 1
#     encoded_l = convnet(left_input)
#     # Auto 2
#     encoded_r = convnet(right_input)
#
#     L1_distance = L1_layer([encoded_l, encoded_r])
#     x = Dense(1024)(L1_distance)
#     x = Dropout(0.2)(x)
#     x = Dense(512)(x)
#     x = Dropout(0.2)(x)
#     x = Dense(256)(x)
#     x = Dropout(0.2)(x)
#     x = Activation('relu')(x)
#
#     prediction = Dense(1, activation='sigmoid')(x)
#     # optimizer = Adam(0.001, decay=2.5e-4)
#     optimizer = SGD(learning_rate=0.0001, momentum=0.4)
#
#     model = Model(inputs=[left_input, right_input], outputs=prediction)
#     model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
#
#     print(model.summary())
#     return model

# TRENOVANIE A VYTVORENIE SETU
# https://github.com/noelcodella/tripletloss-keras-tensorflow/blob/master/tripletloss.py


# https://github.com/michuanhaohao/keras_reid/blob/master/reid_tripletcls.py


#Global pre generatory
datagen = ImageDataGenerator(width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 zoom_range=0.1,
                                 vertical_flip=True,
                                )


T_G_HEIGHT,T_G_WIDTH = 224, 224


def triplet_hard_loss(y_true, y_pred):
    global SN
    global PN
    feat_num = SN * PN  # images num
    y_pred = K.l2_normalize(y_pred, axis=1)
    feat1 = K.tile(K.expand_dims(y_pred, axis=0), [feat_num, 1, 1])
    feat2 = K.tile(K.expand_dims(y_pred, axis=1), [1, feat_num, 1])
    delta = feat1 - feat2
    dis_mat = K.sum(K.square(delta), axis=2) + K.epsilon()  # Avoid gradients becoming NAN
    dis_mat = K.sqrt(dis_mat)
    positive = dis_mat[0:SN, 0:SN]
    negetive = dis_mat[0:SN, SN:]
    for i in range(1, PN):
        positive = tf.concat([positive, dis_mat[i * SN:(i + 1) * SN, i * SN:(i + 1) * SN]], axis=0)
        if i != PN - 1:
            negs = tf.concat([dis_mat[i * SN:(i + 1) * SN, 0:i * SN], dis_mat[i * SN:(i + 1) * SN, (i + 1) * SN:]],
                             axis=1)
        else:
            negs = tf.concat(dis_mat[i * SN:(i + 1) * SN, 0:i * SN], axis=0)
        negetive = tf.concat([negetive, negs], axis=0)
    positive = K.max(positive, axis=1)
    negetive = K.min(negetive, axis=1)
    a1 = 0.6
    loss = K.mean(K.maximum(0.0, positive - negetive + a1))
    return loss


def create_model():
    print('Creating a model ...')
    model = ResNet50(weights="imagenet", include_top=False,
                     input_tensor=Input(shape=config.INPUT_SHAPE))

    opt = Adam(0.001, decay=2.5e-4)
    model.compile(loss=triplet_hard_loss, optimizer=opt, metrics=['accuracy'])

    return model

def loadXML():
    print('loading label xml...', end="")
    xml_data = open(config.VERI_DATASET + 'train_label.xml', 'r').read()  # Read file
    root = ET.XML(xml_data) 

    data = []
    cols = []
    items=root[0]
    
    for i, item in enumerate(items):
        data.append([item.attrib['imageName'],item.attrib['vehicleID']])

    df = pd.DataFrame(data)  # Write in DF and transpose it
    df.columns = ["imageName", "vehicleID"]  # Update column names
    print("DONE")
    return df


def dataCarGenerator(X1, X2, X3, Y, b):
    # Funkcia prebrata z https://github.com/noelcodella/tripletloss-keras-tensorflow/blob/master/tripletloss.py
    local_seed = T_G_SEED
    genX1 = datagen.flow(X1,Y, batch_size=b, seed=local_seed, shuffle=False)
    genX2 = datagen.flow(X2,Y, batch_size=b, seed=local_seed, shuffle=False)
    genX3 = datagen.flow(X3,Y, batch_size=b, seed=local_seed, shuffle=False)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            X3i = genX3.next()

            yield [X1i[0], X2i[0], X3i[0]], X1i[1]

def load_img(p2f):
    
    t_image = cv2.imread(p2f)
    t_image = cv2.resize(t_image, (T_G_HEIGHT,T_G_WIDTH))
    t_image = t_image.astype("float32")
    t_image = keras.applications.resnet50.preprocess_input(t_image, data_format='channels_last')
    return t_image

def createSet(df, batch_size):
    
    batch = []
    dirc=config.VERI_DATASET+"image_train/"

    for i in range(16):
        # Load random sample from data frame
        row = df.sample()          

        # Load car ID and image name for anchor      
        car_A, name_A = row['vehicleID'].values[0], row['imageName'].values[0] 
        
        # Load car ID and image name for positive
        positive_row = (df.loc[df['vehicleID'] == car_A]).sample()
        car_P, name_P = positive_row['vehicleID'].values[0], positive_row['imageName'].values[0]
        
        # Load car ID and image name for negative
        negative_row = (df.loc[df['vehicleID'] != car_A]).sample()
        car_N, name_N = negative_row['vehicleID'].values[0], negative_row['imageName'].values[0]
        
        print("A:",car_A, name_A)
        print("P:",car_P, name_P)
        print("N:",car_N, name_N)
        
        

        img_A = load_img(dirc+name_A)
        img_P = load_img(dirc+name_P)
        img_N = load_img(dirc+name_N)

        batch.append([img_A,img_P,img_N])




    return batch

def main():
    # tensorflow devices (GPU) print
    # print(device_lib.list_local_devices())

   

    model = create_model()
    df=loadXML()
    #print(df)
    
    dir_name = ""
    batch = createSet(df,dir_name)

    print(batch[0][0])

    exit(0)

    model.fit_generator(generator=dataCarGenerator(anchors_t,positives_t,negatives_t,Y_train,batch), steps_per_epoch=len(Y_train) / batch, epochs=1, shuffle=False, use_multiprocessing=True)

    

    

















########################################## NACITAVANIE XML
    
    
##########################################

    
    
    exit(1)


    

    print("Start training ")
    print("Loading files...")

    pathA = "capt/A"
    pathB = "capt/B"
    dataset = [x for x in os.listdir(pathA)
               if os.path.isfile(os.path.join(pathA, x))]

    datasetB = [x for x in os.listdir(pathB)
                if os.path.isfile(os.path.join(pathB, x))]

    print("DONE")

    checkpoint_path = os.getcwd() + "/models"
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    filepath = checkpoint_path + "/weights-improvement-epoch-{epoch:02d}-val-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    if (arguments.optimized):
        pairTrain, labelTrain = generator.create_pair_optimized(config.BATCH_SIZE * config.SPE, dataset, datasetB)
        pairTest, labelTest = generator.create_pair_optimized(config.BATCH_SIZE * config.VSTEPS, dataset, datasetB)

        history = model.fit(
            [pairTrain[:, 0], pairTrain[:, 1]], labelTrain,
            validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_steps=config.VSTEPS,
            callbacks=callbacks_list)

    else:
        trainGeneratorOut = generator.create_pair(config.BATCH_SIZE, dataset, datasetB)
        validGeneratorOut = generator.create_pair(config.BATCH_SIZE, dataset, datasetB)

        history = model.fit(
            trainGeneratorOut,
            validation_data=validGeneratorOut,
            steps_per_epoch=config.SPE,
            epochs=config.EPOCHS,
            validation_steps=config.VSTEPS,
            callbacks=callbacks_list)

        # 

        # my_evaluate(model, img_car1, img_car2)


if __name__ == "__main__":
    main()
