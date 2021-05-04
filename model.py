import argparse
import sys

import tensorflow as tf
import keras
from keras.applications.resnet import ResNet50
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.optimizers import Adam


import xml.etree.ElementTree as ET

import cv2

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
    model = ResNet50(weights="imagenet", include_top=False,
                     input_tensor=Input(shape=config.INPUT_SHAPE))

    opt = Adam(0.001, decay=2.5e-4)
    model.compile(loss=triplet_hard_loss, optimizer=opt, metrics=['accuracy'])

    return model


def main():
    # tensorflow devices (GPU) print
    # print(device_lib.list_local_devices())

    model = create_model()
    #print(model.summary())



########################################## NACITAVANIE XML
    print('loading label xml...')
    
    xml_data = open(config.VERI_DATASET + 'train_label.xml', 'r').read()  # Read file
    root = ET.XML(xml_data) 

    data = []
    cols = []
    items=root[0]
    
    for i, item in enumerate(items):
        print(item.attrib['imageName'], end=" ")
        print(item.attrib['vehicleID'], end=" ")
        print(item.attrib['cameraID'], end=" ")
        print(item.attrib['colorID'], end=" ")
        print(item.attrib['typeID'])
        
        data.append([item.attrib['imageName'],item.attrib['vehicleID']])
        

        #data.append([subchild.text for subchild in child])
        #cols.append(child.tag)
        
    
    

    df = pd.DataFrame(data)  # Write in DF and transpose it
    df.columns = ["imageName", "vehicleID"]  # Update column names
    print(df)
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
