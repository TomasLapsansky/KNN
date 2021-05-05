import argparse
import sys

import tensorflow as tf
import keras
from keras.applications.resnet import ResNet50
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model

import keras.layers as kl


from keras.preprocessing.image import ImageDataGenerator


import xml.etree.ElementTree as ET
import numpy as np

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
T_G_SEED = 1337

def triplet_hard_loss(y_true, y_pred):
    
    ######################################################################################
    
    embeddings = y_pred
    anchor_positive = embeddings[0]+embeddings[1]
    negative = embeddings[2]


    # Compute pairwise distance between all of anchor-positive
    dot_product = K.dot(anchor_positive, K.transpose(anchor_positive))
    square = K.square(anchor_positive)
    a_p_distance = K.reshape(K.sum(square, axis=1), (-1,1)) - 2.*dot_product  + K.sum(K.transpose(square), axis=0) + 1e-6
    a_p_distance = K.maximum(a_p_distance, 0.0) ## Numerical stability


    # Compute distance between anchor and negative
    dot_product_2 = K.dot(anchor_positive, K.transpose(negative))
    negative_square = K.square(negative)
    a_n_distance = K.reshape(K.sum(square, axis=1), (-1,1)) - 2.*dot_product_2  + K.sum(K.transpose(negative_square), axis=0)  + 1e-6
    a_n_distance = K.maximum(a_n_distance, 0.0) ## Numerical stability
    
    hard_negative = K.reshape(K.min(a_n_distance, axis=1), (-1, 1))
    
    distance = (a_p_distance - hard_negative + 0.2)
    loss = K.mean(K.maximum(distance, 0.0))/(2.)     
        
    return loss


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def create_model():

    emb_size = 16

    print('Creating a model ...')

     # Initialize a ResNet50_ImageNet Model
    resnet_input = kl.Input(shape=config.INPUT_SHAPE)
    resnet_model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top = False, input_tensor=resnet_input)

    # New Layers over ResNet50
    net = resnet_model.output
    #net = kl.Flatten(name='flatten')(net)
    net = kl.GlobalAveragePooling2D(name='gap')(net)
    #net = kl.Dropout(0.5)(net)
    net = kl.Dense(emb_size,activation='relu',name='t_emb_1')(net)
    net = kl.Lambda(lambda  x: K.l2_normalize(x,axis=1), name='t_emb_1_l2norm')(net)

    # model creation
    base_model = Model(resnet_model.input, net, name="base_model")

    # triplet framework, shared weights
    input_shape=config.INPUT_SHAPE
    input_anchor = kl.Input(shape=input_shape, name='input_anchor')
    input_positive = kl.Input(shape=input_shape, name='input_pos')
    input_negative = kl.Input(shape=input_shape, name='input_neg')

    net_anchor = base_model(input_anchor)
    net_positive = base_model(input_positive)
    net_negative = base_model(input_negative)

    # The Lamda layer produces output using given function. Here its Euclidean distance.
    positive_dist = kl.Lambda(euclidean_distance, name='pos_dist')([net_anchor, net_positive])
    negative_dist = kl.Lambda(euclidean_distance, name='neg_dist')([net_anchor, net_negative])
    tertiary_dist = kl.Lambda(euclidean_distance, name='ter_dist')([net_positive, net_negative])

    # This lambda layer simply stacks outputs so both distances are available to the objective
    stacked_dists = kl.Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist, tertiary_dist])

    model = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')
    
    # Setting up optimizer designed for variable learning rate

    for layer in model.layers:
        model.trainable = True

    # Variable Learning Rate per Layers
    optim = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    model.compile(loss=triplet_hard_loss, optimizer=optim, metrics=[accuracy])
    model.summary()


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
            print(np.array([X1i[0], X2i[0], X3i[0]]).shape)
            yield [X1i[0], X2i[0], X3i[0]], X1i[1]

def load_img(p2f):
    
    t_image = cv2.imread(p2f)
    t_image = cv2.resize(t_image, (T_G_HEIGHT,T_G_WIDTH))
    t_image = t_image.astype("float32")
    t_image = keras.applications.resnet50.preprocess_input(t_image, data_format='channels_last')
    return t_image

def createSet(df, batch_size):
    
    batch = []
    anchor =   []
    positive = []
    negative = []
    dirc=config.VERI_DATASET+"image_train/"

    for i in range(batch_size):



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



        img_A = load_img(dirc+name_A)
        img_P = load_img(dirc+name_P)
        img_N = load_img(dirc+name_N)

        anchor.append(img_A)
        positive.append(img_P)
        negative.append(img_N)

    return np.array(anchor), np.array(positive), np.array(negative)

def main():
    # tensorflow devices (GPU) print
    # print(device_lib.list_local_devices())

   

    model = create_model()
    df=loadXML()
    #print(df)
    
    size_batch = 2048
    batch= 16
    dir_name = ""
    print("\n"*5)
    print("SPUSTAM TRENOVANIE")

    for epoch in range(100):

        print ('Epoch ' + str(epoch), size_batch / batch) 

        anchor, positive, negative = createSet(df,size_batch)
        Y_train = np.random.randint(2, size=(1,2,anchor.shape[0])).T

        
        

        

        model.fit_generator(generator=dataCarGenerator(anchor,positive,negative,Y_train,batch), steps_per_epoch=5, epochs=1, shuffle=False, use_multiprocessing=True)

    

    exit(0)

    
    

    

















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
