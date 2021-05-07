import tensorflow_addons as tfa
import tensorflow as tf
import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model

import keras.layers as kl

import tensorflow_addons as tfa # Pre pouzitie triplet loss priamo z TF

import numpy as np

# My imports
import config
import generator



SN = 3
PN = 24
identity_num = 751

# TRENOVANIE A VYTVORENIE SETU
# https://github.com/noelcodella/tripletloss-keras-tensorflow/blob/master/tripletloss.py

# https://github.com/michuanhaohao/keras_reid/blob/master/reid_tripletcls.py


# Global pre generatory
T_G_HEIGHT, T_G_WIDTH = 224, 224
T_G_SEED = 1337






def triplet_hard_loss(_, y_pred):
    ######################################################################################

    embeddings = y_pred
    anchor_positive = embeddings[0] + embeddings[1]
    negative = embeddings[2]

    # Compute pairwise distance between all of anchor-positive
    dot_product = K.dot(anchor_positive, K.transpose(anchor_positive))
    square = K.square(anchor_positive)
    a_p_distance = K.reshape(K.sum(square, axis=1), (-1, 1)) - 2. * dot_product + K.sum(K.transpose(square),
                                                                                        axis=0) + 1e-6
    a_p_distance = K.maximum(a_p_distance, 0.0)  ## Numerical stability

    # Compute distance between anchor and negative
    dot_product_2 = K.dot(anchor_positive, K.transpose(negative))
    negative_square = K.square(negative)
    a_n_distance = K.reshape(K.sum(square, axis=1), (-1, 1)) - 2. * dot_product_2 + K.sum(K.transpose(negative_square),
                                                                                          axis=0) + 1e-6
    a_n_distance = K.maximum(a_n_distance, 0.0)  ## Numerical stability

    hard_negative = K.reshape(K.min(a_n_distance, axis=1), (-1, 1))

    distance = (a_p_distance - hard_negative + 0.2)
    loss = K.mean(K.maximum(distance, 0.0)) / (2.)

    return loss


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def accuracy(_, y_pred):
    return K.mean(y_pred[:, 0, 0] < y_pred[:, 1, 0])




def create_model():
    emb_size = 776  # number of classes in dataset

    print('Creating a model ...')

    # Initialize a ResNet50_ImageNet Model
    resnet_input = kl.Input(shape=config.INPUT_SHAPE)
    resnet_model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False,
                                                        input_tensor=resnet_input)

    # New Layers over ResNet50
    net = resnet_model.output
    # net = kl.Flatten(name='flatten')(net)
    net = kl.GlobalAveragePooling2D(name='gap')(net)
    # net = kl.Dropout(0.5)(net)
    net = kl.Dense(emb_size, activation='relu', name='t_emb_1')(net)
    net = kl.Lambda(lambda x: K.l2_normalize(x, axis=1), name='t_emb_1_l2norm')(net)

    # model creation
    base_model = Model(resnet_model.input, net, name="base_model")

    # triplet framework, shared weights
    input_shape = config.INPUT_SHAPE
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
    stacked_dists = kl.Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')(
        [positive_dist, negative_dist, tertiary_dist])

    model = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')

    # Setting up optimizer designed for variable learning rate

    for layer in model.layers:
        layer.trainable = True

    # Variable Learning Rate per Layers
    optimizer = Adam(lr=0.001)
    #model.compile(loss=tfa.losses.TripletSemiHardLoss(), optimizer=optimizer, metrics=[accuracy])
    model.compile(loss=tfa.losses.TripletSemiHardLoss(), optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model


def main():

    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    #tensorflow devices (GPU) print
    print(tf.config.list_physical_devices('GPU'))

    print("Version of Tensor Flow:", tf.__version__)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    

    model = create_model()

    print("\n" * 5)

    
    path_train = config.VERI_DATASET + 'train_label.xml'
    path_test = config.VERI_DATASET + 'test_label.xml'
    batch = config.BATCH_SIZE
    lenitem = batch

    

    gen_train = generator.MyGenerator(path_train, "image_train/", batch, lenitem)
    gen_val = generator.MyGenerator(path_test, "image_test/", batch, lenitem)
    SPE = len(gen_train.Y_train) / config.EPOCHS
    print(SPE)


    model.fit(gen_train.newLocalSet(),
              steps_per_epoch=config.SPE,
              validation_data=gen_val.newLocalSet(),
              epochs=config.EPOCHS,
              validation_steps= 50,
              shuffle=False,
              use_multiprocessing=False)


if __name__ == "__main__":
    main()
