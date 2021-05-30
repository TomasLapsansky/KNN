import argparse
import os
import random

from keras.callbacks import ModelCheckpoint
# from keras.utils import multi_gpu_model


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import keras

import generator as generator
import cv2
import config

width, height, _ = config.INPUT_SHAPE
target_shape = (width, height)


"""
Model bol inspirovany z https://keras.io/examples/vision/siamese_network/

zaciatok citacie
"""


base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable


class DistanceLayer(layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        anc_pos_dist = tf.reduce_sum(tf.square(anchor - positive), -1)
        anc_neg_dist = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (anc_pos_dist, anc_neg_dist)


anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

distances = DistanceLayer()(
    embedding(anchor_input),
    embedding(positive_input),
    embedding(negative_input),
)

siamese_network = Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)


class SiameseModel(Model):
   
    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.acc_tracker = metrics.Mean(name="accuracy")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        """
        Funkcia  pre spocitanie trenovacieho kroku
        """

        with tf.GradientTape() as tape:
            loss = self.get_loss(data)
            acc = self.get_accuracy(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

       
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    def test_step(self, data):
        """
        Funkcia  pre spocitanie testovacieho kroku
        """

        loss, acc = self.get_loss(data), self.get_accuracy(data)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    def get_loss(self, data):
        """
        Funkcia  pre vypocet loss
        """

        anc_pos_dist, anc_neg_dist = self.siamese_network(data)
        loss = anc_pos_dist - anc_neg_dist
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def get_accuracy(self, data):
        """
        Funkcia  pre vypocet accuracy
        """
        anc_pos_dist, anc_neg_dist = self.siamese_network(data)
        return K.mean(anc_pos_dist < anc_neg_dist)

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]  


"""
koniec citacie
"""



"""
Vytvorenie checkpointu
"""

def create_checkpoint():
    checkpoint_path = os.getcwd() + "/checkpoint"
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    filepath = checkpoint_path + "/model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    return callbacks_list


"""
Parsovanie argumentov
"""

def parseArgs():
    parser = argparse.ArgumentParser(description='Directory with captured samples')
    parser.add_argument('-c', action='store', dest='checkpoint',
                        help='Checkpoint file', default=None)
    parser.add_argument('-t', action='store', dest='train',
                        help='Checkpoint file', default=None)
    parser.add_argument('-o', action='store_true', dest='optimized',
                        help='Optimized file loading for large RAM', default=False)

    return parser.parse_args()

"""
Nacitanie obrázku 
"""


def load_i(p2f):
    height, width, _ = config.INPUT_SHAPE

    t_image = cv2.imread(p2f)
    t_image = cv2.resize(t_image, (height, width))
    return t_image

"""
Evaluacia výsledkov
"""


def eval(path_test):
    batch = config.BATCH_SIZE
    lenitem = batch

    gen_val = generator.MyGenerator(path_test, "image_test/", batch, lenitem)
    dirc = config.VERI_DATASET + "image_test/"

    N = 1000
    positive_cnt = 0
    negative_cnt = 0

    for i in range(N):
        row = gen_val.df.sample()

        # Load car ID and image name for anchor
        car_A, name_A, color, = row['vehicleID'].values[0], row['imageName'].values[0], row['colorID'].values[0]

        # Load car ID and image name for positive
        positive_row = (gen_val.df.loc[gen_val.df['vehicleID'] == car_A]).sample()
        car_P, name_P = positive_row['vehicleID'].values[0], positive_row['imageName'].values[0]

        negative_row = (gen_val.df.loc[gen_val.df['vehicleID'] != car_A]).sample()
        car_N, name_N = negative_row['vehicleID'].values[0], negative_row['imageName'].values[0]

        path_test = config.VERI_DATASET + 'test_label.xml'

        gen_val = generator.MyGenerator(path_test, "image_test/", config.IMAGES, config.IMAGES)

        anchor, positive, negative = load_i(dirc + name_A), load_i(dirc + name_P), load_i(dirc + name_N)

        anchor_embedding, positive_embedding, negative_embedding = (
            embedding(keras.applications.resnet50.preprocess_input(np.array([anchor]), data_format='channels_last')),
            embedding(keras.applications.resnet50.preprocess_input(np.array([positive]), data_format='channels_last')),
            embedding(keras.applications.resnet50.preprocess_input(np.array([negative]), data_format='channels_last')),
        )

        cosine_similarity = metrics.CosineSimilarity()

        positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
        print("Positive similarity:", positive_similarity.numpy())

        negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
        print("Negative similarity:", negative_similarity.numpy())

        if positive_similarity > config.THRESHOLD:
            positive_cnt += 1

        if negative_similarity < config.THRESHOLD:
            negative_cnt += 1

    print("Positive accuracy: ", positive_cnt / N)
    print("Negative accuracy: ", negative_cnt / N)
    print("Total accuracy: ", (positive_cnt + negative_cnt) / (2 * N))

"""
Funkcia pre vykonanie predikcie
"""

def make_prediction(path):
    print(path)
    files = os.listdir(path)

    stringsByPrefix = {}

    for string in files:
        prefix, suffix = map(str.strip, string.split("_", 1))
        group = stringsByPrefix.setdefault(prefix, [])
        group.append(string)

    my_set = []
    query = []

    for key in stringsByPrefix:
        temp_list = sorted(stringsByPrefix[key])
        temp_choice = random.choice(temp_list)
        query.append(temp_choice)
        temp_list.remove(temp_choice)
        my_set += temp_list

    APs = []
    tmp = 0

    for quer in query:
        query_img = load_i(path + "/" + quer)
        quer_id = quer.split("_", 1)[0]
        query_prefix, query_suffix = map(str.strip, quer.split("_", 1))

        AP = 0
        tmp1 = 0
        cnt = 0
        for item in my_set:
            item_prefix, item_suffix = map(str.strip, quer.split("_", 1))
            if item_prefix != query_prefix:
                continue
            tmp1 += 1
            cnt += 1
            print(str(tmp1), end="\r")
            set_img = load_i(path + "/" + item)

            anchor_embedding, positive_embedding, negative_embedding = (
                embedding(
                    keras.applications.resnet50.preprocess_input(np.array([query_img]), data_format='channels_last')),
                embedding(
                    keras.applications.resnet50.preprocess_input(np.array([set_img]), data_format='channels_last')),
                embedding(
                    keras.applications.resnet50.preprocess_input(np.array([set_img]), data_format='channels_last')),
            )

            cosine_similarity = metrics.CosineSimilarity()
            positive_similarity = (cosine_similarity(anchor_embedding, positive_embedding)).numpy()

            set_id = item.split("_", 1)

            if ((positive_similarity > config.THRESHOLD and set_id == quer_id) or (
                    positive_similarity < config.THRESHOLD and set_id != quer_id)):
                AP += positive_similarity * 1
                # AP += 1
            else:
                AP += positive_similarity * 0
                # AP += 0
        # new_ap = AP / (len(stringsByPrefix[quer_id]) - 1)
        new_ap = AP / cnt
        APs.append(new_ap)
        tmp += 1
        print("")
        print("image:" + str(quer) + " " + str(tmp) + "/" + str(len(query)) + " " + str(new_ap) + "                             ")

    mAP = sum(APs) / len(APs)

    print("mAP", mAP)


def main():
    

    arguments = parseArgs()                                                 #Nacitanie argumentov

    siamese_model = SiameseModel(siamese_network)                           #Vytvorenie siete

    siamese_model.compile(optimizer=optimizers.Adam(0.0001))                #Kompilacia sieti

    if arguments.checkpoint:
        
        """
        Spustenie natrenovaneho modelu a vykonanie predikcie
        """

        checkpoint = arguments.checkpoint
        print("Using checkpoint", checkpoint)
        if not os.path.exists(checkpoint):
            print("Checkpoint nenajdeny")
            exit(1)
        siamese_model.built = True
        siamese_model.load_weights(checkpoint)

        path_test = config.VERI_DATASET + 'test_label.xml'

        make_prediction(config.VERI_DATASET + "image_query")
        exit(0)
        eval(path_test)

    else:

        """
        Spustenie modelu a spustenie trenovania
        """

        if arguments.train:
            """
            Ak existuje checkpoint 
            """
            checkpoint = arguments.train
            print("Using checkpoint", checkpoint, " and continue training")
            if not os.path.exists(checkpoint):
                print("Checkpoint nenajdeny")
                exit(1)
            siamese_model.built = True
            siamese_model.load_weights(checkpoint)

        callbacks_list = create_checkpoint()

        path_train = config.VERI_DATASET + 'train_label.xml'                                        #Nastavenie cesty k train
        path_test = config.VERI_DATASET + 'test_label.xml'                                          #Nastavenie cesty k test
        batch = config.BATCH_SIZE                                                                   #Nastavenie batch
        lenitem = batch

        gen_train = generator.MyGenerator(path_train, "image_train/", batch, lenitem)               #Vytvorenie generatoru pre train
        gen_val = generator.MyGenerator(path_test, "image_test/", batch, lenitem)                   #Vytvorenie generatoru pre test
        SPE = len(gen_train.Y_train) / config.EPOCHS
        print(SPE)

        model_in = None                                                                             # Trenovanie 
        for i in range(1, config.EPOCHS + 1):
            print("EPOCH:" + str(i) + "/" + str(config.EPOCHS))

            model_in = embedding

            siamese_model.fit(gen_train.miningGen(emb_model=model_in),
                              steps_per_epoch=config.SPE,
                              validation_data=gen_val.LocalSet(),
                              epochs=1,
                              validation_steps=config.VSTEPS,
                              shuffle=False,
                              use_multiprocessing=False,
                              callbacks=callbacks_list)


if __name__ == "__main__":
    main()
