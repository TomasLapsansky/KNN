import argparse
import os

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
## Setting up the embedding generator model

Our Siamese Network will generate embeddings for each of the images of the
triplet. To do this, we will use a ResNet50 model pretrained on ImageNet and
connect a few `Dense` layers to it so we can learn to separate these
embeddings.

We will freeze the weights of all the layers of the model up until the layer `conv5_block1_out`.
This is important to avoid affecting the weights that the model has already learned.
We are going to leave the bottom few layers trainable, so that we can fine-tune their weights
during training.
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

"""
## Setting up the Siamese Network model

The Siamese network will receive each of the triplet images as an input,
generate the embeddings, and output the distance between the anchor and the
positive embedding, as well as the distance between the anchor and the negative
embedding.

To compute the distance, we can use a custom layer `DistanceLayer` that
returns both values as a tuple.
"""


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

distances = DistanceLayer()(
    embedding(anchor_input),
    embedding(positive_input),
    embedding(negative_input),
)

siamese_network = Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)

"""
## Putting everything together

We now need to implement a model with custom training loop so we can compute
the triplet loss using the three embeddings produced by the Siamese network.

Let's create a `Mean` metric instance to track the loss of the training process.
"""


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.acc_tracker = metrics.Mean(name="accuracy")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
            acc = self._compute_acc(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        acc = self._compute_acc(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def _compute_acc(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        return K.mean(ap_distance < an_distance)

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]  # EDIT < NEUPRAVILI SME TO MALO BY TO TAM BYT @FILIP


def create_checkpoint():
    checkpoint_path = os.getcwd() + "/checkpoint"
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    filepath = checkpoint_path + "/model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    return callbacks_list


def parseArgs():
    parser = argparse.ArgumentParser(description='Directory with captured samples')
    parser.add_argument('-c', action='store', dest='checkpoint',
                        help='Checkpoint file', default=None)
    parser.add_argument('-t', action='store', dest='train',
                        help='Checkpoint file', default=None)
    parser.add_argument('-o', action='store_true', dest='optimized',
                        help='Optimized file loading for large RAM', default=False)


    return parser.parse_args()


def load_i(p2f):
    height, width, _ = config.INPUT_SHAPE

    t_image = cv2.imread(p2f)
    t_image = cv2.resize(t_image, (height, width))
    return t_image


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

        # f, axarr = plt.subplots(2, 2)
        # axarr[0, 0] = plt.imshow(positive)
        # axarr[0, 1] = plt.imshow(negative)
        # axarr[1, 0] = plt.imshow(anchor)
        # axarr[1, 1] = plt.imshow(positive)
        #
        # plt.show()

    print("Positive accuracy: ", positive_cnt / N)
    print("Negative accuracy: ", negative_cnt / N)
    print("Total accuracy: ", (positive_cnt + negative_cnt) / (2 * N))


def main():
    """
    ## Training

    We are now ready to train our model.
    """

    arguments = parseArgs()

    siamese_model = SiameseModel(siamese_network)

    siamese_model.compile(optimizer=optimizers.Adam(0.0001))

    if arguments.checkpoint:
       
        checkpoint = arguments.checkpoint
        print("Using checkpoint", checkpoint)
        if not os.path.exists(checkpoint):
            print("Checkpoint nenajdeny")
            exit(1)
        siamese_model.built = True
        siamese_model.load_weights(checkpoint)

        # path_test = config.VERI_DATASET + 'train_label.xml'
        # batch = config.BATCH_SIZE
        # lenitem = batch
        #
        # gen_val = generator.MyGenerator(path_test, "image_train/", batch, lenitem)
        # dirc = config.VERI_DATASET + "image_train/"
        # for i in range(10):
        #     gen_val.df = gen_val.df.sample(frac=1).reset_index(drop=True)
        #     row = gen_val.df.sample()
        #
        #     # Load car ID and image name for anchor
        #     car_A, name_A, color, = row['vehicleID'].values[0], row['imageName'].values[0], row['colorID'].values[0]
        #
        #     # Load car ID and image name for positive
        #     positive_row = (gen_val.df.loc[gen_val.df['vehicleID'] == car_A]).sample()
        #     car_P, name_P = positive_row['vehicleID'].values[0], positive_row['imageName'].values[0]
        #
        #     negative_row = (gen_val.df.loc[gen_val.df['vehicleID'] != car_A])
        #     # Load car ID and image name for negative
        #     negative_row = (negative_row.loc[negative_row['colorID'] == color]).sample()
        #     car_N, name_N = negative_row['vehicleID'].values[0], negative_row['imageName'].values[0]
        #
        #     print(car_A, name_A)
        #     print(car_P, name_P)
        #     print(car_N, name_N)
        #
        #     path_test = config.VERI_DATASET + 'train_label.xml'
        #
        #     gen_val = generator.MyGenerator(path_test, "image_train/", config.IMAGES, config.IMAGES)
        #
        #     anchor, positive, negative = load_i(dirc + name_A), load_i(dirc + name_P), load_i(dirc + name_N)
        #
        #     anchor_embedding, positive_embedding, negative_embedding = (
        #         embedding(
        #             keras.applications.resnet50.preprocess_input(np.array([anchor]), data_format='channels_last')),
        #         embedding(
        #             keras.applications.resnet50.preprocess_input(np.array([positive]), data_format='channels_last')),
        #         embedding(
        #             keras.applications.resnet50.preprocess_input(np.array([negative]), data_format='channels_last')),
        #     )
        #
        #     cosine_similarity = metrics.CosineSimilarity()
        #
        #     positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
        #     print("Positive similarity:", positive_similarity.numpy())
        #
        #     negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
        #     print("Negative similarity:", negative_similarity.numpy())
        #
        #     fig = plt.figure(figsize=(12, 5))
        #     columns = 3
        #     rows = 1
        #     fig.add_subplot(rows, columns, 1)
        #     plt.imshow(anchor)
        #     fig.add_subplot(rows, columns, 2)
        #     plt.imshow(positive)
        #     fig.add_subplot(rows, columns, 3)
        #     plt.imshow(negative)
        #     plt.show()

        path_test = config.VERI_DATASET + 'test_label.xml'
        eval(path_test)

    else:
        if(arguments.train != None):
            checkpoint = arguments.checkpoint
            print("Using checkpoint", checkpoint)
            if not os.path.exists(checkpoint):
                print("Checkpoint nenajdeny")
                exit(1)
            siamese_model.built = True
            siamese_model.load_weights(checkpoint)

        callbacks_list = create_checkpoint()

        path_train = config.VERI_DATASET + 'train_label.xml'
        path_test = config.VERI_DATASET + 'test_label.xml'
        batch = config.BATCH_SIZE
        lenitem = batch

        gen_train = generator.MyGenerator(path_train, "image_train/", batch, lenitem)
        gen_val = generator.MyGenerator(path_test, "image_test/", batch, lenitem)
        SPE = len(gen_train.Y_train) / config.EPOCHS
        print(SPE)

        model_in = None
        for i in range(1, config.EPOCHS):
            print("EPOCH:" + str(i) + "/" + str(config.EPOCHS))

            if (i != 1):
                model_in = embedding

            siamese_model.fit(gen_train.LocalSet(),
                              steps_per_epoch=config.SPE,
                              validation_data=gen_val.LocalSet(),
                              epochs=1,
                              validation_steps=config.VSTEPS,
                              shuffle=False,
                              use_multiprocessing=False,
                              callbacks=callbacks_list)


if __name__ == "__main__":
    main()
