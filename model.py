from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from keras.layers import Dense
from keras import Model
from keras.optimizers import Adam, SGD
from keras.layers.core import Lambda
import numpy as np
from keras import backend as K
import cv2

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


def my_evaluate(model, img_car1, img_car2):
    width, height, rest = config.INPUT_SHAPE

    # print(width,height)
    car1 = cv2.imread(img_car1)
    car1 = cv2.resize(car1, (width, height))

    car2 = cv2.imread(img_car2)
    car2 = cv2.resize(car2, (width, height))

    car1 = car1[:, :]
    car2 = car2[:, :]

    input1 = np.array([car1])
    input2 = np.array([car2])

    out = list(model.predict([input1, input2]))

    return out[0]


def main():
    # tensorflow devices (GPU) print
    # print(device_lib.list_local_devices())

    model = create_model(input_shape=config.INPUT_SHAPE)

    print("Start training ")
    print("Loading files...")

    path = config.VERI_DATASET + 'train_label.xml'
    batch = config.BATCH_SIZE
    lenitem = batch

    gen = generator.MyGenerator(path, batch, lenitem)
    model.fit(gen.localSet(),
              steps_per_epoch=config.SPE,
              epochs=config.EPOCHS,
              batch_size=config.BATCH_SIZE,
              validation_data=gen.localSet(),
              validation_steps=config.VSTEPS,
              validation_batch_size=config.BATCH_SIZE,
              shuffle=False,
              use_multiprocessing=False)


if __name__ == "__main__":
    main()
