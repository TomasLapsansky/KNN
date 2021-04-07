import tensorflow as tf
import keras
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from keras.layers import Dense

from keras import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from keras.layers.core import Lambda
from keras.regularizers import l2
import numpy as np
from keras import backend as K
import pandas as pd



from sklearn.model_selection import train_test_split

INPUT_SHAPE = (128,128,1)

def get_pair(camera, id):
    try:
        df = pd.read_csv("pairs/ground_truth_shot_" + str(camera) + ".csv")
        options = [str(camera) + "A/" + str(id)]
        pair = str(df[df.camA.str.startswith(tuple(options))]['camB'])
        return pair.split("_")[0].split('"')[1]
    except:
        return ""


def initialize_weights(shape, dtype=None):
  return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, dtype=None):
  return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)


def create_model(input_shape=(128,128,1)):
    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
                    kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu',
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu',
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu',
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   bias_initializer=initialize_bias))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model

    siamese_net.summary()

    return siamese_net


def main():
    print(get_pair(1, 16666))
    EPOCHS = 100
    BATCH_SIZE = 16
    SPE = 100
    no_output = 2
    
    model = create_model(input_shape=INPUT_SHAPE)


    

if __name__ == "__main__":
    main()
