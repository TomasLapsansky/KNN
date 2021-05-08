import random
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import config
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import metrics


class MyGenerator():

    def __init__(self, path_xml, mydir, batch, lenitem):
        self.df = self.loadXML(path_xml)
        self.batch = batch
        self.lenItem = lenitem
        self.anchor = []
        self.positive = []
        self.negative = []
        self.Y_train = []
        self.mydir = mydir

        self.datagen = ImageDataGenerator(width_shift_range=0.05,
                                          height_shift_range=0.05,
                                          zoom_range=0.1,
                                          vertical_flip=True,
                                          )

    def loadXML(self, path):
        print('loading label xml...', end="")
        xml_data = open(path, 'r').read()  # Read file
        root = ET.XML(xml_data)

        data = []
        items = root[0]

        for i, item in enumerate(items):
            data.append([item.attrib['imageName'], item.attrib['vehicleID'],item.attrib['colorID']])

        df = pd.DataFrame(data)  # Write in DF and transpose it
        df.columns = ["imageName", "vehicleID", "colorID"]  # Update column names
        print("DONE")
        return df

    def load_img(self, p2f):

        height, width, _ = config.INPUT_SHAPE

        t_image = cv2.imread(p2f)
        t_image = cv2.resize(t_image, (height, width))
        t_image = t_image.astype("float32")
        t_image = keras.applications.resnet50.preprocess_input(t_image, data_format='channels_last')
        return t_image

    def LocalSet(self):

        dirc = config.VERI_DATASET + self.mydir
        while True:
            anchor = []
            positive = []
            negative = []
            for i in range(self.lenItem):
                # Load random sample from data frame
                row = self.df.sample()

                # Load car ID and image name for anchor
                car_A, name_A, color, = row['vehicleID'].values[0], row['imageName'].values[0], row['colorID'].values[0]

                

                # Load car ID and image name for positive
                positive_row = (self.df.loc[self.df['vehicleID'] == car_A]).sample()
                car_P, name_P = positive_row['vehicleID'].values[0], positive_row['imageName'].values[0]



                negative_row = (self.df.loc[self.df['vehicleID'] != car_A])
                
                if(random.random()<0.5):
                    # Load car ID and image name for negative
                    negative_row_color = (negative_row.loc[negative_row['colorID'] != color]).sample()
                else:
                    # Load car ID and image name for negative with same color of car
                    negative_row_color = (negative_row.loc[negative_row['colorID'] == color]).sample()
                
                car_N, name_N = negative_row_color['vehicleID'].values[0], negative_row_color['imageName'].values[0]


                img_A = self.load_img(dirc + name_A)
                img_P = self.load_img(dirc + name_P)
                img_N = self.load_img(dirc + name_N)

                anchor.append(img_A)
                positive.append(img_P)
                negative.append(img_N)

            self.anchor = np.array(anchor)
            self.positive = np.array(positive)
            self.negative = np.array(negative)

            yield [self.anchor, self.positive, self.negative]


    def miningGen(self,embedding=None):

        dirc = config.VERI_DATASET + self.mydir
        while True:
            anchor = []
            positive = []
            negative = []

            for i in range(self.lenItem):
                # Load random sample from data frame
                row = self.df.sample()

                # Load car ID and image name for anchor
                car_A, name_A, color, = row['vehicleID'].values[0], row['imageName'].values[0], row['colorID'].values[0]

                

                # Load car ID and image name for positive
                positive_row = (self.df.loc[self.df['vehicleID'] == car_A]).sample()
                car_P, name_P = positive_row['vehicleID'].values[0], positive_row['imageName'].values[0]


                if(embedding== None):
                    negative_row = (self.df.loc[self.df['vehicleID'] != car_A])
                    # Load car ID and image name for negative
                    negative_row = (negative_row.loc[negative_row['colorID'] != color]).sample()
                    car_N, name_N = negative_row['vehicleID'].values[0], negative_row['imageName'].values[0]

                    img_A = self.load_img(dirc + name_A)
                    img_P = self.load_img(dirc + name_P)
                    img_N = self.load_img(dirc + name_N)

                    anchor.append(img_A)
                    positive.append(img_P)
                    negative.append(img_N)

                else:

                    img_A = self.load_img(dirc + name_A)
                    positive = self.load_img(dirc + name_P)

                    negat_sim = 0
                    margin = 0.5

                    list_emb = []

                    for i in range(0, config.MINI_BATCH_SIZE):
                        negative_row = (self.df.loc[self.df['vehicleID'] != car_A])
                        car_N, name_N = negative_row['vehicleID'].values[0], negative_row['imageName'].values[0]
                        negative_i = self.load_img(dirc + name_N)
                        
                        anchor_embedding, positive_embedding, negative_embedding = (
                            embedding(img_A),
                            embedding(img_P),
                            embedding(negative_i),
                        )

                        cosine_similarity = metrics.CosineSimilarity()

                        negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
                        negat_sim = negative_similarity.numpy()

                        positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
                        posit_sim = positive_similarity.numpy()

                        out_sim = posit_sim - negat_sim
                        list_emb.append([out_sim,name_N])

                    img_N = self.load_img(dirc + min(list_emb,key=lambda item:item[0])[1])

                    anchor.append(img_A)
                    positive.append(img_P)
                    negative.append(img_N)

            self.anchor = np.array(anchor)
            self.positive = np.array(positive)
            self.negative = np.array(negative)

            yield [self.anchor, self.positive, self.negative]
