import os
import random
import re

import cv2
import numpy as np
import pandas as pd

import config


def get_pair(camera, id):
    try:
        df = pd.read_csv("pairs/ground_truth_shot_" + str(camera) + ".csv")
        options = [str(camera) + "A/" + str(id)]
        pair = str(df[df.camA.str.startswith(tuple(options))]['camB'])
        return pair.split("_")[0].split('"')[1]
    except:
        return "none/none"


def create_pair(batch_size, dataset, datasetB):
    
    pathA = "capt/A"
    pathB = "capt/B"
    width, height, rest = config.INPUT_SHAPE

    while True:

        output = []
        labels = []
        i = 0
        while i < batch_size:
            # print(i, batch_size, end='\r')

            image = random.choice(dataset)
            if (image[1] == "B"):
                continue

            imagelist = image.split("_")
            pair = get_pair(int(imagelist[0][0]), int(imagelist[2]))
            if (pair == "none/none"):
                continue

            pair = pair.split("/")
            prefix = pair[0][0] + "B_id_" + pair[1]
            set_list = []

            if random.choice([True, False]):
                pat = re.compile(r"^" + prefix)
                dir_list = os.listdir(pathB)
                list_name = [i for i in dir_list if pat.match(i)]

                for file in list_name:

                    if file.startswith(prefix):
                        set_list.append(file)
                    else:
                        print("Nemam B")

                if set_list == []:
                    continue

                img1 = image
                img2 = random.choice(set_list)
                label = [1]

            else:

                img1 = image
                img2 = random.choice(datasetB)
                label = [0]

            img1 = cv2.imread(pathA + "/" + img1)
            img1 = cv2.resize(img1, (width, height))

            img2 = cv2.imread(pathB + "/" + img2)
            img2 = cv2.resize(img2, (width, height))

            # plt.figure()

            # f, axarr = plt.subplots(2,1)

            # use the created array to output your multiple images. In this case I have stacked 4 images vertically
            # axarr[0].imshow(img1,cmap="hot")
            # axarr[1].imshow(img2,cmap="hot")
            # print(label)
            # plt.show()

            labels.append(label)
            output.append([img1, img2])

            i += 1

        pairTrain = np.array(output)
        labelTrain = np.array(labels)

        yield [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:]


def create_pair_optimized(batch_size, dataset, datasetB):
    pathA = "capt/A"
    pathB = "capt/B"
    width, height, rest = config.INPUT_SHAPE

    output = []
    labels = []
    i = 0
    while i < batch_size:
        print(i, batch_size, end='\r')

        image = random.choice(dataset)
        if (image[1] == "B"):
            continue

        imagelist = image.split("_")
        pair = get_pair(int(imagelist[0][0]), int(imagelist[2]))
        if (pair == "none/none"):
            continue

        pair = pair.split("/")
        prefix = pair[0][0] + "B_id_" + pair[1]
        set_list = []

        if random.choice([True, False]):
            pat = re.compile(r"^" + prefix)
            dir_list = os.listdir(pathB)
            list_name = [i for i in dir_list if pat.match(i)]

            for file in list_name:

                if file.startswith(prefix):
                    set_list.append(file)
                else:
                    print("Nemam B")

            if set_list == []:
                continue

            img1 = image
            img2 = random.choice(set_list)
            label = [1]

        else:

            img1 = image
            img2 = random.choice(datasetB)
            label = [0]

        img1 = cv2.imread(pathA + "/" + img1)
        img1 = cv2.resize(img1, (width, height))

        img2 = cv2.imread(pathB + "/" + img2)
        img2 = cv2.resize(img2, (width, height))

        # plt.figure()

        # f, axarr = plt.subplots(2,1)

        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        # axarr[0].imshow(img1,cmap="hot")
        # axarr[1].imshow(img2,cmap="hot")
        # print(label)
        # plt.show()

        labels.append(label)
        output.append([img1, img2])

        i += 1

    return (np.array(output), np.array(labels))
