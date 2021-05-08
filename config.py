import os
import shutil

INPUT_SHAPE = (224, 224, 3)

EPOCHS = 20
BATCH_SIZE = 64
SPE = 100
VSTEPS = round(SPE/2)
TESTTRAIN = 0.3
IMAGES = 3

try:
    print("Scratchdir exists")
    scratchdir = os.popen('echo $SCRATCHDIR').read()
    scratchdir = scratchdir[:-1]
    if not os.path.exists(scratchdir + '/dataset/VeRi/'):
        print("Copying dataset to scratchdir")
        os.mkdir(scratchdir + '/dataset')
        shutil.copytree('./dataset/VeRi', scratchdir + '/dataset/VeRi')
        print("Copying finished")
    else:
        print("Dataset exists")
    VERI_DATASET = scratchdir + '/dataset/VeRi/'
except:
    print("Scratchdir doesn't exist")
    VERI_DATASET = './dataset/VeRi/'

AIC_DATASET = './dataset/AIC21_Track2_ReID/'
