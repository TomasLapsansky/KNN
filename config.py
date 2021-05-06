import os
import shutil

INPUT_SHAPE = (224, 224, 3)

EPOCHS = 50
BATCH_SIZE = 64
SPE = 100
VSTEPS = 20

try:
    print("Scratchdir exists")
    scratchdir = os.environ["$SCRATCHDIR"]
    if not os.path.exists(scratchdir + '/dataset/VeRi/'):
        print("Copying dataset to scratchdir")
        os.mkdir(scratchdir + '/dataset')
        shutil.copytree('./dataset/VeRi', scratchdir + '/dataset/VeRi')
        print("Copying finished")
    VERI_DATASET = scratchdir + '/dataset/VeRi/'
except:
    print("Scratchdir doesn't exist")
    VERI_DATASET = './dataset/VeRi/'



AIC_DATASET = './dataset/AIC21_Track2_ReID/'
