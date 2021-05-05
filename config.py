import os

INPUT_SHAPE = (224, 224, 3)

EPOCHS = 50
BATCH_SIZE = 64
SPE = 100
VSTEPS = 20

if(os.path.exists('/scratch.ssd/xlapsa00/job_7468248.meta-pbs.metacentrum.cz' + '/dataset/VeRi/')):
    VERI_DATASET = '/scratch.ssd/xlapsa00/job_7468248.meta-pbs.metacentrum.cz' + '/dataset/VeRi/'
else:
    VERI_DATASET = './dataset/VeRi/'



AIC_DATASET = './dataset/AIC21_Track2_ReID/'
