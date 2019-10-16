from easydict import EasyDict as edict

# init
__C_Venice = edict()

cfg_data = __C_Venice

__C_Venice.STD_SIZE = (720,1280) #
__C_Venice.TRAIN_SIZE = (360, 640) #
__C_Venice.DATA_PATH = '../ProcessedData/Venice' #               
 
__C_Venice.MEAN_STD = ([0.53144014, 0.50784626, 0.47360169],[0.19302233, 0.18909324, 0.17572044]) #

__C_Venice.LABEL_FACTOR = 1
__C_Venice.LOG_PARA = 100.

__C_Venice.RESUME_MODEL = '' # model path

__C_Venice.TRAIN_BATCH_SIZE = 1 # imgs
__C_Venice.VAL_BATCH_SIZE = 1 #


