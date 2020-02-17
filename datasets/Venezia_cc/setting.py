from easydict import EasyDict as edict

# init
__C_Venice = edict()

cfg_data = __C_Venice

__C_Venice.STD_SIZE = (1000,1504) #
__C_Venice.TRAIN_SIZE = (1000, 1504) #
__C_Venice.DATA_PATH = '../ProcessedData/Venezia_cc' #               
 
__C_Venice.MEAN_STD = ([0.49964153, 0.50049936, 0.50207442],[0.23948863, 0.242029055, 0.23773752]) #

__C_Venice.LABEL_FACTOR = 1
__C_Venice.LOG_PARA = 100.

__C_Venice.RESUME_MODEL = '' # model path

__C_Venice.TRAIN_BATCH_SIZE = 1 # imgs
__C_Venice.VAL_BATCH_SIZE = 1 #


