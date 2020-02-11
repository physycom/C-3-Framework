from easydict import EasyDict as edict

# init
__C_Venice = edict()

cfg_data = __C_Venice

__C_Venice.STD_SIZE = (4000,6016) #
__C_Venice.TRAIN_SIZE = (1080, 1920) #
__C_Venice.DATA_PATH = '../ProcessedData/Venezia_cc' #               
 
__C_Venice.MEAN_STD = ([0.49964153766, 0.50049936771, 0.50207442045],[0.23948863148, 0.242029055953, 0.23773752152]) #

__C_Venice.LABEL_FACTOR = 1
__C_Venice.LOG_PARA = 100.

__C_Venice.RESUME_MODEL = '' # model path

__C_Venice.TRAIN_BATCH_SIZE = 1 # imgs
__C_Venice.VAL_BATCH_SIZE = 1 #


