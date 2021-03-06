from easydict import EasyDict as edict
import time

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 3035 # random seed,  for reproduction
__C.DATASET = 'Venezia_cc' # dataset selection: GCC, SHHA, SHHB, UCF50, QNRF, WE, Venice, Venezia_cc
__C.NET = 'SANet' # net selection: MCNN, VGG, VGG_DECODER, Res50, CSRNet, SANet

__C.PRE_GCC = False # use the pretrained model on GCC dataset
__C.PRE_GCC_MODEL = '' # path to model

__C.RESUME = False # contine training
__C.RESUME_PATH = '' #

__C.GPU_ID = [0] # sigle gpu: [0], [1] ...; multi gpus: [0,1]

# learning rate settings
__C.LR = 1e-5 # learning rate 
__C.LR_DECAY = 1 # decay rate
__C.LR_DECAY_START = -1 # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1 # decay frequency
__C.MAX_EPOCH = 3000

# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on

__C.LAMBDA_1 = 1e-3


# print 
__C.PRINT_FREQ = 30

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
			 + '_' + __C.DATASET \
             + '_' + __C.NET \
             + '_' + str(__C.LR)

if __C.DATASET == 'UCF50':
	__C.EXP_NAME += '_' + str(__C.VAL_INDEX)	

if __C.DATASET == 'GCC':
	__C.EXP_NAME += '_' + __C.VAL_MODE	

__C.EXP_PATH = './exp' # the path of logs, checkpoints, and current codes


#------------------------------VAL------------------------
__C.VAL_DENSE_START = 200
__C.VAL_FREQ = 10 # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

#------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1 #  must be 1 for training images with the different sizes

#-----------------------MEAN/STD and SIZE-----------------
if __C.DATASET == 'SHHA':
    from datasets.SHHA.setting import cfg_data 
elif __C.DATASET == 'SHHB':
    from datasets.SHHB.setting import cfg_data 
elif __C.DATASET == 'Venice':
    from datasets.Venice.setting import cfg_data 
elif __C.DATASET == 'Venezia_cc':
    from datasets.Venezia_cc.setting import cfg_data
elif __C.DATASET == 'QNRF':
    from datasets.QNRF.setting import cfg_data 
elif __C.DATASET == 'UCF50':
    from datasets.UCF50.setting import cfg_data
    __C.VAL_INDEX = cfg_data.VAL_INDEX 
elif __C.DATASET == 'WE':
    from datasets.WE.setting import cfg_data 
elif __C.DATASET == 'GCC':
    from datasets.GCC.setting import cfg_data
    __C.VAL_MODE = cfg_data.VAL_MODE 
elif __C.DATASET == 'Mall':
    from datasets.Mall.setting import cfg_data
elif __C.DATASET == 'UCSD':
    from datasets.UCSD.setting import cfg_data 
else:
  raise Exception("Unknown dataset")
  
__C.MEAN_STD = cfg_data.MEAN_STD
__C.STD_SIZE = cfg_data.STD_SIZE
#================================================================================  