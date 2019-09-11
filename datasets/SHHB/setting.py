from easydict import EasyDict as edict

# init
__C_SHHB = edict()

cfg_data = __C_SHHB

__C_SHHB.STD_SIZE = (768,1024)
__C_SHHB.TRAIN_SIZE = (384, 512) #(576,768) ###
__C_SHHB.DATA_PATH = '../ProcessedData/shanghaitech_part_B'               

__C_SHHB.MEAN_STD = ([0.45163909, 0.44693739, 0.43153589],[0.23758833, 0.22964933, 0.2262614]) ###

__C_SHHB.LABEL_FACTOR = 1
__C_SHHB.LOG_PARA = 100.

__C_SHHB.RESUME_MODEL = ''#model path
__C_SHHB.TRAIN_BATCH_SIZE = 6 #imgs

__C_SHHB.VAL_BATCH_SIZE = 6 #


