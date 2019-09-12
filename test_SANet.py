from matplotlib import pyplot as plt

import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd
import numpy as np

from models.M2TCC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

exp_name = '../SHHB_results'
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

if not os.path.exists(exp_name+'/pred'):
    os.mkdir(exp_name+'/pred')

if not os.path.exists(exp_name+'/gt'):
    os.mkdir(exp_name+'/gt')

mean_std = ([0.45163909, 0.44693739, 0.43153589],[0.23758833, 0.22964933, 0.2262614])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()

dataRoot = '../ProcessedData/shanghaitech_part_B/test'

model_path = 'xxx.pth'

def main():
    
    file_list = [filename for root,dirs,filename in os.walk(dataRoot+'/img/')]                                           

    test(file_list[0], model_path)
   

def test(file_list, model_path):

    loss_1_fn = nn.MSELoss()
    from misc import pytorch_ssim
    loss_2_fn = pytorch_ssim.SSIM(window_size=11)
    net = CrowdCounter(cfg.GPU_ID,cfg.NET, loss_1_fn,loss_2_fn)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()


    f1 = plt.figure(1)

    gts = []
    preds = []

    for filename in file_list:
        print( filename )
        imgname = dataRoot + '/img/' + filename
        filename_no_ext = filename.split('.')[0]

        denname = dataRoot + '/den/' + filename_no_ext + '.csv'

        den = pd.read_csv(denname, sep=',',header=None).values
        den = den.astype(np.float32, copy=False)

        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')


        img = img_transform(img) # need to add padding to multiples of 8
        
        # SLICING THE PATCHES #################################################
        x4 = img.shape[2]   # full image
        x1 = x4 // 2        # half image
        x2 = x1 // 2        # quarter image
        x3 = x1 + x2
        y4 = img.shape[1]
        y1 = y4 // 2
        y2 = y1 // 2
        y3 = y1 + y2
        img_list = [
                img.clone()[:,  0:y1 ,  0:x1],  # to check if cloning is needed
                img.clone()[:,  0:y1 , x2:x3],
                img.clone()[:,  0:y1 , x1:x4],
                img.clone()[:, y2:y3 ,  0:x1],
                img.clone()[:, y2:y3 , x2:x3],
                img.clone()[:, y2:y3 , x1:x4],
                img.clone()[:, y1:y4 ,  0:x1],
                img.clone()[:, y1:y4 , x2:x3],
                img.clone()[:, y1:y4 , x1:x4]
                ]
        
        gt = np.sum(den)
        gts.append(gt)
        pred_maps = []
        
        # NET FORWARD #########################################################
        for inputs in img_list:
            with torch.no_grad():
                img = Variable(img_list[0][None,:,:,:]).cuda()
                pred_maps.append(net.test_forward(img))
                
        pred_map = pred_maps[0].new_empty((1,1,y4,x4))
        
        # GETTING DENSITY FROM NEAREST CENTER PATCH ###########################
        x3 = int(x4 * 3/8)
        x5 = int(x4 * 5/8)
        y3 = int(y4 * 3/8)
        y5 = int(y4 * 5/8)
        for y in range(y3):
            for x in range(x3):
                pred_map[0,0,y,x] = pred_maps[0][0,0,y,x]
            for x in range(x3,x5):
                pred_map[0,0,y,x] = pred_maps[1][0,0,y,x-x2]
            for x in range(x5,x4):
                pred_map[0,0,y,x] = pred_maps[2][0,0,y,x-x1]
        for y in range(y3,y5):
            for x in range(x3):
                pred_map[0,0,y,x] = pred_maps[3][0,0,y-y2,x]
            for x in range(x3,x5):
                pred_map[0,0,y,x] = pred_maps[4][0,0,y-y2,x-x2]
            for x in range(x5,x4):
                pred_map[0,0,y,x] = pred_maps[5][0,0,y-y2,x-x1]
        for y in range(y5,y4):
            for x in range(x3):
                pred_map[0,0,y,x] = pred_maps[6][0,0,y-y1,x]
            for x in range(x3,x5):
                pred_map[0,0,y,x] = pred_maps[7][0,0,y-y1,x-x2]
            for x in range(x5,x4):
                pred_map[0,0,y,x] = pred_maps[8][0,0,y-y1,x-x1]
        
        sio.savemat(exp_name+'/pred/'+filename_no_ext+'.mat',{'data':pred_map.squeeze().cpu().numpy()/100.})
        sio.savemat(exp_name+'/gt/'+filename_no_ext+'.mat',{'data':den})

        pred_map = pred_map.cpu().data.numpy()[0,0,:,:]

        pred = np.sum(pred_map)/100.0
        preds.append(pred)
        pred_map = pred_map/np.max(pred_map+1e-20)
        
        den = den/np.max(den+1e-20)
        
        # GRAPHS ##############################################################        
        den_frame = plt.gca()
        plt.imshow(den, 'jet')
        den_frame.axes.get_yaxis().set_visible(False)
        den_frame.axes.get_xaxis().set_visible(False)
        den_frame.spines['top'].set_visible(False) 
        den_frame.spines['bottom'].set_visible(False) 
        den_frame.spines['left'].set_visible(False) 
        den_frame.spines['right'].set_visible(False) 
        plt.savefig(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)

        plt.close()
        
        # sio.savemat(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.mat',{'data':den})

        pred_frame = plt.gca()
        plt.imshow(pred_map, 'jet')
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False) 
        pred_frame.spines['bottom'].set_visible(False) 
        pred_frame.spines['left'].set_visible(False) 
        pred_frame.spines['right'].set_visible(False) 
        plt.savefig(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)

        plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.mat',{'data':pred_map})

        diff = den-pred_map

        diff_frame = plt.gca()
        plt.imshow(diff, 'jet')
        plt.colorbar()
        diff_frame.axes.get_yaxis().set_visible(False)
        diff_frame.axes.get_xaxis().set_visible(False)
        diff_frame.spines['top'].set_visible(False) 
        diff_frame.spines['bottom'].set_visible(False) 
        diff_frame.spines['left'].set_visible(False) 
        diff_frame.spines['right'].set_visible(False) 
        plt.savefig(exp_name+'/'+filename_no_ext+'_diff.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)

        plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_diff.mat',{'data':diff})
                     



if __name__ == '__main__':
    main()




