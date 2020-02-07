#This scripts tests SANet on a test dataset and gives MAE and MSE.

from matplotlib import pyplot as plt

import os
import torch
import torchvision
import pandas as pd
import numpy as np

from models.M2TCC import CrowdCounter
from misc import pytorch_ssim
from config import cfg
import scipy.io as sio
from PIL import Image

## SETUP AND PARAMETERS #######################
torch.cuda.set_device(0)
torch.backends.cudnn.enabled = True

exp_name = '../Venice_results'
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

if not os.path.exists(exp_name+'/pred'):
    os.mkdir(exp_name+'/pred')

if not os.path.exists(exp_name+'/gt'):
    os.mkdir(exp_name+'/gt')

if not os.path.exists(exp_name+'/diff'):
    os.mkdir(exp_name+'/diff')

slicing = False # use the paper test method. may give better results, but slower
save_graphs = False # save density maps images
dataRoot = '../ProcessedData/Venice/test'
model_path = './checkpoints/venice_all_ep_1286_mae_5.9_mse_7.4.pth'
mean_std = cfg.MEAN_STD
##################################################

img_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*mean_std)
    ])
pil_to_tensor = torchvision.transforms.ToTensor()

def main():
    print(dataRoot)
    file_list = [filename for root,dirs,filename in os.walk(dataRoot+'/img/')]                                           

    test(file_list[0], model_path)

def test(file_list, model_path):
    loss_1_fn = torch.nn.MSELoss()
    loss_2_fn = pytorch_ssim.SSIM(window_size=11)
    net = CrowdCounter(cfg.GPU_ID,cfg.NET, loss_1_fn,loss_2_fn)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    f1 = plt.figure(1)

    gts = []
    preds = []

    for filename in file_list:
        print( filename , end = ', ')
        imgname = dataRoot + '/img/' + filename
        filename_no_ext = filename.split('.')[0]

        denname = dataRoot + '/den/' + filename_no_ext + '.csv'

        den = pd.read_csv(denname, sep=',',header=None).values
        den = den.astype(np.float32, copy=False)

        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')

        img = img_transform(img)
        
        if slicing:
            xr = (8-img.shape[2]%8)%8
            yr = (8-img.shape[1]%8)%8
            img = torch.nn.functional.pad(img, (xr,xr,yr,yr), 'constant',0)
            pred_maps = []
            x4 = img.shape[2]   # full image
            x1 = x4 // 2        # half image
            x2 = x1 // 2        # quarter image
            x3 = x1 + x2
            y4 = img.shape[1]
            y1 = y4 // 2
            y2 = y1 // 2
            y3 = y1 + y2
            img_list = [img[:,  0:y1 ,  0:x1],img[:,  0:y1 , x2:x3],img[:,  0:y1 , x1:x4],
                        img[:, y2:y3 ,  0:x1],img[:, y2:y3 , x2:x3],img[:, y2:y3 , x1:x4],
                        img[:, y1:y4 ,  0:x1],img[:, y1:y4 , x2:x3],img[:, y1:y4 , x1:x4]]

            for inputs in img_list:
                with torch.no_grad():
                    img = torch.autograd.Variable(inputs[None,:,:,:]).cuda()
                    pred_maps.append(net.test_forward(img))
    
            x3, x5 = int(x4 * 3/8), int(x4 * 5/8)
            y3, y5 = int(y4 * 3/8), int(y4 * 5/8)
            x32, x52, x51, x41 = x3-x2, x5-x2, x5-x1, x4-x1
            y32, y52, y51, y41 = y3-y2, y5-y2, y5-y1, y4-y1
            
            slice0 = pred_maps[0].cpu().data.numpy()[0,0,0:y3,0:x3]
            slice1 = pred_maps[1].cpu().data.numpy()[0,0,0:y3,x32:x52]
            slice2 = pred_maps[2].cpu().data.numpy()[0,0,0:y3,x51:x41]
            slice3 = pred_maps[3].cpu().data.numpy()[0,0,y32:y52,0:x3]
            slice4 = pred_maps[4].cpu().data.numpy()[0,0,y32:y52,x32:x52]
            slice5 = pred_maps[5].cpu().data.numpy()[0,0,y32:y52,x51:x41]
            slice6 = pred_maps[6].cpu().data.numpy()[0,0,y51:y41,0:x3]
            slice7 = pred_maps[7].cpu().data.numpy()[0,0,y51:y41,x32:x52]
            slice8 = pred_maps[8].cpu().data.numpy()[0,0,y51:y41,x51:x41]
            
            pred_map = np.vstack((np.hstack((slice0,slice1,slice2)),
                                  np.hstack((slice3,slice4,slice5)),
                                  np.hstack((slice6,slice7,slice8))
                                ))
            sio.savemat(exp_name+'/pred/'+filename_no_ext+'.mat',{'data':pred_map/100.})

        else:
            with torch.no_grad():
               img = torch.autograd.Variable(img[None,:,:,:]).cuda()
               pred_map = net.test_forward(img)
            sio.savemat(exp_name+'/pred/'+filename_no_ext+'.mat',{'data':pred_map.squeeze().cpu().numpy()/100.})
            pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
        
        pred = np.sum(pred_map)/100.0
        preds.append(pred)

        gt = np.sum(den)
        gts.append(gt)
        sio.savemat(exp_name+'/gt/'+filename_no_ext+'.mat',{'data':den})
        pred_map = pred_map/np.max(pred_map+1e-20)
        den = den/np.max(den+1e-20)

        if save_graphs:      
            den_frame = plt.gca()
            plt.imshow(den, 'jet')
            den_frame.axes.get_yaxis().set_visible(False)
            den_frame.axes.get_xaxis().set_visible(False)
            den_frame.spines['top'].set_visible(False) 
            den_frame.spines['bottom'].set_visible(False) 
            den_frame.spines['left'].set_visible(False) 
            den_frame.spines['right'].set_visible(False) 
            plt.savefig(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.png',bbox_inches='tight',pad_inches=0,dpi=150)
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
            plt.savefig(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.png',bbox_inches='tight',pad_inches=0,dpi=150)
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
            plt.savefig(exp_name+'/'+filename_no_ext+'_diff.png',bbox_inches='tight',pad_inches=0,dpi=150)
            plt.close()
            # sio.savemat(exp_name+'/diff/'+filename_no_ext+'_diff.mat',{'data':diff})
    preds=np.asarray(preds)
    gts=np.asarray(gts)
    print('\nMAE= ' + str(np.mean(np.abs(gts-preds))))
    print('MSE= ' + str(np.sqrt(np.mean((gts-preds)**2))))

if __name__ == '__main__':
    main()


