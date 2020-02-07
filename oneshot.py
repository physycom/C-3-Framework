#This script runs SANet on a image and shows result.

import torch
import torchvision
import numpy as np
import argparse
import cv2

from models.M2TCC import CrowdCounter
from misc import pytorch_ssim
from config import cfg

img_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(*cfg.MEAN_STD)
                ])
pil_to_tensor = torchvision.transforms.ToTensor()

def main():
    torch.cuda.set_device(0)
    torch.backends.cudnn.enabled = True
    net = CrowdCounter(cfg.GPU_ID, cfg.NET, torch.nn.MSELoss(), pytorch_ssim.SSIM(window_size=11))
    net.load_state_dict(torch.load(args.model))
    net.cuda()
    net.eval()
    h,w = cfg.STD_SIZE

    mask = None
    if args.mask != "" :
      mask = cv2.imread(args.mask, 0)      

    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    if img.shape[0]!=h or img.shape[1]!=w:
      img = cv2.resize(img, (w,h), interpolation=cv2.INTER_CUBIC)
    gray = np.repeat(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis], 3, axis=2)
    gray = gray.astype(np.float32)/255
    img = img_transform(img)
    
    if args.slicing:
        # this slices the image in 9 patches with 50% overlap, runs SANet on each patch and then takes only the values nearest to the patch center for the final prediction map
        # this may help reducing statistical errors, but runs slower than standard method. could also cause some visual artifacts
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
                              np.hstack((slice6,slice7,slice8))))

    else:
        with torch.no_grad():
           img = torch.autograd.Variable(img[None,:,:,:]).cuda()
           pred_map = net.test_forward(img)
        pred_map = pred_map.cpu().data.numpy()[0,0,:,:]

    if mask is not None:
        if mask.shape != pred_map.shape:
            mask = cv2.resize(mask, (pred_map.shape[1],pred_map.shape[0]), cv2.INTER_NEAREST)
        _,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
        pred_map = cv2.bitwise_and(pred_map, pred_map, mask = mask)

    dmap = cv2.applyColorMap(np.array(pred_map/np.max(pred_map+1e-20) * 255, dtype = np.uint8), 11)
    dmap = (cv2.cvtColor(dmap, cv2.COLOR_BGR2RGB)).astype(np.float32)/255
    res = (np.clip(dmap*args.opacity + gray,0,1)*255).astype(np.uint8)

    cv2.putText(res, str(int(np.sum(pred_map)/100 +0.5)), (10, gray.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Crowd Counting", cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
    # wait for any key
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./checkpoints/venice_all_ep_1286_mae_5.9_mse_7.4.pth", type=str, help="Weights file to use. Careful: if using different model you may need to change the trained dataset in config file.")
    parser.add_argument("--slicing", default=False, type=bool, help = "Run SANet on overlapping patches. May give better results, but slower.")
    parser.add_argument("--image", default="../ProcessedData/Venice/test/img/4895_000060.jpg", type=str, help="Image of the crowd to count.")
    parser.add_argument("--opacity", default=0.7, type=float, help="Opacity value for the density map overlap.")
    parser.add_argument("--mask", default="", type=str, help="Mask image to filter outputs with.")
    args = parser.parse_args()

    main()
