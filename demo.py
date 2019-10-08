# -*- coding: utf-8 -*-
# This scripts uses SANet on a video file / video cam to estimate density map
# and crowd count and shows them overlapped to the video.

import torch
import torchvision
import numpy as np
import cv2
import argparse

from models.M2TCC import CrowdCounter
from config import cfg
from misc import pytorch_ssim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./checkpoints/venice_all_ep_1286_mae_5.9_mse_7.4.pth", type=str, help="Weights file to use.")
    parser.add_argument("--video", default="D:/Alex/venice/test_data/videos/4895.mp4", type=str, help="Video file to elapse or camera ip.")
    parser.add_argument("--opacity", default=0.7, type=float, help="Opacity value for the density map overlap.")
    parser.add_argument("--cmap", default=11, type=int, help="Colormap to use. Commons are 2 for Jet or 11 for Hot.", choices = [0,1,2,3,4,5,6,7,8,9,10,11])
    parser.add_argument("--mode", default='Add', type=str, help="Blend mode to use.", choices = ['Add','Lighten','Mix','Multiply'])
    args = parser.parse_args()
    print("Crowd Counter Demo running. Press Q to quit.")
    
    # setup the model
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.backends.cudnn.enabled = True # use cudnn?
    net = CrowdCounter(cfg.GPU_ID,cfg.NET, torch.nn.MSELoss(),pytorch_ssim.SSIM(window_size=11))
    net.load_state_dict(torch.load(args.model))
    net.cuda()
    net.eval()
    mean, std = [0.53144014, 0.50784626, 0.47360169], [0.19302233, 0.18909324, 0.17572044]

    # open the video stream / file
    cap = cv2.VideoCapture(args.video)
    while(cap.isOpened()):
        _, frame = cap.read()
        if frame is None:
            break
        # convert to pytorch tensor and normalize
        tensor = torchvision.transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = torchvision.transforms.functional.normalize(tensor, mean=mean, std=std)
        # forward propagation
        with torch.no_grad():
               tensor = torch.autograd.Variable(tensor[None,:,:,:]).cuda()
               pred_map = net.test_forward(tensor)
        pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
        # converts frame to grayscale and density map to color map, then overlap them
        gray = np.repeat(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis], 3, axis=2)
        dmap = cv2.applyColorMap(np.array(pred_map/np.max(pred_map+1e-20) * 255, dtype = np.uint8), args.cmap)
        dmap = dmap.astype(np.float32)/255
        gray = gray.astype(np.float32)/255
        if args.mode == 'Add':
            res = dmap*args.opacity + gray
        elif args.mode == 'Lighten':
            res = np.maximum(gray, dmap*args.opacity)
        elif args.mode == 'Mix':
            res = dmap*args.opacity + gray*(1-args.opacity)
        elif args.mode == 'Multiply':
            res = gray*(dmap*args.opacity+(1-args.opacity))
        else:
            raise Exception("No such blend mode found.") 
        res = (np.clip(res,0,1)*255).astype(np.uint8)
        # write total people count on image and show results
        cv2.putText(res, str(int(np.sum(pred_map)/100 +0.5)), (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Crowd Counting", res)
        # wait for escape key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
