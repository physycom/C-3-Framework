# -*- coding: utf-8 -*-

import scipy.io as sio
import os
import numpy as np

exp_name = '../SHHB_results'
file_list_pred = [filename for root,dirs,filename in os.walk(exp_name+'/pred/')]
file_list_gt = [filename for root,dirs,filename in os.walk(exp_name+'/gt/')]

pred_list = []
for file in file_list_pred[0]:
    pred_list.append(np.sum(sio.loadmat(exp_name + '/pred/' + file)['data']))
preds=np.asarray(pred_list)

gt_list = []
for file in file_list_gt[0]:
    gt_list.append(np.sum(sio.loadmat(exp_name + '/gt/' + file)['data']))
gts= np.asarray(gt_list)

print('MAE= ' + str(np.mean(np.abs(gts-preds))))
print('MSE= ' + str(np.sqrt(np.mean((gts-preds)**2))))