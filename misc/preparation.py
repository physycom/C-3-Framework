# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import glob
from matplotlib import pyplot as plt

# This func converts the gt from the .npy format of CACC to .csv format of SANet
def convert_gt_npy2csv(path):
    path_mat = path + "den/"
    path_img = path + "img/"
    
    for _,_,c in os.walk(path_img):
        for file in c:
            imgn = file.split('.')[0]
            mat = np.load(path_mat + imgn + ".npy")
            df = pd.DataFrame(mat)
            df.to_csv(path_mat + imgn + ".csv", sep=',', header=False, index=False)

if __name__ == '__main__':
    path = "D:/Alex/ProcessedData/venice/train/"
    root = 'D:/Alex/venice/'
    ptrain = os.path.join(root,'train_data','images')
    ptest = os.path.join(root,'test_data','images')
    sigma = 4 # std of the gaussian!!!!!
    
# This func generates the gt with fixed gaussian kernels for venice dataset
    path_sets = [ptrain,ptest]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    
    for img_path in img_paths:
        print(img_path)
        mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth'))
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0],img.shape[1]))
        gt = mat["annotation"]
        for i in range(0,len(gt)):
            if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                k[int(gt[i][1]),int(gt[i][0])]=1
        k = gaussian_filter(k,sigma)
        df = pd.DataFrame(k)
        out_path = img_path.replace('.jpg','.csv').replace('images','den').replace('test_data','test').replace('train_data','train').replace('venice','ProcessedData/venice')
        df.to_csv(out_path, sep=',', header=False, index=False)