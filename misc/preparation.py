#This script generates ground truth files in various formats and with fixed or adaptive kernels
#%%
import numpy as np
import os
import pandas as pd
import scipy
from scipy import spatial
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import glob
from matplotlib import pyplot as plt
import PIL.Image as Image
from matplotlib import cm as CM

# This func converts the gt from the .npy format to .csv format of SANet
def convert_gt_npy2csv(path):
    path_mat = path + "den/"
    path_img = path + "img/"
    
    for _,_,c in os.walk(path_img):
        for file in c:
            imgn = file.split('.')[0]
            mat = np.load(path_mat + imgn + ".npy")
            df = pd.DataFrame(mat)
            df.to_csv(path_mat + imgn + ".csv", sep=',', header=False, index=False)
            print(imgn)

# This func creates adaptive gaussian kernels
def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.column_stack((np.nonzero(gt)[1], np.nonzero(gt)[0]))
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize) # build kdtree
    distances, locations = tree.query(pts, k=4) # query kdtree

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density

#%%
#################  change root and paths to point the directory you need
#################  uncomment the section for the dataset you need
    
# ShanghaiA's ground truth with adaptive kernels
"""
root = '../ShanghaiTech/'
ptrain = os.path.join(root,'part_A_final/train_data','images')
ptest = os.path.join(root,'part_A_final/test_data','images')
path_sets = [ptrain,ptest]
img_paths = []

for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k = gaussian_filter_density(k)
    np.save(img_path.replace('.jpg','.npy').replace('images','ground_truth'), k)
"""
#%%
# ShanghaiB's ground truth with fixed kernels
"""
ptrain = os.path.join(root,'part_B_final/train_data','images')
ptest = os.path.join(root,'part_B_final/test_data','images')
path_sets = [ptrain,ptest]
img_paths = []

for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k = gaussian_filter(k,15) # sigma = 15
    np.save(img_path.replace('.jpg','.npy').replace('images','ground_truth'), k)
"""
#%%
# Venice ground truth with fixed kernels
"""
root = "../Venice/"
ptrain = os.path.join(root,'train_data','images')
ptest = os.path.join(root,'test_data','images')
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
    k = gaussian_filter(k,4) # sigma = 4
    np.save(img_path.replace('.jpg','.npy').replace('images','ground_truth'), k)
"""
#%%
# Venezia_cc ground truth with adaptive kernels
root = "../Venezia_cc/"
ptrain = os.path.join(root,'train_data','images')
ptest = os.path.join(root,'test_data','images')
path_sets = [ptrain,ptest]
img_paths = []

for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    df = pd.read_csv(img_path.replace('.jpg','.csv').replace('images','ground_truth'))
    gt = df[['X','Y']].to_numpy()
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k = gaussian_filter(k,32)
    np.save(img_path.replace('.jpg','.npy').replace('images','ground_truth'), k)
