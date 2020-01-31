#!/usr/bin/env python
# coding: utf-8
"""
@authors: sl7a19 / yg6m19 / sz1f19 (Team: TFA)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import os

# In[]: definition of function

def get_tiny_images(path):
    imagelist = os.listdir(path)
    imagelist.sort(key=lambda x: int(x[:-4]))
    N = len(imagelist)
    img_features = np.zeros((N,256), dtype = np.float32)
    i = 0
    for image in imagelist:
        if(image.endswith(".jpg")):
            img = cv2.imread(path+image, cv2.IMREAD_GRAYSCALE)   
            img = cv2.resize(img,(16,16),cv2.INTER_LINEAR)
            f = img.flatten()
            img_features[i,:] = (f - np.mean(f))/np.std(f)
            i = i + 1
    return img_features

# In[]: data path and set variables
    
train_path = './training/training/'
test_path = './testing/testing/'
label = [] 

train_feats = [] 
for folders in os.listdir(train_path):
    label = np.append(label, folders)
    if len(train_feats)==0:
        train_feats = get_tiny_images(train_path+folders+'/').copy()
    else:
        train_feats = np.append(train_feats,(get_tiny_images(train_path+folders+'/')),axis = 0)
test_feats = get_tiny_images(test_path)
name = []

allimg = os.listdir(test_path)
allimg.sort(key=lambda x: int(x[:-4]))
img_nums=len(allimg)
for i in range(img_nums):
    img_name=allimg[i]
    name = np.append(name,img_name)

# In[]: K-nearest-neighbour algorithm
    
K = np.arange(len(15))
train_labels = np.repeat(K,100)[:,np.newaxis]
test_labels = np.repeat(K,100)[:,np.newaxis]

knn = cv2.ml.KNearest_create() # KNN create
knn.train(train_feats, cv2.ml.ROW_SAMPLE, train_labels) # training model

# accuracys = []
# kk = []
# for kn in range(0,15):
#     ret,result,neighbours,dist = knn.findNearest(X_test,k=kn+1)
#     num = 0
#     for j in result:
#         if y_test[int(j)] == result[int(j)]:
#             num = num + 1
#     accuracy = num / len(result)
#     accuracys.append(accuracy)
#     kk.append(kn)
#     print('K =',kn+1,'  and  accuracy = ', '%.5f' % accuracy)

k = 8
ret,result,neighbours,dist = knn.findNearest(test_feats,k)

# In[]: make run1.txt file

num = len(result)
f = open("./run1.txt","w+")
for i in range(num):
    print(name[int(i)],'  ',label[int(result[i])])
    f.write(name[int(i)]+'   '+label[int(result[i])]+ '\n')
