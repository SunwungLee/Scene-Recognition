#!/usr/bin/env python
# coding: utf-8
"""
@authors: sz1f19 / sl7a19 / yg6m19 (Team: TFA)
"""

from os.path import exists, isdir, basename, join, splitext
from glob import glob
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq
import numpy as np
import os
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# In[]: definitions of functions

def get_categories(path):
    folders = [files for files in glob(path + "/*") if isdir(files)]
    categories = [basename(folders) for folders in folders]
    return categories
def get_bags_of_words(image, size):
    m,n = image.shape
    img_features = []
    for i in range((m-size)//4):
        for j in range((n-size)//4):
            img_feature = image[4*i:4*i+size,4*j:4*j+size]
            f = img_feature.flatten()
            temp = np.zeros(64)
            np.divide((f - np.mean(f)), np.std(f), out=temp, where=(f - np.mean(f))!= 0)
            img_features.append(temp)
    return np.array(img_features)

# In[]: data path and set variables

train_path = './training/training'
test_path = './testing/testing'
categories = get_categories(train_path)
print(categories)
ImageSet = {}
train_set = []

for name in categories:
    files = os.listdir(train_path + "/" + name)
    files.sort(key= lambda x:int(x[:-4]))
    ImageSet[name] = len(files)
    for filename in files:
        if filename != '.DS_Store':
            image_path = os.path.join(train_path + "/" + name,filename)
            train_set += [image_path]
training_set = np.array(train_set)

# In[]: bags of words

test_set = []
files_test = os.listdir(test_path)
files_test.sort(key= lambda x:int(x[:-4]))
for filename in files_test:
    image_path = os.path.join(test_path,filename)
    test_set += [image_path]
test_set = np.array(test_set)

K = np.arange(len(categories))
train_labels = np.repeat(K,100)[:,np.newaxis]

#training_set, test_set, train_labels, test_labels = train_test_split(train_set, train_labels, test_size = 0.3)

features_list=[]
for i in range(len(training_set)):
    print(training_set[i])
    img=  Image.open(training_set[i]).convert('L')
    features = get_bags_of_words(np.array(img), 8)
    features_list.append((i, features))

# In[]: K means 
    
print("Kmeans Starts")
training_features = features_list[0][1]

for image_path, training_feature in features_list[1:]:
    training_features = np.vstack((training_features, training_feature))  

print(training_features.shape)

clusters = 300  # set how many clusters we use
means,var = kmeans(training_features, clusters,1) # k-means

print("Kmeans Ends")
# In[]: training

im_features = np.zeros((len(training_set), clusters), "float32")
for i in range(len(training_set)):
    words, distance = vq(features_list[i][1],means)
    for w in words:
        im_features[i][w] += 1

clf = LinearSVC()
clf.fit(im_features, np.array(train_labels))

print("training Ends")
# In[]: test

test_features_list = []
for i in range(len(test_set)):
 
#    img = cv2.imread(test_set[i],cv2.IMREAD_GRAYSCALE)
    img = Image.open(test_set[i]).convert('L')
    features = get_bags_of_words(np.array(img), 8)
    test_features_list.append((i, features))    

print("test set")

# In[]: make run2.txt file

test_features = np.zeros((len(test_set), clusters), "float32")
for i in range(len(test_set)):
    words, distance = vq(test_features_list[i][1],means)
    for w in words:
        test_features[i][w] += 1
predictions =  clf.predict(test_features)
output = ''
for i in range(len(test_set)):
    output = output + files_test[i]+' ' + categories[predictions[i]]+'\n'
    
path_file_name = './run2.txt'
str_data = "1\n2"
if not os.path.exists(path_file_name):
    with open(path_file_name, "w") as f:
        print(f)
    with open(path_file_name, "a") as f:
        f.write(output)

