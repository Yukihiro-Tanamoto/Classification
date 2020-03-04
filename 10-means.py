import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from natsort import natsorted

np.random.seed(5)

img_paths = []
dir_paths = input("画像が存在するディレクトリ名:") #画像が存在するディレクトリ
for file in sorted(glob.glob(dir_paths + "/*.jpg")):
    img_paths.append(file)
img_paths = natsorted(img_paths)

print("Image number:", len(img_paths))
print("Image list make done.")

plt.close("all")
img_paths = img_paths[:15552]
#len(img_paths)
print(len(img_paths))
 
def img_to_matrix(img):
    img_array = np.asarray(img)
 
    return img_array
 
def flatten_img(img_array):
    s = img_array.shape[0] * img_array.shape[1] * img_array.shape[2]
    img_width = img_array.reshape(1, s)
 
    return img_width[0]
 
dataset = []
for i in img_paths:
    img = Image.open(i)
 
    img = img.resize((int(480/6), int(360/6)), Image.BICUBIC)
 
    img = img_to_matrix(img)
    img = flatten_img(img)
 
    dataset.append(img)
    
dataset = np.array(dataset)
print(dataset.shape)
print("Dataset make done.")

n = dataset.shape[0]
batch_size = 180
ipca = IncrementalPCA(n_components=100)
 
for i in range(n//batch_size):
    r_dataset = ipca.partial_fit(dataset[i*batch_size:(i+1)*batch_size])
    
r_dataset = ipca.transform(dataset)
print(r_dataset.shape)
print("PCA done.")

# K-means clustering
import shutil
n_clusters_10 = 10
kmeans_10 = KMeans(n_clusters=n_clusters_10, random_state=5).fit(r_dataset)
labels_10 = kmeans_10.labels_
print("K-means clustering done.")

make_dir = input("新規作成するディレクトリ:")
for i in range(n_clusters_10):
    label_10= np.where(labels_10==i)[0]
 
    # Image placing
    if not os.path.exists(make_dir + "/label" + str(i)):
        os.makedirs(make_dir + "/label" + str(i))
        
    for j in label_10:
        img = Image.open(img_paths[j])
        fname = img_paths[j].split('\\')[-1]
        #print(fname)
        #shutil.copy(dir_paths+"/"+fname, make_dir+"/label"+str(i))
        
print("Image placing done.")

#各重心間の距離
centers_10 = kmeans_10.cluster_centers_

distance_10 = []
for i in range(len(centers_10)):
    d = []
    for j in range(len(centers_10)):
        length = np.linalg.norm(centers_10[i]-centers_10[j])
        d.append(length)
        print("x:" + str(i) + " y:" + str(j) + "→" + str(length))
    distance_10.append(d)
    print("----------------------------")

#各重心と一番近い重心
m = float('inf')
place_10 = 0
now_10 = 0
link = []
for i in distance_10:
    for j in range(len(distance_10)):
        if m > i[j] and i[j] > 0:
            m = i[j]
            place_10 = j
    print('label' + str(now_10) +' - label' + str(place_10) + '  min:' + str(m))
    sim = [now_10, place_10]
    link.append(sim)
    place_10 = 0
    m = float('inf')
    now_10 += 1


import copy
def join(list_b):
    mod = []
    for x in list_b:
        temp = []
        count = 0
        for y in list_b:
            same = list(set(x) & set(y))
            if x != y and same != []:
                diff1 = list(set(x) - set(y))
                diff2 = list(set(y) - set(x))
                same.extend(diff1)
                same.extend(diff2)
                temp = same
                mod.append(temp)
            else:
                count += 1
        if count  == len(list_b):
                mod.append(x)
    mod = sort(mod)
    return mod

def sort(list_b):
    for x in list_b:
        for y in list_b:
            if set(x) >= set(y) and x != y:
                list_b.remove(y)
    #list_b = join(list_b)
    list_b= list(map(list, set(map(tuple, list_b))))
    return list_b

com = []
com = copy.copy(link)
print(com)

com = sort(com)
while len(com) > 3:
    com = join(com)
    #com = sort(com)
    print(com)
    
for c in com:
    c.sort()
print(com)

import shutil
from distutils import dir_util
for in_list in com:
    first = in_list[0]
    later = [l for l in in_list[1:]]
    for el in later:
        dir_util.copy_tree(make_dir + '/label' + str(el), 
                           make_dir + '/label' + str(first))
        shutil.rmtree(make_dir + '/label' + str(el))

