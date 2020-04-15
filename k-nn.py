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
import random
import pandas as pd
import copy
import shutil
from distutils import dir_util
import openpyxl
import pprint

np.random.seed(5)
clusters = 3
img_paths = []
#dir_paths = input("画像が存在するディレクトリ名:") #画像が存在するディレクトリ
dir_paths = "C:/Users/r201703362ko/zemi/dataset/suzu/suzu_20190814"
#dir_paths = "C:/Users/r201703362ko/zemi/dataset/suzu/suzu_20190814"
for file in sorted(glob.glob(dir_paths + "/*.jpg")):
    img_paths.append(file)
img_paths = natsorted(img_paths)

print("Image number:", len(img_paths))
print("Image list make done.")
num = [] #何番目にあるか
df = pd.read_excel('C:/Users/r201703362ko/zemi/.xlsx/labels_suzu.xlsx')
state = df["state"]
#print(state)

    
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
#n_clusters_k = 7
kmeans = KMeans(n_clusters=clusters, random_state=5).fit(r_dataset)
labels_k = kmeans.labels_
print("K-means clustering done.")
iter = kmeans.n_iter_
print ("inter:", iter)
# make_dir = input("新規作成するディレクトリ:")
# for i in range(n_clusters_k):
#     label_k = np.where(labels_k==i)[0]

#     # Image placing
#     if not os.path.exists(make_dir + "/label" + str(i)):
#         os.makedirs(make_dir + "/label" + str(i))
        
#     for j in label_k:
#         img = Image.open(random_paths[j])
#         fname = random_paths[j].split('\\')[-1]
#         #print(fname)
#         shutil.copy(dir_paths+"/"+fname, make_dir+"/label"+str(i))
        
# print("Image placing done.")
centers_k = kmeans.cluster_centers_ #重心

distance = []
for data in r_dataset:
    d = []
    for c in centers_k:
        length = np.linalg.norm(data-c)
        d.append(length)
    distance.append(d)


# for i, k in enumerate(centers_k):
#     if i == 0:
#         plt.scatter(k[0], k[1], c="blue")
#     elif i == 1:
#         plt.scatter(k[0], k[1], c="red")
#     else:
#         plt.scatter(k[0], k[1], c="green")
# plt.show()
wb = openpyxl.load_workbook('C:/Users/r201703362ko/zemi/.xlsx/suzu_distance.xlsx')
sheet = wb.active
for i, k in enumerate(distance):
    for j in range(clusters):
        sheet.cell(row = i+1, column = j+1, value = k[j])
wb.save(filename = 'C:/Users/r201703362ko/zemi/.xlsx/suzu_distance.xlsx')
#print(distance)