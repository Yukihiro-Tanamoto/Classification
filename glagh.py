import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(5)

img_paths = []
dir_paths = "C:/Users/r201703362ko/zemi/dataset/suzu/suzu_20190814" #画像が存在するディレクトリ
for file in sorted(glob.glob(dir_paths + "/*.jpg")):
    img_paths.append(file)

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
ipca = IncrementalPCA(n_components=2)
 
for i in range(n//batch_size):
    r_dataset = ipca.partial_fit(dataset[i*batch_size:(i+1)*batch_size])
    
r_dataset = ipca.transform(dataset)
print(r_dataset.shape)
print("PCA done.")

# K-means clustering
import shutil
n_clusters_10 = 7
kmeans_10 = KMeans(n_clusters=n_clusters_10, random_state=5).fit(r_dataset)
labels_10 = kmeans_10.labels_
print("K-means clustering done.")
# print(r_dataset)
# make_dir = input("新規作成するディレクトリ:")
# for i in range(n_clusters_10):
#     label_10= np.where(labels_10==i)[0]
 
#     # Image placing
#     if not os.path.exists(make_dir + "/label" + str(i)):
#         os.makedirs(make_dir + "/label" + str(i))
        
#     for j in label_10:
#         img = Image.open(img_paths[j])
#         fname = img_paths[j].split('\\')[-1]
#         #print(fname)
#         #shutil.copy(dir_paths+"/"+fname, make_dir+"/label"+str(i))
# centers_k = kmeans_10.cluster_centers_ 
# df = pd.read_excel('C:/Users/r201703362ko/zemi/.xlsx/labels.xlsx')
# state = df["state"]
# o = 0
# c = 0
# n = 0
# for i, r in enumerate(r_dataset):
#     if state[i] == "open":
#         plt.scatter(r[0], r[1], c="red")
#         o += 1
#     elif state[i] == "close":
#         plt.scatter(r[0], r[1], c="blue")
#         c += 1
#     else:
#         plt.scatter(r[0], r[1], c="green")
#         n += 1
    # elif labels_10[i] == 3:
    #     plt.scatter(r[0], r[1], c="cyan")
    # elif labels_10[i] == 4:
    #     plt.scatter(r[0], r[1], c="magenta")
    # elif labels_10[i] == 5:
    #     plt.scatter(r[0], r[1], c="yellow")
    # elif labels_10[i] == 6:
    #     plt.scatter(r[0], r[1], c="black")
# for c in centers_k:
#     plt.scatter(c[0], c[1], c="black")
for i, data in enumerate(r_dataset):
    if labels_10[i] == 0:
        plt.scatter(data[0], data[1], c="blue")
    elif labels_10[i] == 1:
        plt.scatter(data[0], data[1], c="green")
    elif labels_10[i] == 2:
        plt.scatter(data[0], data[1], c="steelblue")
    elif labels_10[i] == 3:
        plt.scatter(data[0], data[1], c="cyan")
    elif labels_10[i] == 4:
        plt.scatter(data[0], data[1], c="magenta")
    elif labels_10[i] == 5:
        plt.scatter(data[0], data[1], c="red")
    elif labels_10[i] == 6:
        plt.scatter(data[0], data[1], c="yellowgreen")

plt.show()
#print("open:", o, "close:", c, "night:", n)
print("Image placing done.")