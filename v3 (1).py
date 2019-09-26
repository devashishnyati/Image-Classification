import pandas as pd
import numpy as np
#from IPython.display import Image
import glob
#import io
#import matplotlib.pyplot as plt
import os
from imutils import paths
import imutils
from skimage import feature
import cv2
import time
from sklearn.decomposition import PCA


print('Welcome to image classification')
p_time = time.time()
#smalldata = "/data/cmpe255-sp19/data/pr2/traffic-small/train/"

#labellistfile = "/data/cmpe255-sp19/data/pr2/traffic-small/train.labels"

def label_dataframe(label, file):
    curr_time = time.time()
    print("--- %s seconds ---" % (time.time() - curr_time))
    labeldf = pd.read_csv(label, header=None)
    labeldf = labeldf.rename(index=str, columns={0: "label"})
    image_list = sorted(os.listdir(file))
    labeldf['name'] = image_list
    print("--- %s seconds ---" % (time.time() - curr_time))
    return labeldf

def hog(file):
    s_time = time.time()
    curr_time = time.time()
    print("--- %s seconds ---" % (time.time() - s_time))
    data = []
    labels = []
    for i,imagePath in enumerate(glob.glob(file+"/*.jpg")):
        make = imagePath.split("/")[-1]
        labels.append(make)
        image = cv2.imread(imagePath)
        resized = cv2.resize(image, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        #edged = imutils.auto_canny(gray)
        H = feature.hog(gray, orientations=10, pixels_per_cell=(10, 10),cells_per_block=(1, 1), transform_sqrt=True, block_norm="L1")
        data.append(H)
        if i%1000==0:
            print(i)
            print("--- %s seconds ---" % (time.time() - curr_time))
            curr_time = time.time()
    print("--- %s seconds ---" % (time.time() - s_time))
    return labels, data
    
bigdata_train = "/data/cmpe255-sp19/data/pr2/traffic/train/"
bigdata_labels = "/data/cmpe255-sp19/data/pr2/traffic/train.labels"
bigdata_test = "/data/cmpe255-sp19/data/pr2/traffic/test/"

print('Label Train')
bigdata_label_df = label_dataframe(bigdata_labels,bigdata_train)

print('Data Train')
bigdata_labels_list, bigdata_data_list = hog(bigdata_train)

print('Data Train to Data Frame')
finaldf = pd.DataFrame()
finaldf['name'] = bigdata_labels_list
finaldf['data'] = bigdata_data_list

resultdf = pd.merge(finaldf,bigdata_label_df, on='name')

#resultdf.to_csv('resultdf.csv')

datadf = pd.DataFrame(resultdf.data.tolist())


print('Data Test')
bigdata_labels_list_test, bigdata_data_list_test = hog(bigdata_test)


finaldf_test = pd.DataFrame()
finaldf_test['name'] = bigdata_labels_list_test
finaldf_test['data'] = bigdata_data_list_test

finaldf_test = finaldf_test.sort_values('name')
#finaldf_test.to_csv('finaldf_test.csv')

datadf_test = pd.DataFrame(finaldf_test.data.tolist())



X_train = datadf
y_train = resultdf['label']

X_test = datadf_test
print('Data Mining Done!')


# pca = PCA(n_components=50)
# pca.fit(X_train)
# X_pca_train = pca.transform(X_train)

# X_pca_test = pca.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
#knn = KNeighborsClassifier(n_neighbors=3)
print('Applying Random Forest')
clf = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight="balanced")
clf.fit(X_train,y_train)
print('Predicting')
curr_time = time.time()
predictions=clf.predict(X_test)
# print('KNN Training')
# knn.fit(X_train,y_train)

# print('Predicting')
# 
# print("--- %s seconds ---" % (time.time() - curr_time))
# pred = knn.predict(X_test)
print("--- %s seconds ---" % (time.time() - curr_time))
print('Prediction Done!')

with open('result3.dat', 'w') as f:
    for item in predictions:
        f.write("%s\n" % item)
print("--- %s seconds ---" % (time.time() - p_time))
