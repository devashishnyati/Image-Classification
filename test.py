import pandas as pd
import numpy as np
from IPython.display import Image
import glob
import io
import matplotlib.pyplot as plt
import os
from imutils import paths
import imutils
from skimage import feature
import cv2
import time

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
    curr_time = time.time()
    print("--- %s seconds ---" % (time.time() - curr_time))
    data = []
    labels = []
    for imagePath in glob.glob(file+"/*.jpg"):
        make = imagePath.split("/")[-1]
        labels.append(make)
        image = cv2.imread(imagePath)
        resized = cv2.resize(image, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edged = imutils.auto_canny(gray)
        H = feature.hog(edged, orientations=9, pixels_per_cell=(10, 10),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
        data.append(edged)
    print("--- %s seconds ---" % (time.time() - curr_time))
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

datadf = pd.DataFrame(resultdf.data.tolist())


print('Data Test')
bigdata_labels_list_test, bigdata_data_list_test = hog(bigdata_test)


finaldf_test = pd.DataFrame()
finaldf_test['name'] = bigdata_labels_list_test
finaldf_test['data'] = bigdata_data_list_test

finaldf_test = finaldf_test.sort_values('name')

datadf_test = pd.DataFrame(finaldf_test.data.tolist())



X_train = datadf
y_train = resultdf['label']

X_test = datadf_test




from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

print('KNN Training')
knn.fit(X_train,y_train)

print('Predicting')
pred = knn.predict(X_test)

with open('result1.dat', 'w') as f:
    for item in pred:
        f.write("%s\n" % item)