from sklearn.ensemble import AdaBoostClassifier
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import h5py
import skimage
import pandas
import mahotas


seed = 9

data_path = "../flowers/"

flower_types = os.listdir(data_path)
if '.DS_Store' in flower_types:
    flower_types.remove('.DS_Store')
print(flower_types)

data_image_paths = []
data_images = []
data_label = []

size = 100, 100

for folder in flower_types:
    for file in os.listdir(os.path.join(data_path, folder)):
        if file.endswith("jpg"):
            data_image_paths.append(os.path.join(data_path, folder, file))
            data_images.append(cv2.resize(cv2.imread(os.path.join(data_path, folder, file)), size))
            data_label.append(folder)
        else:
            continue


merge_set = list(zip(data_images, data_label))

random.shuffle(merge_set)

data_images, data_label = zip(*merge_set)


train_img, test_img, train_label, test_label = train_test_split(
    data_images, data_label, test_size=0.3)


train_data = np.array(train_img)
train_label = np.array(train_label)
test_data = np.array(test_img)
test_label = np.array(test_label)

label_dummies = pandas.get_dummies(train_label)
train_labelidx = label_dummies.values.argmax(1)
target = train_labelidx

test_label_dummies = pandas.get_dummies(test_label)
test_labelidx = test_label_dummies.values.argmax(1)


bins = 8

# feature-descriptor-1: Hu Moments


def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture


def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram


def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


global_features = []

# loop over the training data sub-folders
for x in range(0, len(train_img)):

    image = train_img[x]
    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    # update the list of labels and feature vectors
    global_features.append(global_feature)


# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)

# save the feature vector using HDF5
h5f_data = h5py.File(
    '../flowers/output/data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(
    '../flowers/output/labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)

h5f_data = h5py.File(
    '../flowers/output/data.h5', 'r')
h5f_label = h5py.File(
    '../flowers/output/labels.h5', 'r')


global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

trainDataGlobal = np.array(global_features)
trainLabelsGlobal = np.array(global_labels)

# Train Adaboost Classifer
abc.fit(trainDataGlobal, trainLabelsGlobal)

prediction = []

for file in range(0, len(test_img)):

    image = test_img[file]
    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    # predict label of test image
    prediction.append(abc.predict(global_feature.reshape(1, -1))[0])


print("Accuracy:", metrics.accuracy_score(test_labelidx, prediction))
