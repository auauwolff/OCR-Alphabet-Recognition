# -*- coding: utf-8 -*-
"""
Project: Assessment 3 
Author: Felipe Wolff - 23785092
Description: Image Segmentation - Assessment 3.

Dataset Specifications***
CSV (combined labels and images)
Each row is a separate image
785 columns
Each column after represents one pixel value (784 total for a 28 x 28 image)
"""
################## Imports #####################
import sys
import cv2
import pathlib
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Conv2D
import matplotlib.pyplot as plt
from keras.layers import Flatten
from sklearn.utils import shuffle
from keras.optimizers import Adam
from skimage.filters import sobel
from keras.layers import MaxPool2D
from keras.models import Sequential
from keras.utils import to_categorical
from skimage.segmentation import watershed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


################## Read Image #####################
#### Read original Image
img = cv2.imread('letters.jpg')
#### Try with a different image!
# img = cv2.imread('demo.png')

#### Make a copy of the original image
inputCopy = img.copy()

################## Pre-processing #####################
print("\nPre-processing starts...")
print("1-Converting image to grayscale")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print("2-Applying blur")
gaussian_blur = cv2.GaussianBlur(gray,(5,5),0)
print("3-Applying threshold")
_,thresh_otsu = cv2.threshold(gaussian_blur,127,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print("4-Applying elevation map")
elevation_map_otsu_blur = sobel(thresh_otsu)
print("5-Applying segmentation")
markers = np.zeros_like(thresh_otsu)
markers[thresh_otsu < 30] = 1
markers[thresh_otsu > 150] = 2
segmentation_otsu_blur = watershed(elevation_map_otsu_blur, markers)
segmentation_normalised = cv2.normalize(src=segmentation_otsu_blur, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
print("\nPre-processing is Done!")

#### Show image processed from original to final
# titles = ['Original Image', 'Gray', 'Gaussian Blur', 'Threshhold','Elevation Map', 'Segmentation']
# images = [img, gray, gaussian_blur, thresh_otsu, elevation_map_otsu_blur, segmentation_otsu_blur]
 
# for i in np.arange(len(images)):
#     plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

################## Crop Image #####################
print("\nCropping image starts...")
#### Find countours from thresholded image
contours, hierarchy = cv2.findContours(segmentation_normalised, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

#### Check for outer contours
contours_poly = [None] * len(contours)
boundRect = []
for i, c in enumerate(contours):
    if hierarchy[0][i][3] == -1:
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect.append(cv2.boundingRect(contours_poly[i]))


#### Draw retangles around the letters in the copied image
# for i in range(len(boundRect)):
#     color = (0, 255, 0)
#     cv2.rectangle(inputCopy, (int(boundRect[i][0]), int(boundRect[i][1])), \
#                   (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)

#### Show the copied image with retangles ready to be cropped
# cv2.imshow('Image ready to be cropped', inputCopy)

#### Crop the characteres from the image
cropped_list = []    
for i in range(len(boundRect)):
    #### Offset the image so is more in the center
    dimension_offset = 5
    position_offset = 2
    x, y, w, h = boundRect[i]
    h = h+dimension_offset
    w = w+dimension_offset
    x = x-position_offset
    y = y-position_offset

    #### Crop
    croppedImg = segmentation_normalised[y:y + h, x:x + w]
    cropped_list.append(croppedImg)
    #### Show each character cropped
    # cv2.imshow("Cropped Character: "+str(i), croppedImg)
    
print("\nCropping is Done!")

#### Show 10 samples of the characters cropped from cropped_list
# plt.figure(figsize=(10,8))
# for i in range(10):  
#     plt.subplot(1, 10, i+1)
#     plt.imshow(cropped_list[i],cmap=plt.cm.gray)
#     plt.axis('off')

################## Model #####################
################## Read data file #####################
path = str(pathlib.Path(__file__).resolve().parent) + "/letters/"
sys.path.append(path)

print("\nReading files located in " + path)
test = pd.read_csv(path + 'emnist-letters-test.csv').astype('float32')
train = pd.read_csv(path + 'emnist-letters-train.csv').astype('float32')

print("\nCurrent shape of the test data:")
rows, columns = test.shape
print( str(rows) + " rows and " + str(columns) + " columns")
print("Current shape of the train data:")
rows, columns = train.shape
print( str(rows) + " rows and " + str(columns) + " columns")

################## Split data #####################
X_train = np.array(train.iloc[:,1:].values) ### images
y_train = np.array(train.iloc[:,0].values) ### labels

X_test = np.array(test.iloc[:,1:].values) ### images
y_test = np.array(test.iloc[:,0].values)  ### labels

X_train = np.reshape(X_train, (X_train.shape[0], 28,28))
X_test = np.reshape(X_test, (X_test.shape[0], 28,28))

#### Dictionary for prediction
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

###### Show 9 Training dataset samples
# fig, ax = plt.subplots(3,3, figsize = (10,10))
# axes = ax.flatten()
# for i in range(9):
#     _, shu = cv2.threshold(X_train[i], 30, 200, cv2.THRESH_BINARY)
#     axes[i].imshow(np.reshape(X_train[i], (28,28)), cmap="Greys")
# plt.show()

################## Reshape data #####################
print("\nReshaping data to fit in the model...")
train_X = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
test_X = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],1)

#### Converting labels to categorical
train_y = to_categorical(y_train, num_classes = 37, dtype='int')
test_y = to_categorical(y_test, num_classes = 37, dtype='int')

print("\nCurrent shape of the test data:", test_X.shape)
print("Current shape of the train data:", train_X.shape)
print("\nReshaping is Done!")


################## Building Model #####################
# print("\nCNN Model processing starts...")
# model = Sequential()

# model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
# model.add(MaxPool2D(pool_size=(2, 2), strides=2))

# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
# model.add(MaxPool2D(pool_size=(2, 2), strides=2))

# model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
# model.add(MaxPool2D(pool_size=(2, 2), strides=2))

# model.add(Flatten())

# model.add(Dense(64,activation ="relu"))
# model.add(Dense(128,activation ="relu"))

# model.add(Dense(37,activation ="softmax"))

# model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
# early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

# history = model.fit(train_X, train_y, epochs=3, callbacks=[reduce_lr, early_stop],  validation_data = (test_X,test_y))
# print("\nModel is Done!")

# print("\nStats:")
# print("The validation accuracy is :", history.history['val_accuracy'])
# print("The training accuracy is :", history.history['accuracy'])
# print("The validation loss is :", history.history['val_loss'])
# print("The training loss is :", history.history['loss'])

################## Save model #####################
# model.save(r'letter_model.h5')

################## Load model #####################
from tensorflow.keras.models import load_model
print("\nLoading OCR model...")
model = load_model("letter_model.h5")
print("\nLoading is Done!")

# ################## Test prediction with cropped letters #####################
print("\nPredicting...")
fig, axes = plt.subplots(3,6, figsize=(50,50))
axes = axes.flatten()
for i,ax in enumerate(axes):
    img_final = cv2.resize(cropped_list[i], (28,28))
    ax.imshow(img_final, cmap="Greys")
    img_final = np.reshape(img_final, (1,28,28,1))
    img_pred = word_dict[np.argmax(model.predict(img_final))]
    ax.set_title("Prediction: "+img_pred)
    ax.grid()    
    
print("\nPrediction is Done!")

################## Test prediction with same dataset #####################
# print("\nPredicting...")
# pred = model.predict(test_X[:9])
# fig, axes = plt.subplots(3,3, figsize=(8,9))
# axes = axes.flatten()
# for i,ax in enumerate(axes):
#     img = np.reshape(test_X[i], (28,28))
#     ax.imshow(img, cmap="Greys")
    
#     pred = word_dict[np.argmax(test_y[i])]
#     ax.set_title("Prediction: "+pred)
#     ax.grid()    
# print("\nPrediction is Done!")








