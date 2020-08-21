# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 00:42:16 2020

@author: LENOVO
"""
### CNN classifier

### Imports for the classifier
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout
import numpy as np
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import imutils
from PIL import Image
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
#from tensorflow.keras.layers.core import Dense, Dropout, Flatten
#from tensorflow.keras.layers.convolutional import Conv2D, MaxPooling2D

"""
### Preparing the datset and the architecture to use the keras function flow_from_directory


dataset_dir_default = 'EXERCISE_1_train_data_with_default/train_data_with_default'
files_images_with_defaults = [name for name in os.listdir(dataset_dir_default)]

dataset_dir_without_defaults = 'EXERCISE_1_train_data_without_default/train_data_without_default'
files_images_without_defaults = [name for name in os.listdir(dataset_dir_without_defaults)]

path_training_set = 'dataset/training_set'
path_test_set = 'dataset/test_set'
proportion = 0.8  ### Proportion between training and test set

### Take the training images
images_train_default = files_images_with_defaults[:int(proportion*len(files_images_with_defaults))]
images_train_without_default = files_images_without_defaults[:int(proportion * len(files_images_without_defaults))]

### Extract the test images
images_test_default = files_images_with_defaults[int(proportion*len(files_images_with_defaults)):]
images_test_without_default = files_images_without_defaults[int(proportion * len(files_images_without_defaults)):]

### Move the training images with defaults to the folder dataset/training_set/images_with_default
for i in range(0,len(images_train_default)):
    shutil.move(os.path.join(dataset_dir_default, images_train_default[i]), os.path.join("dataset/training_set/images_with_default",images_train_default[i]))
   
### Move the training images without defaults to the folder dataset/training_set/images_without_default
for i in range(0,len(images_train_without_default)):
    shutil.move(os.path.join(dataset_dir_without_defaults, images_train_without_default[i]), os.path.join("dataset/training_set/images_without_default",images_train_without_default[i]))

### Move the test images with defaults to the folder dataset/test_set/images_with_default
for i in range(0,len(images_test_default)):
    shutil.move(os.path.join(dataset_dir_default, images_test_default[i]), os.path.join("dataset/test_set/images_with_default",images_test_default[i]))
    
### Move the test images without defaults to the folder dataset/test_set/images_without_default
for i in range(0,len(images_test_without_default)):
    shutil.move(os.path.join(dataset_dir_without_defaults, images_test_without_default[i]), os.path.join("dataset/test_set/images_without_default",images_test_without_default[i]))

"""
### Prepare the training set with image aumentation with the object ImageDataGenerator from Keras 
    
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (256, 256),
                                                 batch_size = 64,
                                                class_mode = 'binary')


### Prepare the test set with ImageDataGenerator
test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (256, 256),
                                            batch_size = 64,
                                            class_mode = 'binary')



### Model creation

cnn = Sequential()

cnn.add(Conv2D(filters=32, kernel_size=4, activation="relu", input_shape=[256, 256, 3]))
cnn.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))
cnn.add(Dropout(0.2))
cnn.add(Conv2D(filters=32, kernel_size=4, activation="relu"))
cnn.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))
cnn.add(Flatten())
cnn.add(Dense(units = 128, activation = 'relu'))
cnn.add(Dense(units = 1, activation = 'sigmoid'))

### Optimizers tested
opt = Adam(learning_rate=0.1) 
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=1000,
    decay_rate=0.9)
opt2 = SGD(learning_rate=lr_schedule)
 
cnn.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy','Precision','Recall'])

cnn.summary()

batch_size = 64 
cnn.fit_generator(training_set,
                  steps_per_epoch = 880//batch_size,
                  epochs = 4,
                  validation_data = test_set,
                  validation_steps = 220//batch_size) 

### Save the model

cnn.save_weights("model.h5")
print("Saved model to disk")

### Load the pretrained model

cnn.load_weights("model.h5", by_name=False, skip_mismatch=False)

print("Loaded model from disk")


### Computation of the confusion matrix

path = 'dataset/test_set/images_with_default'
fs = [name for name in os.listdir(path)]

y_pred = []
for image_path in fs:
    image = Image.open(os.path.join(path,image_path))
    new_image = image.resize((256, 256))
    img = np.array(new_image)
    img = np.expand_dims(img,axis = 0)
    y = cnn.predict(img)
    y_pred.append(y)
    print(y)

path_d = 'dataset/test_set/images_without_default'
fsd = [name for name in os.listdir(path_d)]

for image_path in fsd:
    image = Image.open(os.path.join(path_d,image_path))
    new_image = image.resize((256, 256))
    img = np.array(new_image)
    img = np.expand_dims(img,axis = 0)
    y = cnn.predict(img)
    y_pred.append(y)
    print(y)
  
classification_threshold = 0.2
for i in range(len(y_pred)):
    if y_pred[i] < classification_threshold:  
        y_pred[i] = 0
    else:
        y_pred[i] = 1
y_pred

print(confusion_matrix(test_set.classes, y_pred))



### Object detection 

### Object detection imports


import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from imutils.object_detection import non_max_suppression


### Functions for object detection


def sliding_window(image, step, ws):
	# slide a window across the image
	for y in range(0, np.array(image).shape[0] - ws[1], step):
		for x in range(0, np.array(image).shape[1] - ws[0], step):
			yield (x, y, np.array(image)[y:y + ws[1], x:x + ws[0]])
            
def image_pyramid(image, scale=1.5, minSize=(224, 224)):
	# yield the original image
	yield image
	# keep looping over the image pyramid
	while True:
		# compute the dimensions of the next image in the pyramid
		w = int(np.array(image).shape[1] / scale)
		image = imutils.resize(np.array(image), width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image


### Parameters of the object detection algorithm
        
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (64,64)
INPUT_SIZE = (256, 256)

# Load a test image to apply the object detection

image_test = Image.open(os.path.join(path,fs[0]))
image_test = image_test.resize((256,256))
(H, W) = np.array(image_test).shape[:2]


# initialize the image pyramid
pyramid = image_pyramid(image_test, scale=PYR_SCALE, minSize=ROI_SIZE)

rois = []  ### sub-images that we are going to process
locs = [] ### Location of these sub images
preds = []  ### Prediction of the CNN on these sub-images

# loop over the image pyramid

for image in pyramid:
    scale = W/float(np.array(image).shape[1])
    pred_window = []
    print("a")
    for (x,y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = np.array(roi)
        roi2 = np.expand_dims(roi, axis = 0)
        rois.append(roi2)
        locs.append((x, y, w, h))
        pred = cnn.predict(roi2)
        pred_window.append(pred)
        preds.append(pred)

### Plotting the spots where the model thinks there is a default
        
default_detected = []
for i in range(len(locs)):
    if preds[i] < 0.30 : ### filtering the location that we estimate being a default
        default_detected.append((locs[i],preds[i]))
    

for box,pred in default_detected:
            fig,ax = plt.subplots(1)          
            rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            ax.imshow(image_test)
print("Object final detected")
plt.show()
    

   
   
   
   

