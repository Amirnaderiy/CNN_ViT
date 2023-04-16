# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:53:54 2023

@author: Asus
"""
from vit_keras import vit, utils
import tensorflow as tf
import os , shutil
import cv2
from keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import SGD , Adam
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import math


original_dataset_dir = 'path/to/original_dataset_dir'

# Define the four groups based on the filename
group0 = [filename for filename in os.listdir(original_dataset_dir) if filename.startswith('Group0')]
group1 = [filename for filename in os.listdir(original_dataset_dir) if filename.startswith('Group1')]
group2 = [filename for filename in os.listdir(original_dataset_dir) if filename.startswith('Group2')]
group3 = [filename for filename in os.listdir(original_dataset_dir) if filename.startswith('Group3')]

# Define a function to load the images and resize them to a specific size
def load_and_resize_image(filename, target_size):
    img = cv2.imread(os.path.join(original_dataset_dir, filename))
    img = cv2.resize(img, target_size)
    return img

# Define the target size for the images
target_size = (224, 224)

# Load the images for each group and resize them
group0_images = [load_and_resize_image(filename, target_size) for filename in group0]
group1_images = [load_and_resize_image(filename, target_size) for filename in group1]
group2_images = [load_and_resize_image(filename, target_size) for filename in group2]
group3_images = [load_and_resize_image(filename, target_size) for filename in group3]

# Define the labels for each group
group0_labels = [0] * len(group0_images)
group1_labels = [1] * len(group1_images)
group2_labels = [2] * len(group2_images)
group3_labels = [3] * len(group3_images)

image_size = 224
num_classes=4
# Load the pre-trained VIT model with imagenet weights
vit_model = vit.vit_l32(
    image_size=image_size,
    activation='sigmoid',
    include_top=True,
    pretrained=True,
    classes=1000)

# Remove the last layer
vit_model.layers.pop()

# Add your own fully connected layers
model = models.Sequential()
model.add(vit_model)
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))



train_dir = 'path/to/train_dir'
validation_dir = 'path/to/validation_dir'
test_dir = 'path/to/test_dir'

# Define the batch size for training and validation
batch_size = 32

# Define the number of classes
num_classes = 4

# Define the data generators for training, validation, and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.5, 1.5],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)



test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    shuffle=True,
    batch_size=32,
    class_mode='categorical'
)

validation_generator=val_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size= 16 ,
    class_mode='categorical'
)
    
    

def step_decay(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 50
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(step_decay)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=["accuracy"])


    
history = model.fit(train_generator, 
                    steps_per_epoch=len(train_generator), 
                    epochs=250, 
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator),
                    callbacks=[lr_scheduler])

    
history_dict=history.history
accuracy_values=history_dict['accuracy']
val_accuracy_values=history_dict['val_accuracy']
epochs=range(1,len(accuracy_values)+1)
plt.plot (epochs,accuracy_values,'bo',label='Training Accuracy')
plt.plot (epochs,val_accuracy_values,'r',label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show
history_dict=history.history
accuracy_values=history_dict['loss']
val_accuracy_values=history_dict['val_loss']
epochs=range(1,len(accuracy_values)+1)
plt.plot (epochs,accuracy_values,'b*',label='Training Loss')
plt.plot (epochs,val_accuracy_values,'r',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show