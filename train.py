#Importing Libratries
import tensorflow as tf
import keras
from keras.preprocessing.image import img_to_array, load_img 
import numpy as np
import glob
import os
import cv2 as cv

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import imutils

# Load the preprocessed training data
training_data = np.load('./Avenue Dataset/training_data.npy')
# Ensure the number of frames is divisible by 10
frames = training_data.shape[2] # Get the number of frames
frames = frames - frames%10 # Adjust frames to be divisible by 10

training_data = training_data[:,:,:frames]# Trim the data to the adjusted number of frames
# Reshape the data to have sequences of 10 frames each
training_data = training_data.reshape(-1, 227, 227, 10)
# Expand dimensions to add channel dimension
training_data = np.expand_dims(training_data, axis=4)
target_data = training_data.copy() # For autoencoder, target is same as input

epochs = 5
batch_size = 16

# Building the model
# The model is a combination of Conv3D and ConvLSTM2D layers to capture spatiotemporal features from video frames.
#The first 2 layers are Conv3D layers which extract spatial features from the input frames.
# The next 3 layers are ConvLSTM2D layers which capture temporal dependencies across the frames.
# The last 2 layers are Conv3DTranspose layers which reconstruct the output frames from the learned features.
model=Sequential()

model.add(Conv3D(filters=128,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',input_shape=(227,227,10,1),activation='tanh'))

model.add(Conv3D(filters=64,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))

model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.4,recurrent_dropout=0.3,return_sequences=True))

model.add(ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,padding='same',dropout=0.3,return_sequences=True))

model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,return_sequences=True, padding='same',dropout=0.5))

model.add(Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))

model.add(Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='tanh'))

# Compiling the model
model.compile(optimizer='adam',loss='mean_squared_error', metrics=['accuracy'])

# Training the model
model.fit(training_data,target_data, batch_size=batch_size, epochs=epochs)

# Saving the model
model.save("./model/saved_model-ExtraLayers.keras")