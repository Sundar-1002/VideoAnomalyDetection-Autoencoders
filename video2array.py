import numpy as np
import cv2 as cv
import os   
from keras.preprocessing.image import img_to_array, load_img

img_data = []
train_path = './Avenue Dataset/training_videos/'
train_videos = os.listdir(train_path)
train_img_path = './Avenue Dataset/training-frames/'

if not os.path.exists(train_img_path):
    os.makedirs(train_img_path)

def data_store(image_path):
    img = load_img(image_path)
    img = img_to_array(img)
    img = cv.resize(img, (227,227), interpolation=cv.INTER_CUBIC)
    gray =  0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    img_data.append(gray)


for video in train_videos:
    video_path = os.path.join(train_path, video)
    cap = cv.VideoCapture(video_path)
    isTrue, frame = cap.read()
    count = 0

    while isTrue:
        frame_path = os.path.join(train_img_path, f"{count:03d}.jpg")
        cv.imwrite(frame_path, frame)
        isTrue, frame = cap.read()
        count += 1

    cap.release()

    images = os.listdir(train_img_path)

    for img in images:
        image_path = os.path.join(train_img_path, img)
        data_store(image_path)

image_data = np.array(img_data)
a,b,c = image_data.shape
image_data = image_data.reshape(b,c,a)
image_data = (image_data - image_data.mean()) / (image_data.std())
image_data = np.clip(image_data, 0 , 1)

np.save('./Avenue Dataset/training_data.npy', image_data)