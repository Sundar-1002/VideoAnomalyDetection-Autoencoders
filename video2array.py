#Importing Libratries
import numpy as np
import cv2 as cv
import os   
from keras.preprocessing.image import img_to_array, load_img

# This img_data list will hold all the processed images. These images are nothing `but frames extracted from all the training videos. 
img_data = []
train_path = './Avenue Dataset/training_videos/'
#Get list of all training videos.
train_videos = os.listdir(train_path)
#This is a folder to temporarily store all the extracted frames from training videos. For each video, the frames stored in this folder will be overwritten.
train_img_path = './Avenue Dataset/training-frames/'

#If the path does not exist, create it.
if not os.path.exists(train_img_path):
    os.makedirs(train_img_path)

# This function is used load the image from the given path, resize it to 227x227, convert it to grayscale and append it to the img_data list.
def data_store(image_path):
    img = load_img(image_path)#Load image
    img = img_to_array(img)#Convert image to array
    img = cv.resize(img, (227,227), interpolation=cv.INTER_CUBIC)#Resize image to 227x227. Resizing too 227x227 because the it is perfectly divisible by 2 multiple times which is required for Conv3D layers.
    gray =  0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]#Convert image to grayscale using luminosity method.
    img_data.append(gray)#Append the processed image to img_data list.

# Loop through all the training videos, extract frames from each video and store them temporarily in train_img_path. Then, load each frame, process it using data_store function and append it to img_data list.
for video in train_videos:
    #Extract frames from video and store them temporarily in train_img_path
    video_path = os.path.join(train_path, video)
    cap = cv.VideoCapture(video_path)  #Capture the video
    isTrue, frame = cap.read()#Read the first frame
    count = 0#Frame counter

    #Loop to read each frame from the video
    while isTrue:
        # Save the frame to the temporary folder
        frame_path = os.path.join(train_img_path, f"{count:03d}.jpg")
        cv.imwrite(frame_path, frame)#Write the frame to the specified path
        isTrue, frame = cap.read()#Read the next frame
        count += 1#Increment frame counter

    cap.release()#Release the video capture object

    images = os.listdir(train_img_path)#Get list of all frames stored in the temporary folder

    #Process each frame using data_store function and append it to img_data list
    for img in images:
        image_path = os.path.join(train_img_path, img)
        data_store(image_path)

#Convert img_data list to numpy array and reshape it to (num_frames, height, width)
image_data = np.array(img_data)
a,b,c = image_data.shape
image_data = image_data.reshape(b,c,a)#Reshape to (num_frames, height, width) because height is b and width is c. This is done to make it compatible for training.
image_data = (image_data - image_data.mean()) / (image_data.std())#Normalize the pixel values
image_data = np.clip(image_data, 0 , 1)#Clip the pixel values to be in the range [0, 1]

np.save('./Avenue Dataset/training_data.npy', image_data)#Save the processed data as a numpy array for later use in training