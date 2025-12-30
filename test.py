import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import imutils

def mean_squared_loss(x1, x2):
    diff = x1 - x2
    a,b,c,d,e = diff.shape
    n_samples = a*b*c*d*e
    sq_diff = diff ** 2
    total = sq_diff.sum()
    distance = np.sqrt(total)
    mean_distance = distance / n_samples
    return mean_distance

model = load_model('./model/saved_model.keras', compile=False)

cap = cv.VideoCapture('./Avenue Dataset/testing_videos/04.avi')

print(cap.isOpened())

while cap.isOpened():
    im_frames = []
    isTrue, frame = cap.read()
    if not isTrue:
        break

    for i in range(10):
        isTrue, frame = cap.read()
        if not isTrue:
            break
        image = imutils.resize(frame, width = 700, height = 600)
        frame = cv.resize(image, (227, 227), interpolation=cv.INTER_CUBIC)
        gray = 0.2989 * frame[:,:,0] + 0.5870 * frame[:,:,1] + 0.1140 * frame[:,:,2]
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)
        im_frames.append(gray)

    im_frames = np.array(im_frames)
    im_frames = im_frames.reshape(227, 227, 10)
    im_frames = np.expand_dims(im_frames, axis=0)
    im_frames = np.expand_dims(im_frames, axis=4)

    output = model.predict(im_frames)
    loss = mean_squared_loss(im_frames, output)
    print("Mean Squared Loss:", loss)

    if frame is None:
        print("Frame is None")

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

    if 0.00041 < loss < 0.00060:  
        print('Abnormal Event Detected')

        cv.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), 2)
        
        text = "Abnormal Event"
        (text_width, text_height), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
        

        cv.rectangle(image, (50, 50 - text_height), (50 + text_width, 50), (255, 255, 255), -1)
        
        cv.putText(image, text, (45, 46), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    resized_frame = cv.resize(image, None, fx=1.5, fy=1.5, interpolation=cv.INTER_LINEAR)
    cv.imshow("DeepEYE Anomaly Surveillance", resized_frame)

cap.release()
cv.destroyAllWindows()