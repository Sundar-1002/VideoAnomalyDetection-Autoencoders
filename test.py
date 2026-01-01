 #Importing Libraries
import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import imutils

# Function to calculate mean squared loss. We are doing this by ourselves to have more control over the calculation. This function computes the mean squared loss between the input frames and the reconstructed output frames from the model.
def mean_squared_loss(x1, x2):
    diff = x1 - x2 # Calculate the difference between input and output
    a,b,c,d,e = diff.shape # Get the shape of the difference tensor
    n_samples = a*b*c*d*e # Total number of elements
    sq_diff = diff ** 2 # Square the differences
    total = sq_diff.sum() # Sum all squared differences
    distance = np.sqrt(total) # Compute the Euclidean distance
    mean_distance = distance / n_samples # Normalize by number of elements
    return mean_distance

# Load the trained model
model = load_model('./model/saved_model.keras', compile=False)

#Loading Abnormal Video
cap = cv.VideoCapture('./Avenue Dataset/testing_videos/04.avi')

print(cap.isOpened())

# Process video frames
while cap.isOpened():
    im_frames = [] # List to hold frames
    isTrue, frame = cap.read() # Read a frame from the video
    if not isTrue:
        break

    # Preprocess 10 frames
    for i in range(10):
        isTrue, frame = cap.read()
        if not isTrue:
            break
        image = imutils.resize(frame, width = 700, height = 600) # Resize frame for consistency
        frame = cv.resize(image, (227, 227), interpolation=cv.INTER_CUBIC) # Resize to model input size
        gray = 0.2989 * frame[:,:,0] + 0.5870 * frame[:,:,1] + 0.1140 * frame[:,:,2] # Convert to grayscale
        gray = (gray - gray.mean()) / gray.std()    # Normalize the frame
        gray = np.clip(gray, 0, 1) # Clip values to [0, 1]
        im_frames.append(gray)

    im_frames = np.array(im_frames) # Convert list to numpy array
    im_frames = im_frames.reshape(227, 227, 10) # Reshape to (1, 227, 227, 10)
    im_frames = np.expand_dims(im_frames, axis=0) # Add batch dimension
    im_frames = np.expand_dims(im_frames, axis=4) # Add channel dimension

    output = model.predict(im_frames) # Get model prediction
    loss = mean_squared_loss(im_frames, output) # Calculate mean squared loss
    print("Mean Squared Loss:", loss)

    # Check for None values
    if frame is None:
        print("Frame is None")

    # wait for 10 ms before moving on to the next frame and wait for 'q' key to stop
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

    # If loss is within the abnormal range, draw rectangle and text
    if 0.00040 < loss < 0.00060:  
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