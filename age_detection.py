import cv2
import numpy as np

# Load the age detection model
age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')

# Load the age labels
age_labels = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    _, frame = cap.read()

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Feed the blob to the age detection model
    age_net.setInput(blob)
    predictions = age_net.forward()

    # Get the age with the highest probability
    i = np.argmax(predictions[0])
    age = age_labels[i]
    age_confidence = predictions[0][i]

    # Draw the age label on the frame
    cv2.putText(frame, f'Age: {age} ({age_confidence * 100:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam
cap.release()
