import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pygame import mixer
import os

# Initialize sound alert
mixer.init()
sound = mixer.Sound('alert.wav')  # .wav beep sound file

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load trained model
model = load_model("mask_detector.model.h5")

# Load video file
cap = cv2.VideoCapture("input_video.mp4")  # तुझं video इथे वापर

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        (mask, no_mask) = model.predict(face)[0]

        label = "Mask" if mask > no_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        if label == "No Mask":
            sound.play()

        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Show video frame
    cv2.imshow("Video - Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

