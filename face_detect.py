import cv2
import os
import urllib.request

# Automatically download the Haar Cascade file locally if it's missing
xml_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(xml_path):
    print("Downloading face cascade model...")
    url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    urllib.request.urlretrieve(url, xml_path)

# Load the local face detector
face_cascade = cv2.CascadeClassifier(xml_path)

video_capture = cv2.VideoCapture(0)
print("Webcam started! Press 'q' on your keyboard to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Webcam Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()