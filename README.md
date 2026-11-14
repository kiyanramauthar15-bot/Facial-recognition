# Real-Time Facial Detection

A lightweight, real-time webcam face detection application built with Python and OpenCV.

## Overview
This project captures live video from a computer's default webcam and detects human faces in real time using OpenCV's Haar Cascade classifier. Detected faces are highlighted with a bounding box directly on the live video feed.

## Project Files
- **`face_detect.py`**: Main Python script that initializes the webcam, processes frames in grayscale, runs face detection, and displays the tracking window.
- **`haarcascade_frontalface_default.xml`**: Pre-trained Haar Cascade XML model used for frontal face detection (automatically downloaded if missing).
- **`run_face.bat`**: Windows batch script for launching the application directly with a double-click.

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)

## Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/kiyanramauthar15-bot/Facial-recognition.git](https://github.com/kiyanramauthar15-bot/Facial-recognition.git)
   cd Facial-recognition
