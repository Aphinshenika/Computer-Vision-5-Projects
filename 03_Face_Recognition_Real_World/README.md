# Face Recognition Under Real-World Conditions

## Project Overview
A computer vision project using Python and OpenCV to detect faces from a live webcam feed.

## Features
- Real-time webcam capture
- Face detection using Haar Cascade
- Green bounding boxes around detected faces
- Live video display
- Press Q to exit

## Technologies
- Python
- OpenCV
- NumPy
- Computer Vision

## Project Structure
```text
src/main.py
dataset/
results/
README.md
requirements.txt
```

## Installation

Create and activate a virtual environment on Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the project:

```powershell
python src/main.py
```

## How It Works
1. Capture frames from the webcam.
2. Convert each frame to grayscale.
3. Detect faces using a Haar Cascade classifier.
4. Draw a green rectangle around each detected face.
5. Display the live output.

## Current Status
Real-time face detection is implemented and tested. Identity recognition of known people and systematic testing under different real-world conditions are future improvements.

## Author
ASPHIN SHENIKA
