# Face Mask Detection with Live Alert System (Video Based)

This project detects whether people in a video are wearing face masks or not, and triggers a live alert (beep sound) when someone is not wearing a mask.

---

## 🎯 Objective

To build a system that analyzes pre-recorded video footage and detects if people are wearing face masks in real-time using a deep learning model, and gives a sound alert if a person without a mask is detected.

---

## 🧠 Technologies Used

- **Python 3**
- **OpenCV** – For face detection & video processing
- **Keras + TensorFlow** – For deep learning mask detection model
- **Haarcascade** – For face detection (no external models required)
- **NumPy** – Array handling
- **Pygame** – For playing alert sounds (`alert.wav`)

## 📁 Project Structure

FaceMaskDetection_with_alert_system/
│
├── detect_mask_video_file.py # Main file for video-based detection
├── train_mask_detector.py # Script to train the mask detection model
├── mask_detector.model.h5 # Trained model (saved in H5 format)
├── alert.wav # Beep sound for live alert
├── input_video.mp4 # Sample or test video file
└── dataset/
├── with_mask/ # Images of people wearing masks
└── without_mask/ # Images of people without masks

Dataset download link:
https://www.kaggle.com/datasets/shiekhburhan/face-mask-dataset

# ✅ How to Use

### 1. Install required libraries

Open terminal and run:

```bash
pip install opencv-python tensorflow keras pygame numpy
and also imstall if needed for matplotlib.

2. Prepare the dataset
Organize your dataset like:
dataset/
├── with_mask/
└── without_mask/

3. Train the model
```bash
python train_mask_detector.py

note: This will create a file mask_detector.model.h5

4. Place your video file
Put your video file in the project folder and name it input_video.mp4, or change the filename in the code.

5. Run the detection
```bash
python detect_mask_video_file.py

The script will:

Open the video
Detect faces frame by frame
Show label: "Mask" or "No Mask"
Play beep sound if "No Mask"

📌 Notes
-- You do not need a webcam.

-- This project works with pre-recorded .mp4 or .avi files.

-- Haar cascade face detection is lightweight and does not require additional downloads.

🙌 Developed By
Chaitany Chindarkar
Python Project: Face Mask Detection with Live Alert System
