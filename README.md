# Vehicle Detection and Counting using YOLO11

**YOLOv11 (You Only Look Once)** is a state-of-the-art object detection model known for its speed and accuracy. It uses deep learning techniques to efficiently detect and track objects in images and videos, making it ideal for real-time applications like vehicle counting and traffic monitoring.

This project implements **vehicle detection and counting** using **YOLOv11** and OpenCV. It processes a video file to track and count vehicles that cross a predefined red line, providing real-time visualizations of the detections and counts.

## 🚀 Features

- **Real-time vehicle detection and tracking** using YOLOv11.
- **Counts vehicles** that cross a red line in the video.
- **Bounding boxes and track IDs** displayed for each detected vehicle.
- **Video output with overlayed tracking results** is saved.

## 🛠️ Tech Stack

- **Python**
- **YOLOv11** (Ultralytics)
- **OpenCV** (Computer Vision Library)
- **PyTorch** (for YOLO model)
- **Numpy** (Array manipulations)

## 🎥 Input and Output

- **Input:** Video file (`./test videos/test_1.mp4`)
- **Output:** Processed video saved as `output_video.mp4`
- **Visualization:** Displays the tracking results with bounding boxes and counts

## 📜 Code Explanation

1. **Loads YOLO model** using Ultralytics.
2. **Reads input video** and extracts properties like width, height, and FPS.
3. **Processes each frame** to detect and track vehicles (cars, bikes, etc.).
4. **Draws a red line** and counts vehicles crossing it.
5. **Saves processed video** with detected objects and counts.
6. **Displays real-time output** while processing.

## 🎯 Customization

- Change the input video path in `cap = cv2.VideoCapture('./test videos/test video_1.mp4')`.
- Modify `line_y_red = 430` to change the red line position.
- Adjust `classes=[1,2,3,5,6,7]` to track specific object categories:
  - `1` - Bicycle 🚲
  - `2` - Car 🚗
  - `3` - Motorcycle 🏍️
  - `5` - Bus 🚌
  - `6` - Train 🚆
  - `7` - Truck 🚛

## 📝 Future Improvements

- Add support for real-time webcam input.
- Implement speed estimation of detected vehicles.
- Export vehicle count data to a CSV file.

## 📷 Preview

<img width="886" height="499" alt="image" src="https://github.com/user-attachments/assets/caa7242f-3917-4c36-b8a9-f408098c8c7a" />

<img width="886" height="497" alt="image" src="https://github.com/user-attachments/assets/3e36ae6f-af8a-4576-900d-78f6a34ecdcc" />

<img width="886" height="496" alt="image" src="https://github.com/user-attachments/assets/1c82c015-4a86-429f-9276-1dffe4f73626" />




