# Autonomous Car Camera Object Detection

## 📌 Description
This team project with Joseph Mouscadet, Simon Drouet and Maxime Kunsch implements **camera-based object detection** for optimizing vehicle navigation in autonomous driving applications. The implementation utilizes **YOLOv5 models** for real-time object detection through a vehicle-mounted camera system.

The project focuses on efficient detection algorithms that can run on resource-constrained environments, with multiple optimization strategies including NCNN model conversion.

---

## Features
 **Real-time Object Detection** - Using YOLOv5 variants optimized for speed and accuracy  
 **Model Optimization** - NCNN conversion for embedded deployment  
 **Multiple Detection Modes** - Standard, real-time, and centered detection options  
 **Extensive Testing Framework** - Comprehensive testing for various scenarios  
 **Cross-platform Compatibility** - Works across different operating systems and hardware  

---

## Project Structure

```
/rc_camera_project
│── camera.py                      # Base camera module
│── camerayolo.py                  # YOLOv5 integration with camera
│── camerayolo-realtime.py         # Real-time detection implementation
│── camera-test-realtime.py        # Testing script for real-time performance
│── camera-test-centrage.py        # Testing for centered object detection
│── camerancnn-realtime.py         # NCNN-based real-time detection
│── yolo_ncnn.py                   # NCNN model wrapper for YOLO
│── ncnn-convert.py                # Conversion utility for NCNN format
│── test.py                        # Basic testing script
│── testall.py                     # Comprehensive testing
│
├── Models
│   │── yolov5n.pt                 # YOLOv5n model (smallest variant)
│   │── yolov5s.pt                 # YOLOv5s model (small variant)
│   │── yolov5nu.pt                # YOLOv5nu model (nano ultralytics)
│   │── yolov5su.pt                # YOLOv5su model (small ultralytics)
│   │── yolov5nu.torchscript       # TorchScript version of yolov5nu
│   │── yolov5su.torchscript       # TorchScript version of yolov5su
│
├── yolov5nu_ncnn_model            # NCNN converted model folder for yolov5nu
│   │── model_ncnn.py              # Python wrapper for NCNN model
│   │── model.ncnn.param           # NCNN model parameters
│   │── model.ncnn.bin             # NCNN model binary
│   │── metadata.yaml              # Model metadata
│
└── yolov5su_ncnn_model            # NCNN converted model folder for yolov5su
│── model_ncnn.py              # Python wrapper for NCNN model
│── model.ncnn.param           # NCNN model parameters
│── model.ncnn.bin             # NCNN model binary
│── metadata.yaml              # Model metadata
```
