# cat-dog-detection

## Cat-Dog Detection Model Using YOLOv5

A brief description of what this project does and who it's for

## Aim and Objectives
The aim of this project is to develop a robust and accurate model for detecting cats and dogs in images using the YOLOv5 architecture. The objectives include:
- Collecting and preparing a comprehensive dataset of cat and dog images.
- Training a YOLOv5 model to accurately detect and classify cats and dogs.
- Evaluating the performance of the model.
- Deploying the model for real-time detection.

## Abstract
This project focuses on creating a machine learning model to detect cats and dogs in images using the YOLOv5 framework. By leveraging a well-curated dataset and the advanced YOLOv5 architecture, the model aims to achieve high accuracy and efficiency in detection tasks. The project includes data collection, model training, evaluation, and deployment.

## Introduction
Object detection is a critical task in computer vision, with applications ranging from security surveillance to autonomous driving. This project aims to develop an object detection model specifically for identifying cats and dogs. Using YOLOv5, a state-of-the-art object detection framework, the model will be trained and tested on a robust dataset to ensure high performance.

## Literature Review
Several object detection frameworks have been developed over the years, including R-CNN, Fast R-CNN, and YOLO. YOLO (You Only Look Once) is known for its speed and accuracy, making it suitable for real-time applications. YOLOv5, the latest version, improves on its predecessors by offering better performance and easier implementation. Previous studies have demonstrated the effectiveness of YOLOv5 in various detection tasks, which motivates its use in this project.

## Methodology
1. **Dataset Preparation**:
   - Use the Roboflow platform to collect and annotate images of cats and dogs.
   - Split the dataset into training, validation, and test sets.

2. **Model Training**:
   - Clone the YOLOv5 repository.
   - Install necessary dependencies.
   - Train the model using the prepared dataset.

3. **Model Evaluation**:
   - Evaluate the model on the test set to measure its performance.
   - Fine-tune the model based on evaluation results.

4. **Model Deployment**:
   - Use the trained model for real-time detection of cats and dogs in images and videos.

## Installation
```bash
# Clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5
%cd yolov5

# Install dependencies
%pip install -qr requirements.txt
%pip install -q roboflow

import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
os.environ["DATASET_DIRECTORY"] = "/content/datasets"

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("your-workspace").project("cat_dog_detection")
version = project.version(1)
dataset = version.download("yolov5")




## Train the model
!python train.py --img 416 --batch 4 --epochs 100 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache

##  Test the model
!python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.1 --source {dataset.location}/test/images
!python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.1 --source /content/cat1.jpg

##  Applications
Pet monitoring systems.
Animal behavior studies.
Security systems to detect pets in restricted areas.
## Future Scope
Expanding the model to detect other animals.
Improving the model's accuracy with more diverse datasets.
Integrating the model into mobile and IoT devices for real-time applications.
## Conclusion
This project demonstrates the effectiveness of YOLOv5 in detecting cats and dogs in images. The trained model achieves high accuracy and can be used in various applications, from pet monitoring to security systems. Future work will focus on expanding the model's capabilities and improving its performance.
## References
YOLOv5 GitHub Repository
Relevant research papers and articles on object detection and YOLOv5.




#### Demo 



https://github.com/Priyanka-Dongre1992/cat-dog-detection/assets/174680239/8a3ac5aa-3602-40c9-a4aa-557c5266c6f6

Link :- https://youtu.be/5ODFO9cIIio
