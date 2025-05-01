# 🧠 Object Detection Using CNN and RCNN

## 📌 Introduction

Object detection is a crucial task in computer vision, enabling machines to identify and localize objects within an image. This project compares two deep learning approaches for object detection:

- **CNN (Convolutional Neural Network)**: Fast and efficient for real-time applications.
- **RCNN (Region-based CNN)**: More accurate due to region proposals but slower in execution.

The goal is to understand the strengths and weaknesses of both architectures in bounding box prediction tasks.

---

## 🏗️ Model Architectures

### ✅ CNN Architecture
- **Convolutional Layers**:
  - Conv1: 32 filters, kernel size 5  
  - Conv2: 64 filters, kernel size 5  
  - Conv3: 128 filters, kernel size 5  
- **Fully Connected Layers**:
  - Dense layer with 2046 neurons  
  - Output layer with 4 neurons (bounding box coordinates)  
- **Dropout**: 50% after the first fully connected layer

### ✅ RCNN Architecture
- **Region Proposal Network (RPN)**: Generates object region proposals  
- **Feature Extraction**: Convolutional layers + adaptive average pooling  
- **Fully Connected Layers**: Followed by dropout and ReLU activations for bounding box regression  

---

## ⚙️ Methodology

### 🏋️‍♂️ Training
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum = 0.9
- **Learning Rate**: 1e-6 with decay every 30 epochs
- **Epochs**: 50

### 🧪 Testing
- Models evaluated using Intersection-over-Union (IoU) for predicted vs ground-truth bounding boxes

### 📁 Dataset
- **Training Set**: 1,000 labeled images  
- **Test Set**: 200 labeled images  

---

## 📊 Results

| Metric                  | CNN Model     | RCNN Model    |
|------------------------|---------------|---------------|
| **IoU (Average Overlap)** | 0.76          | 0.84          |
| **Training Time**       | Fast          | Slow          |
| **Inference Speed**     | Fast          | Slow          |
| **Model Complexity**    | Low           | High          |

> RCNN provides better accuracy, while CNN offers faster performance suitable for real-time applications.

---

## 🔍 Comparative Analysis

| Metric              | CNN        | RCNN       |
|---------------------|------------|------------|
| Overlap Score       | 0.3730     | 0.4372     |
| Training Time       | ✅ Faster  | ❌ Slower  |
| Inference Speed     | ✅ Faster  | ❌ Slower  |
| Model Complexity    | Low        | High       |

---

## ✅ Conclusion

- **RCNN** is suitable when accuracy is a top priority (e.g., offline processing).  
- **CNN** is ideal for scenarios requiring speed (e.g., real-time detection).  
- Both models have proven effective, and the choice depends on application constraints.

---

## 🚀 Future Work

- ⚡ **Optimize RCNN**: Implement Fast R-CNN or Faster R-CNN  
- 🧠 **Data Augmentation**: Improve model generalization with rotation, cropping, flipping, etc.  
- 🔀 **Hybrid Models**: Combine region proposals from RCNN with CNN refinement for performance gains  



