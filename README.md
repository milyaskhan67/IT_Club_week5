# IT_Club_week5
# MNIST Handwritten Digit Classification 🖊️🔢  

This repository contains a **Python script** for training a deep learning model on the **MNIST dataset** using **TensorFlow** and **Keras**. The script implements a **fully connected neural network (MLP)** for handwritten digit recognition and explores different **hyperparameter settings** to optimize performance.  

---

## 📌 Key Features  

### ✅ **Data Preprocessing**  
- Loads the **MNIST dataset** (60,000 training and 10,000 test images).  
- Normalizes pixel values to **scale between 0 and 1** for better training performance.  
- Splits data into **training and testing sets** for evaluation.  

### 🏗 **Model Architecture**  
- A simple **feedforward neural network (MLP)** with:  
  - **Flatten layer** to convert 28×28 images into 1D vectors.  
  - **Two dense hidden layers** with 128 and 64 neurons, using **ReLU activation**.  
  - **Output layer** with **softmax activation** to classify digits (0-9).  

### 🎯 **Training & Evaluation**  
- Compiles the model using **Adam optimizer** and **sparse categorical cross-entropy loss**.  
- Trains for **10 epochs** with **batch size = 32** (default configuration).  
- Evaluates model performance on the **test set** using **accuracy** and **loss metrics**.  

### 📊 **Visualization**  
- **Accuracy & loss curves** plotted using **Matplotlib** to track performance trends.  
- Helps identify **overfitting** or **underfitting** during training.  

### 🔍 **Hyperparameter Tuning**  
- Runs **multiple experiments** to test different:  
  - **Learning rates**: 0.0001, 0.001, 0.01  
  - **Batch sizes**: 32, 64, 128  
- Stores **validation accuracy** results for comparison.  
- Identifies the best **learning rate & batch size combination** for optimal performance.  

### 🤖 **Automated Hyperparameter Testing**  
- Iterates through different configurations and **automatically trains models** with varying hyperparameters.  
- Prints a **summary report** of results with the best model settings.  

---

## 🚀 Why Use This Script?  
✔ **Beginner-friendly** deep learning implementation with TensorFlow.  
✔ **Hands-on** experience with **model training, evaluation, and tuning**.  
✔ **Practical** use case for image classification tasks.  
✔ **Visualization tools** for better model understanding.  
✔ **Automated hyperparameter tuning** for performance improvement.  

This script is a great starting point for **machine learning enthusiasts, students, and researchers** looking to understand the fundamentals of deep learning and optimize models efficiently! 🎯🔥
