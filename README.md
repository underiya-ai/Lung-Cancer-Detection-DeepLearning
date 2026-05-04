# 🧠 Lung Cancer Detection using DeepLearning 

## 📌 Overview
This project focuses on building a **Deep Learning model using Convolutional Neural Networks (CNNs)** to detect and classify lung cancer from medical images.

The model performs **multi-class classification** to identify different types of lung cancer and distinguish them from normal lung tissue.

---

## 🎯 Objectives
- Classify lung images into different cancer categories  
- Apply CNN for medical image classification  
- Improve understanding of deep learning in healthcare  

---

## 📂 Dataset
A custom dataset (~2400 images) containing three categories:

- 🟢 **Normal (lung_n)**  
- 🔴 **Adenocarcinoma (lung_aca)**  
- 🔵 **Squamous Cell Carcinoma (lung_scc)**  

---

## 🛠️ Tech Stack & Libraries

- **Programming Language:** Python  
- **Data Handling:** NumPy, Pandas  
- **Visualization:** Matplotlib  
- **Machine Learning:** Scikit-learn  
- **Image Processing:** OpenCV, PIL  
- **Deep Learning:** TensorFlow, Keras  

---

## ⚙️ Data Preprocessing

- Used `ImageDataGenerator` for efficient data loading  
- Rescaled pixel values (normalization)  
- Split dataset:
  - **80% Training**
  - **20% Validation**
- Loaded data in batches to prevent memory overflow  

---

## 🧩 Model Architecture

The CNN model is built using **Keras Sequential API**:

- `Conv2D` – Feature extraction  
- `MaxPooling2D` – Downsampling  
- `BatchNormalization` – Stabilizes training  
- `Dropout` – Prevents overfitting  
- `Flatten` – Converts feature maps to vector  
- `Dense` – Fully connected layers  
- `Softmax` – Multi-class classification  

---

## 🚀 Model Training

- **Optimizer:** Adam  
- **Loss Function:** Categorical Cross-Entropy  
- **Metric:** Accuracy  

### 🔁 Training Optimization Techniques

- **EarlyStopping:** Stops training when validation accuracy stops improving  
- **ReduceLROnPlateau:** Reduces learning rate when validation loss plateaus  
- **Custom Callback:** Stops training when validation accuracy exceeds 90%  

---

## 📊 Results
*(Update this section with your results)*

- Validation Accuracy: **92%**  
- Training vs Validation Accuracy Graph  

---

## 📈 Visualization

The model performance is visualized using accuracy plots:

```python
history_df[['accuracy', 'val_accuracy']].plot()
