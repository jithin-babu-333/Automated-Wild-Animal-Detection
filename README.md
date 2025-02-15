# Automated-Wild-Animal-Detection
# Wildlife Image Classification using ResNet

## 📌 Overview
This project implements a **ResNet-based deep learning model** to classify images of wildlife species. The dataset includes images of animals such as buffalo, elephants, monkeys, tigers, and wild boars. The model utilizes **data augmentation, early stopping, and transfer learning with ResNet** to achieve high classification accuracy.

## 🚀 Features
- **Image Classification** using **ResNet-50**
- **Data Augmentation** for class balance
- **Transfer Learning** with pre-trained ResNet-50
- **Early Stopping** to prevent overfitting
- **Performance Evaluation** with Confusion Matrix & ROC Curve
- **Interactive Model Testing** on sample images

## 📂 Dataset
The dataset consists of images of five animal species:
- 🦬 Buffalo
- 🐘 Elephant
- 🐵 Monkey
- 🐅 Tiger
- 🐗 Wild Boar

### 📊 Data Preprocessing
- **Image Augmentation**: Augment images to ensure each class has 300 samples.
- **Normalization**: Scale pixel values between 0 and 1.
- **Splitting**: Training (70%), Testing (20%), Validation (10%).

## 🔧 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/jithin-babu-gif/Wildlife-ResNet-Classification.git
cd Wildlife-ResNet-Classification
```
### 2️⃣ Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Running the Model
```bash
python miniproject_with_resnet_model.py
```

## 🎯 Model Architecture
The model is built using **ResNet-50**, with additional CNN layers:
- **Convolutional Layers** for feature extraction
- **Batch Normalization** for training stability
- **Dropout Layers** to prevent overfitting
- **Fully Connected Layers** for classification

## 📊 Model Performance
| Metric       | Score |
|-------------|-------|
| Accuracy    | 82.66% |
| Precision   | 81%   |
| Recall      | 79%   |
| F1-Score    | 80%   |

## 📉 Evaluation Metrics
- **Confusion Matrix** to analyze misclassifications
- **ROC Curves** for multi-class classification
- **Correct & Incorrect Predictions Visualization**

## 🔮 Future Enhancements
- Train on a larger, more diverse dataset
- Experiment with other architectures (EfficientNet, Vision Transformers)
- Deploy as a web-based classification tool

## 👨‍💻 Author
[Jithin Babu](https://github.com/jithin-babu-gif)  
[LinkedIn Profile](https://www.linkedin.com/in/jithin-babu-a34287246)

## 📜 License
This project is licensed under the **MIT License**.

