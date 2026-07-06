# 🧪 CNN-Based Soft Infectious Biomedical Waste Classification System

A deep learning-based web application that classifies biomedical waste into **General Waste** and **Infectious Waste** using the **MobileNetV2** Convolutional Neural Network (CNN). The application is built with **TensorFlow** and **Streamlit**, providing real-time predictions through image upload or webcam capture.

---

## 📌 Project Overview

Biomedical waste segregation is essential for maintaining hygiene and preventing the spread of infections in healthcare facilities. This project automates the classification of biomedical waste using a trained MobileNetV2 model, enabling fast and accurate waste segregation.

---

## ✨ Features

- 🧠 MobileNetV2 Deep Learning Model
- 📁 Image Upload
- 📷 Webcam Image Capture
- 🔍 Real-Time Waste Classification
- 📊 Prediction Confidence Score
- 📜 Prediction History
- 📄 Download Prediction Report (PDF)
- 📥 Download Prediction History (CSV)
- 🌐 English & Tamil Language Support
- 🌙 Professional Dark Theme
- 📱 Mobile-Friendly Responsive Design

---

## 🗂️ Dataset Structure

```
dataset/
│
├── General/
│   ├── food
│   ├── glass
│   ├── metal
│   ├── paper
│   └── plastic
│
└── Infectious/
    ├── bandages
    ├── cotton
    ├── gauze
    ├── gloves
    └── mask
```

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- ReportLab
- QRCode

---

## 📁 Project Structure

```
Biomedical_Waste_App/
│
├── app.py
├── predict.py
├── requirements.txt
├── class_names.txt
│
├── Model/
│   └── mobilenetv2_waste_model.h5
│
├── images/
│   └── logo.png
│
├── history/
│   └── prediction_history.csv
│
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https:https://github.com/visalatchit23bm-tech/CNN-Based-Soft-infectious-waste-classification
```

Move to the project folder:

```bash
cd CNN Based soft infectious waste classification
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🎯 Model

- Model: **MobileNetV2**
- Framework: TensorFlow / Keras
- Input Size: **224 × 224**
- Output Classes:
  - General Waste
  - Infectious Waste

---

## 📸 Application Workflow

1. Upload an image or capture one using the webcam.
2. The image is preprocessed and resized.
3. MobileNetV2 predicts the waste category.
4. The application displays:
   - Predicted Class
   - Confidence Score
   - Waste Description
   - Disposal Instructions
5. Prediction history is saved automatically.

---

## 📊 Future Enhancements

- Multi-class biomedical waste classification
- IoT-enabled smart waste bin integration
- Cloud database for prediction storage
- Barcode/QR-based waste tracking
- Android application deployment

---

## 👩‍💻 Developer

**Visalatchi T**

Biomedical Engineering Student

---

## 📄 License

This project is developed for **academic and educational purposes** as a Final Year Biomedical Engineering Project.
