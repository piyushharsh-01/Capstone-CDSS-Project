# 🫁 Pneumonia Detection CDSS (Clinical Decision Support System)

A Machine Learning-based decision support tool designed to automate the classification of pediatric Chest X-Rays (CXRs) as either **Normal** or **Pneumonia**.

## 📌 Overview
Pneumonia is a leading cause of mortality among children worldwide. Rapid and accurate diagnosis is critical but often delayed due to a shortage of radiologists in resource-constrained regions. 

This project implements a **Convolutional Neural Network (CNN)** to serve as a "second opinion" tool for medical professionals. It processes raw X-ray images and outputs a diagnostic prediction with a confidence score.

## 🚀 Key Features
* **Custom CNN Architecture:** Lightweight 5-layer Convolutional Neural Network built from scratch using TensorFlow/Keras.
* **High Sensitivity:** Optimized to detect lung opacities characteristic of pneumonia.
* **Failure Analysis Module:** Includes a dedicated script to identify, visualize, and analyze False Positives/Negatives to improve model robustness.
* **Speed:** Designed for rapid inference, making it suitable for deployment on standard hardware.

## 📂 Dataset
This project uses the **[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** dataset from Kaggle.
* **Total Images:** 5,863 JPEG X-Rays (anterior-posterior)
* **Classes:** Normal vs. Pneumonia (Viral & Bacterial)
* **Source:** Guangzhou Women and Children’s Medical Center

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Deep Learning:** TensorFlow 2.x, Keras
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

## 📊 Results
The prototype model achieved the following metrics during training:
* **Training Accuracy:** ~97% (Epoch 3)
* **Validation Accuracy:** ~75% (Fluctuating due to small validation set)
* **Inference:** Successfully predicted unseen test cases with >99% confidence.

## 💻 How to Run
### Option 1: Google Colab (Recommended)
1.  Upload the `.ipynb` file to Google Colab.
2.  Ensure you have your **Kaggle API Key** (`kaggle.json`) ready.
3.  Run the cells sequentially to download the data and train the model.

### Option 2: Local Installation
1.  Clone this repository:
    ```bash
    git clone [https://github.com/yourusername/pneumonia-cdss.git](https://github.com/yourusername/pneumonia-cdss.git)
    ```
2.  Install dependencies:
    ```bash
    pip install tensorflow numpy matplotlib pandas scikit-learn
    ```
3.  Download the dataset and extract it into a folder named `chest_xray`.
4.  Run the script.

## 🔮 Future Improvements
* **Data Augmentation:** To reduce overfitting and improve validation stability.
* **Grad-CAM Integration:** To visualize heatmaps indicating *where* the model is looking.
* **Web Deployment:** Wrap the model in a Streamlit or Flask app for real-world usage.
