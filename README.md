#  Hybrid Mamba–Transformer Architecture for Breast Cancer Classification

### Authors
**Niloofar Dehghani**, **Omid Nejatimanzari**, **Mohammad Naserisafavi**, **Seyed MohammadHosein Enjavimadar**, **Sajad Ahmadian Fini**  
*Applied Machine Learning Course Project – Concordia University (2025)*

---

## 📘 Overview
This project explores a **hybrid deep learning architecture** that combines the strengths of **Mamba state-space models** and **Vision Transformers (ViT)** for accurate and efficient **breast cancer image classification**.  
The hybrid model leverages the **long-range dependency modeling** capability of Mamba and the **global attention mechanism** of Transformers to improve diagnostic accuracy while maintaining computational efficiency.

---

## 🎯 Objectives
- Develop a **robust hybrid model** that integrates Mamba and Transformer layers.  
- Evaluate the architecture on **breast cancer imaging datasets** (e.g., BreakHis, IDC, or custom histopathology datasets).  
- Compare performance against traditional baselines such as **CNNs, ResNet, ConvNeXt, ViT, and pure Mamba models**.  
- Analyze **accuracy, F1-score, precision, recall**, and **model complexity**.

---

## 🧩 Model Architecture
The proposed **Hybrid Mamba–Transformer** combines:
- **Mamba Blocks** for efficient sequence modeling and implicit recurrence.
- **Transformer Blocks** for global context learning.
- A **fusion head** that aggregates both representations before final classification.

The architecture aims to balance **local feature extraction** and **global representation learning**—making it highly suitable for medical image analysis tasks.

---

## 🧠 Dataset:
Dataset name: Breast Ultrasound Images Dataset (Al-Dhabyani et al., 2020)

Link to dataset: https://www.kaggle.com/datasets/anasmohammedtahir/breast-ultrasound-images-dataset
- Collected in 2018 from 600 female patients aged between 25 and 75 years  
- Contains 780 ultrasound images with an average resolution of 500×500 pixels in PNG format  
- Images are labeled into three diagnostic classes: Normal, Benign, and Malignant

| **Case**      | **Number of Images** |
|----------------|----------------------|
| Benign         | 487                  |
| Malignant      | 210                  |
| Normal         | 133                  |
| **Total**      | **780**              |

## 🧮 Exploratory Data Analysis (EDA)

A detailed EDA was conducted to understand the structure, balance, and quality of the BUSI dataset before model training. The analysis included:

Class distribution: Verified the number of images in each category (benign, malignant, normal) using Seaborn bar plots to identify class imbalance.

Sample visualization: Displayed representative ultrasound images from each class to confirm labeling accuracy and image quality.

Image resolution consistency: Checked width and height distributions (in pixels) to ensure uniform image size.

Brightness analysis: Calculated mean pixel intensity for all samples to evaluate exposure and illumination uniformity across classes.

Data integrity: Detected and excluded corrupted or non-image files.

All EDA steps were implemented in EDA.ipynb using Python libraries (Pillow, OpenCV, Matplotlib, Seaborn, NumPy, Pandas).
Results confirmed that the dataset is grayscale, properly exposed, and ready for standardized preprocessing and augmentation.

  
## 📂 Repository Structure
