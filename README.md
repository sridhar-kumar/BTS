---
title: Brain Tumor Segmentation
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.52.2
app_file: app.py 
pinned: false
license: apache-2.0
---


# 🧠 Brain Tumor Detection & Segmentation

This application uses a deep learning **U-Net (ResNet34)** model to detect and segment brain tumors from MRI images.

## 🚀 Features
- Upload grayscale brain MRI images
- Automatic tumor region segmentation
- Visual mask and overlay output
- CPU-friendly deployment

## 🧪 Model Details
- Architecture: U-Net  
- Encoder: ResNet34  
- Input size: 256 × 256 (grayscale)  
- Framework: PyTorch + segmentation-models-pytorch  

## 📌 Usage
1. Upload an MRI image  
2. View predicted tumor mask  
3. Analyze overlay visualization  

⚠️ **For educational and research purposes only**