---
title: SignBERT - 50 Classes ISL Recognition
emoji: 🤟
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
hardware: t4-small
---

# 🤟 SignBERT - Indian Sign Language Recognition (50 Classes)

An advanced deep learning model for recognizing 50 Indian Sign Language (ISL) gestures in real-time.

## 🎯 Features

- **50 ISL Signs Recognition**
- **Real-time Video Processing**
- **Top-3 Predictions with Confidence Scores**
- **MobileNetV2 + Transformer + BiLSTM Architecture**
- **96x96 Image Size, 12 Frames per Video**

## 📊 Model Architecture

- **Backbone:** MobileNetV2 (ImageNet pretrained)
- **Temporal Processing:** Transformer with Multi-Head Attention
- **Sequence Modeling:** Bidirectional LSTM
- **Classification:** Dense layers with Dropout

## 🚀 How to Use

1. Upload a video of an ISL sign gesture (MP4, AVI, MOV)
2. Click "Predict Sign"
3. View the predicted sign with confidence score
4. Check top-3 predictions for alternatives

## 📋 Supported Signs (50 Classes)

The model recognizes 50 different ISL signs including common phrases, emotions, and daily communication needs.

## 🔧 Technical Details

- **Framework:** TensorFlow/Keras
- **Input:** 12 frames @ 96x96 pixels
- **Confidence Threshold:** 60%
- **Model Size:** Optimized for deployment

## 📝 License

MIT License
