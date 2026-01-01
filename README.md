# 🎥 Video Anomaly Detection using Autoencoders

This repository implements video anomaly detection using autoencoder-based deep learning models. The goal is to automatically identify abnormal events in surveillance videos by learning normal patterns and detecting deviations based on reconstruction error.

The project focuses on unsupervised learning, making it suitable for real-world surveillance scenarios where labeled anomaly data is scarce.

# 🧠 Project Overview

Autoencoders are trained on videos containing only normal behavior. During inference, frames or sequences that cannot be well reconstructed by the autoencoder are flagged as anomalies.

### This repository contains:

- Video preprocessing utilities

- Autoencoder training pipeline

- Testing pipeline for anomaly detection

A sample MP4 demo showing anomaly detection results.

# ✨ Key Features

- Unsupervised video anomaly detection

- Autoencoder-based reconstruction approach

- Frame-level anomaly scoring

- Video-to-array preprocessing

- Simple and modular project structure

- Docker support for reproducibility

# 📁 Repository Structure
```
.
├── model/                                       # Saved trained models
├── Video Anomaly Detection - Autoencoders.mp4   # Demo / testing video
├── train.py                                     # Train autoencoder model
├── test.py                                      # Test and detect anomalies
├── video2array.py                               # Convert video to frame arrays
├── Dockerfile                                   # Docker configuration
├── README.md                                    # Project documentation
└── .gitignore
```

# 🏋️ Training the Model

Convert normal surveillance videos into frame arrays using:

```bash
python video2array.py
```


Train the autoencoder on normal video data:

```bash
python train.py
```


The model learns to reconstruct normal motion and appearance patterns from the video data.

# 🔍 Testing & Anomaly Detection

To detect anomalies in a new video:

```bash
python test.py
```

### During testing:

- The trained autoencoder attempts to reconstruct video frames

- Reconstruction error is calculated

- Frames with high reconstruction error are classified as anomalous

# 📹 Demo Output

A sample testing video demonstrating anomaly detection results is included in this repository:

📁 ./Results/Video Anomaly Detection - Autoencoders.mp4

This video shows how abnormal events stand out when reconstruction error increases.

# 🧩 Why Autoencoders for Anomaly Detection?

- No labeled anomaly data required

- Learns compact representations of normal behavior

- Anomalies naturally produce higher reconstruction errors

- Widely used in video surveillance and industrial monitoring

# 🙏 Acknowledgements & Inspiration

This project was heavily inspired by and developed through learning from the following repository:

🔗 DeepEYE – Video Surveillance with Anomaly Detection
https://github.com/jnagidi/DeepEYE-Video-Surveillance-with-Anomaly-Detection

#### The DeepEYE project helped in understanding:

- Spatio-temporal autoencoder concepts

- Video preprocessing strategies

- Reconstruction-based anomaly detection logic

- Special thanks to the author for making the work open source.

# 🚀 Future Improvements

- Use 3D Convolutional Autoencoders

- Add temporal sequence modeling (ConvLSTM)

- Threshold optimization using ROC curves

- Real-time anomaly detection

- Visualization of anomaly heatmaps