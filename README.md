# Hyperspectral Anomaly Detection Pipeline (Spectral-Spatial)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-unsupervised-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, high-performance Python pipeline for **spectral-spatial anomaly detection** in hyperspectral remote sensing datasets. This framework integrates classical multivariate statistical models with deep unsupervised learning, utilizing a fast O(1) sliding-window average box filter to suppress natural boundaries while isolating compact target anomalies.

---

## 🌟 Key Features

* **Dual-Inflow Ingestion:** Native loading for IEEE Dataport MATLAB `.mat` files or fallback to a highly realistic synthetic spatial-spectral simulator.
* **PCA Noise Suppression:** Dimensionality reduction to compress spectral bands, retaining $>97\%$ variance while filtering out high-frequency sensor noise.
* **Global RX (Reed-Xiaoli):** Classical baseline benchmark modeling background as a global multivariate Gaussian.
* **Local Mean RX:** Spatial-spectral de-biased anomaly detector utilizing a fast 2D integral box filter to suppress natural borders and riverbanks.
* **PyTorch Spectral Autoencoder:** An unsupervised deep learning bottleneck model identifying anomalies via reconstruction Mean Squared Error (MSE).
* **Adaptive Thresholding Suite:** Optimized oracle threshold search maximizing the $F_1$-score alongside ROC-AUC, PR-AUC, Sensitivity, and False Alarm Rate (FAR) evaluations.
* **Interactive Dashboard:** Complete visualization suite mapping RGB composites, ground-truth overlays, continuous heatmaps, binarized predictions, and ROC/PR curves side-by-side.

---

## 🛠️ Pipeline Architecture & Mathematical Foundations

### 1. Global RX (Reed-Xiaoli)
Global RX assumes the background is a global multivariate Gaussian. The anomaly score for a pixel vector $\mathbf{x}_i$ is its **Mahalanobis Distance** to the global mean $\mathbf{\mu}$ using the inverse covariance matrix $\mathbf{\Sigma}^{-1}$:

$$\text{Score}_i = (\mathbf{x}_i - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}_i - \mathbf{\mu})$$

* **Limitation:** Tends to flag prominent natural structures (like river borders) as anomalies due to high-contrast edges.

### 2. Local Mean RX (Edge-Suppressed Spatial-Spectral RX)
To avoid false alarms at boundary transitions, Local Mean RX subtracts a local spatial average $\mathbf{\mu}_{local}$ (computed within an $11 \times 11$ sliding window) for each pixel before computing the Mahalanobis distance:

$$\text{Residual}_i = \mathbf{x}_i - \mathbf{\mu}_{local}$$
$$\text{Score}_i = \text{Residual}_i^T \mathbf{\Sigma}^{-1} \text{Residual}_i$$

* **O(1) Integral Box Filter:** The pipeline avoids expensive sliding-window loops by building a **2D Cumulative Sum (Integral Image)**. By prepending a zero boundary, any arbitrary window sum is calculated in constant $O(1)$ time with just 4 array lookups, enabling real-time local statistics computation.

### 3. PyTorch Autoencoder (Deep Learning)
An unsupervised neural network that compresses the full 100-band spectral profile down to a highly restricted 4-dimensional latent bottleneck, then attempts to reconstruct it. Because background pixels make up $>99.9\%$ of the scene, the network optimizes to reconstruct background spectra. It fails to reconstruct rare anomalies, yielding a high **Reconstruction MSE** which indicates anomaly likelihood.

---

## 📊 Dashboard Visualizer
The pipeline generates an automated analytics dashboard saved as `hyperspectral_anomaly_dashboard.png`. 

* **Row 1:** Shows a Pseudo-RGB composite of the scene, the expert-curated ground truth binary anomaly map, and the spatial background average intensity highlighting natural features.
* **Subsequent Rows:** Map each model's **Continuous Anomaly Scores**, its **Binarized Detections** at the optimal $F_1$ threshold, and its relative **ROC and Precision-Recall Curves**.

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Vivek12042/hyperspectralcode.git
cd hyperspectralcode
```

### 2. Install Dependencies
```bash
pip install numpy scipy matplotlib scikit-learn reportlab
# Install PyTorch to enable the deep learning autoencoder (recommended)
pip install torch
```

### 3. Run the Pipeline
```bash
python3 anomaly_detection_pipeline.py
```

This will:
1. Generate the realistic synthetic spatial-spectral scene (or load your own `.mat` data).
2. Fit the PCA + Global RX models, compute Local Mean RX, and train the PyTorch Autoencoder.
3. Print statistical metrics (ROC-AUC, PR-AUC, optimal threshold, FAR, Sensitivity).
4. Save the multi-panel comparison dashboard to `hyperspectral_anomaly_dashboard.png`.

### 4. Generate the PDF Report
To compile the highly detailed, publication-quality technical report PDF:
```bash
python3 generate_report_pdf.py
```
This generates `Hyperspectral_Pipeline_Report.pdf`.

---

## 📁 Repository Structure
```align
├── anomaly_detection_pipeline.py  # Core end-to-end pipeline execution script
├── generate_report_pdf.py          # PDF report compiler using ReportLab
├── Hyperspectral_Pipeline_Report.pdf # Publication-quality technical report PDF
├── hyperspectral_anomaly_dashboard.png # Generated comparison visualization dashboard
├── .gitignore                     # standard Python / OS git ignore configurations
└── README.md                      # Detailed project documentation and architecture guide
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
