#!/usr/bin/env python3
"""
Hyperspectral Anomaly Detection Pipeline (Spectral-Spatial)
=========================================================
Author: Expert ML Engineer & Remote Sensing Specialist

This script implements a complete, end-to-end Python pipeline for spectral-spatial
anomaly detection in hyperspectral remote sensing datasets. It is designed to be:
1. Lightweight and highly efficient (minimal CPU/GPU memory footprint).
2. Statistically rigorous (implementing Global RX, Local Mean RX, and PyTorch Autoencoder).
3. Self-contained (generates realistic simulated data if no file is provided).
4. Robust in suppressing natural anomalies (boundaries, vegetation transitions) while
   highlighting compact manmade targets.

Pipeline Steps:
  1. Data Ingestion & Preprocessing (Loads MATLAB .mat or simulates data)
  2. Model Construction (Statistical PCA-Mahalanobis RX and PyTorch Autoencoder)
  3. Execution & Training Loops
  4. Statistical Evaluation (F1-Score, ROC-AUC, PR-AUC with adaptive thresholding)
  5. Visualization (Multi-panel comparison maps and statistical curves)
"""

import os
import time
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, roc_curve

# PyTorch imports for lightweight Deep Learning Model
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch is not installed. Deep learning autoencoder will be bypassed. "
          "Please install torch to enable the deep learning pipeline.")

# Set random seed for reproducibility
np.random.seed(42)
if TORCH_AVAILABLE:
    torch.manual_seed(42)


# ==========================================
# STEP 1: DATA INGESTION & PREPROCESSING
# ==========================================

class HyperspectralDataCube:
    """
    Handles loading of real hyperspectral files (.mat or .tiff) or generates
    a highly realistic simulated hyperspectral cube for testing.
    """
    def __init__(self, file_path=None, var_name=None, gt_name=None):
        self.file_path = file_path
        self.data_cube = None  # Shape: (Height, Width, Bands)
        self.ground_truth = None  # Shape: (Height, Width), binary [0, 1]
        self.height = 0
        self.width = 0
        self.bands = 0
        
        if file_path is not None and os.path.exists(file_path):
            self.load_mat_file(file_path, var_name, gt_name)
        else:
            print("[INFO] No external file path provided (or file not found). "
                  "Generating a highly realistic simulated spectral-spatial data cube...")
            self.generate_simulated_cube()

    def load_mat_file(self, file_path, var_name=None, gt_name=None):
        """
        Loads a standard IEEE Dataport MATLAB (.mat) file containing a 3D data cube
        and corresponding ground truth mask.
        """
        try:
            mat = scipy.io.loadmat(file_path)
            print(f"[INFO] Successfully loaded .mat file from {file_path}")
            print(f"Available keys in .mat: {list(mat.keys())}")
            
            # Autodetect data key if not provided
            if var_name is None:
                candidates = [k for k in mat.keys() if not k.startswith('__')]
                # Find the key that has a 3D array
                for c in candidates:
                    if isinstance(mat[c], np.ndarray) and mat[c].ndim == 3:
                        var_name = c
                        break
            
            if var_name is None or var_name not in mat:
                raise ValueError("Could not automatically locate the 3D data cube variable. "
                                 "Please specify 'var_name'.")
            
            self.data_cube = mat[var_name].astype(np.float32)
            self.height, self.width, self.bands = self.data_cube.shape
            print(f"[INFO] Loaded Data Cube shape: {self.data_cube.shape} (H x W x Bands)")
            
            # Autodetect ground truth key if not provided
            if gt_name is None:
                candidates = [k for k in mat.keys() if not k.startswith('__') and k != var_name]
                for c in candidates:
                    if isinstance(mat[c], np.ndarray) and mat[c].ndim == 2:
                        gt_name = c
                        break
            
            if gt_name is not None and gt_name in mat:
                self.ground_truth = mat[gt_name].astype(np.int32)
                # Binarize just in case
                self.ground_truth = (self.ground_truth > 0).astype(np.int32)
                print(f"[INFO] Loaded Ground Truth shape: {self.ground_truth.shape}")
            else:
                print("[WARNING] Ground truth labels not found in .mat. Creating a blank mask.")
                self.ground_truth = np.zeros((self.height, self.width), dtype=np.int32)
                
        except Exception as e:
            print(f"[ERROR] Failed to load .mat file: {e}")
            print("[INFO] Falling back to simulated data cube.")
            self.generate_simulated_cube()

    def generate_simulated_cube(self):
        """
        Generates a highly realistic 3D hyperspectral cube containing:
        - 3 distinct background regions (grasslands, soil, and a diagonal river/road boundary)
          to simulate continuous, natural spatial-spectral transitions.
        - Natural anomalies (e.g., local natural patch variations) that the models must suppress.
        - True manmade anomalies: small, localized, compact shapes with highly distinct
          spectral signatures (metallic comb reflections).
        - Multi-band Gaussian noise (SNR = ~30dB) to model sensor imperfections.
        """
        self.height, self.width, self.bands = 100, 100, 100
        H, W, B = self.height, self.width, self.bands
        
        # 1. Define synthetic baseline spectra for backgrounds (using mathematical curves)
        wavelengths = np.linspace(400, 2500, B) # 400nm to 2500nm
        
        # Background A: Vegetation-like (high reflection in NIR/SWIR, absorption in red)
        spec_veg = 0.1 + 0.6 * (1.0 / (1.0 + np.exp(-(wavelengths - 700)/50))) # Chlorophyll red-edge
        spec_veg += 0.15 * np.sin(wavelengths / 150)
        spec_veg = np.clip(spec_veg, 0.0, 1.0)
        
        # Background B: Soil-like (gradual rise, flatter)
        spec_soil = 0.15 + 0.45 * (wavelengths - 400) / 2100
        spec_soil += 0.05 * np.cos(wavelengths / 300)
        spec_soil = np.clip(spec_soil, 0.0, 1.0)
        
        # Background C: Water/River-like (absorbs almost everything, peaks slightly in blue-green)
        spec_water = 0.12 * np.exp(-(wavelengths - 500)/200)
        spec_water = np.clip(spec_water, 0.01, 1.0)

        # 2. Build the spatial distribution of background classes
        self.data_cube = np.zeros((H, W, B), dtype=np.float32)
        
        for y in range(H):
            for x in range(W):
                # Generate a smooth transition between Vegetation (A) and Soil (B)
                # left side has more vegetation, right side has more soil
                weight_soil = 1.0 / (1.0 + np.exp(-(x - W/2) / 10.0))
                weight_veg = 1.0 - weight_soil
                
                mixed_spectrum = weight_veg * spec_veg + weight_soil * spec_soil
                
                # Superimpose a diagonal river structure (Background C)
                # Equation of line: y = x + offset. Width of river = 6 pixels.
                river_center = x - 10
                distance_to_river = abs(y - river_center)
                if distance_to_river < 4:
                    # Pure water
                    mixed_spectrum = spec_water
                elif distance_to_river < 8:
                    # Transition riverbank
                    blend = (distance_to_river - 4) / 4.0
                    mixed_spectrum = blend * mixed_spectrum + (1 - blend) * spec_water
                
                self.data_cube[y, x] = mixed_spectrum
        
        # 3. Add a "Natural Anomaly" (suppression target)
        # A localized patch representing dry soil or water turbidity. 
        # It's a natural variation, so the model should successfully SUPPRESS it.
        # Let's put a natural circular patch at (y=30, x=75) with radius 8
        cy_nat, cx_nat, r_nat = 30, 75, 8
        for y in range(H):
            for x in range(W):
                dist = np.sqrt((y - cy_nat)**2 + (x - cx_nat)**2)
                if dist < r_nat:
                    # Blend in slightly drier soil spectrum (slight peak shift)
                    blend = (1.0 - (dist / r_nat)) * 0.4
                    self.data_cube[y, x] = (1.0 - blend) * self.data_cube[y, x] + blend * (spec_soil * 1.2)

        # 4. Insert True Manmade Anomalies and build Ground Truth
        # Manmade anomalies should be highly localized, geometric, and spectrally distinct.
        self.ground_truth = np.zeros((H, W), dtype=np.int32)
        
        # Anomaly 1: Small metal target (3x3) at (y=25, x=25)
        # Spectrum is highly active with sharp metallic comb reflection bands
        spec_anom1 = 0.2 + 0.6 * (np.sin(wavelengths / 40) > 0.3).astype(float)
        self.inject_anomaly(y_start=24, x_start=24, size=3, spectrum=spec_anom1)
        
        # Anomaly 2: Tiny compact target (2x2) at (y=75, x=75)
        # Spectrum is flat, highly reflective (mirror-like/white tarp)
        spec_anom2 = 0.85 * np.ones(B)
        self.inject_anomaly(y_start=74, x_start=74, size=2, spectrum=spec_anom2)
        
        # Anomaly 3: Single-pixel target at (y=50, x=15) - representing a subpixel object
        # Spectrum has a distinct sharp emission/absorption peak
        spec_anom3 = 0.1 * np.ones(B)
        spec_anom3[35:45] = 0.95 # Sharp peak in the middle bands
        self.inject_anomaly(y_start=50, x_start=15, size=1, spectrum=spec_anom3)

        # 5. Add sensor noise (Gaussian white noise)
        noise = np.random.normal(loc=0.0, scale=0.02, size=(H, W, B)).astype(np.float32)
        self.data_cube += noise
        self.data_cube = np.clip(self.data_cube, 0.0, 1.0)
        
        print(f"[INFO] Simulated Data Cube successfully generated. Shape: {self.data_cube.shape}")
        print(f"[INFO] Ground Truth anomalies injected at (25,25), (75,75), and (50,15).")

    def inject_anomaly(self, y_start, x_start, size, spectrum):
        """Helper to inject a target spectrum and mark it in ground truth."""
        for y in range(y_start, y_start + size):
            for x in range(x_start, x_start + size):
                if 0 <= y < self.height and 0 <= x < self.width:
                    self.data_cube[y, x] = spectrum
                    self.ground_truth[y, x] = 1

    def get_flattened_data(self):
        """Reshapes the H x W x B cube to a 2D matrix of shape (N, B) where N = H * W."""
        return self.data_cube.reshape(-1, self.bands)


# ==========================================
# STEP 2: MODEL CONSTRUCTION (LIGHTWEIGHT)
# ==========================================

class PCAMahalanobisDetector:
    """
    Lightweight, statistically rigorous statistical anomaly detector.
    Combines:
    1. Principal Component Analysis (PCA) for spectral dimension reduction and noise suppression.
    2. Reed-Xiaoli (RX) Anomaly Detection (Global RX & Local Mean Subtraction RX).
    """
    def __init__(self, n_components=8):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.scaler = StandardScaler()
        self.global_mean_ = None
        self.global_cov_inv_ = None

    def fit_global(self, X_flat):
        """
        Fits PCA and calculates the global background distribution statistics.
        X_flat: numpy array of shape (N, B)
        """
        start_time = time.time()
        # Scale and fit PCA
        X_scaled = self.scaler.fit_transform(X_flat)
        self.X_pca = self.pca.fit_transform(X_scaled)  # Shape: (N, n_components)
        
        # Calculate Global Mean and Covariance Matrix in PCA subspace
        self.global_mean_ = np.mean(self.X_pca, axis=0)
        cov_matrix = np.cov(self.X_pca, rowvar=False)
        
        # Use pseudo-inverse for high numerical stability
        self.global_cov_inv_ = np.linalg.pinv(cov_matrix)
        
        elapsed = time.time() - start_time
        print(f"[INFO] Statistical Model fit completed in {elapsed:.4f}s. "
              f"PCA explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.4f}")

    def compute_global_rx(self):
        """
        Computes Global RX (Mahalanobis distance of each pixel to global mean)
        Returns:
            anomaly_map: 2D array of shape (Height, Width)
        """
        # Calculate Mahalanobis distance in PCA subspace
        # r = (x - mu)^T * Sigma^-1 * (x - mu)
        diff = self.X_pca - self.global_mean_
        scores = np.sum(diff @ self.global_cov_inv_ * diff, axis=1)
        return scores

    def compute_local_mean_rx(self, height, width, window_size=11):
        """
        An ultra-fast, robust, and highly efficient spatial-spectral local detector.
        
        Instead of evaluating slow dual-window covariance matrices per pixel,
        this algorithm subtracts the local spatial mean (box filter) from the PCA features,
        removing localized natural spatial variations (suppressing boundaries), and
        computes the Mahalanobis distance of these local spatial residuals using the 
        global background covariance matrix.
        
        This effectively acts as a high-pass spatial-spectral filter, highlighting 
        compact manmade targets and heavily suppressing natural transitions.
        """
        start_time = time.time()
        X_pca_spatial = self.X_pca.reshape(height, width, self.n_components)
        local_mean = np.zeros_like(X_pca_spatial)
        
        # Vectorized box-filter spatial mean subtraction
        half_w = window_size // 2
        for b in range(self.n_components):
            # Pad spatial dimensions to handle boundaries elegantly
            padded = np.pad(X_pca_spatial[:, :, b], half_w, mode='edge')
            # Pad with 0 row and column at the top and left to make cumsum index boundaries work perfectly
            padded_zero = np.pad(padded, ((1, 0), (1, 0)), mode='constant', constant_values=0)
            # Cumulative sum tricks for O(1) sliding window average
            cumsum = np.cumsum(np.cumsum(padded_zero, axis=0), axis=1)
            
            # Extract box sums
            box_sums = (cumsum[window_size:, window_size:] 
                        - cumsum[:-window_size, window_size:] 
                        - cumsum[window_size:, :-window_size] 
                        + cumsum[:-window_size, :-window_size])
            local_mean[:, :, b] = box_sums / (window_size * window_size)
            
        # Compute spatial-spectral residual
        residuals = (X_pca_spatial - local_mean).reshape(-1, self.n_components)
        
        # Calculate Mahalanobis distance of residuals using global covariance structure
        scores = np.sum(residuals @ self.global_cov_inv_ * residuals, axis=1)
        elapsed = time.time() - start_time
        print(f"[INFO] Local Mean RX computed in {elapsed:.4f}s.")
        return scores


# ==========================================
# PYTORCH SPECTRAL AUTOENCODER (DEEP LEARNING)
# ==========================================

if TORCH_AVAILABLE:
    class PyTorchAutoencoder(nn.Module):
        """
        A compact, lightweight Multi-Layer Perceptron Autoencoder for hyperspectral data.
        It compresses spectral bands into a tight latent space and tries to reconstruct them.
        Since anomalies are statistically rare, it learns to reconstruct the background perfectly
        while failing to reconstruct anomalous manmade materials, yielding high reconstruction errors.
        """
        def __init__(self, in_features, latent_dim=4):
            super().__init__()
            # Lightweight encoder
            self.encoder = nn.Sequential(
                nn.Linear(in_features, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, latent_dim)
            )
            # Lightweight decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Linear(64, in_features)
            )

        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed


    class AutoencoderDetector:
        """
        Wrapper to handle scaling, batch training, and score computation
        for the PyTorch Spectral Autoencoder.
        """
        def __init__(self, in_features, latent_dim=4, lr=0.003, epochs=20, batch_size=256):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                       ('mps' if torch.backends.mps.is_available() else 'cpu'))
            print(f"[INFO] Autoencoder will run on device: {self.device}")
            self.model = PyTorchAutoencoder(in_features, latent_dim).to(self.device)
            self.scaler = StandardScaler()
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size

        def fit(self, X_flat):
            """Trains the Autoencoder on the entire image dataset."""
            start_time = time.time()
            X_scaled = self.scaler.fit_transform(X_flat)
            
            # Prepare PyTorch DataLoader
            tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
            dataset = TensorDataset(tensor_x)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
            criterion = nn.MSELoss()
            
            self.model.train()
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch in loader:
                    inputs = batch[0].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, inputs)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item() * inputs.size(0)
                
                # Logging occasionally
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1:02d}/{self.epochs:02d} | Avg Loss: {epoch_loss / len(X_flat):.6f}")
            
            elapsed = time.time() - start_time
            print(f"[INFO] Autoencoder training completed in {elapsed:.4f}s.")

        def compute_anomaly_scores(self, X_flat):
            """
            Computes anomaly scores based on Reconstruction Error.
            Combines Mean Squared Error (MSE) and Spectral Angle Mapper (SAM).
            """
            self.model.eval()
            X_scaled = self.scaler.transform(X_flat)
            tensor_x = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                reconstructed = self.model(tensor_x)
                
                # 1. Compute MSE in scaled feature space
                mse_scores = torch.mean((tensor_x - reconstructed) ** 2, dim=1).cpu().numpy()
                
                # 2. Compute Spectral Angle Mapper (SAM) - mathematically robust to illumination intensity
                # SAM = arccos( (x . x_hat) / (||x|| * ||x_hat||) )
                eps = 1e-8
                dot_product = torch.sum(tensor_x * reconstructed, dim=1)
                norm_x = torch.norm(tensor_x, p=2, dim=1)
                norm_recon = torch.norm(reconstructed, p=2, dim=1)
                cosine_sim = dot_product / (norm_x * norm_recon + eps)
                cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
                sam_scores = torch.acos(cosine_sim).cpu().numpy()
                
            # Normalize and combine both metrics (50% MSE, 50% SAM) for optimal manmade detection
            mse_norm = (mse_scores - mse_scores.min()) / (mse_scores.max() - mse_scores.min() + eps)
            sam_norm = (sam_scores - sam_scores.min()) / (sam_scores.max() - sam_scores.min() + eps)
            
            combined_scores = 0.5 * mse_norm + 0.5 * sam_norm
            return combined_scores


# ==========================================
# STEP 4: STATISTICAL EVALUATION
# ==========================================

class PipelineEvaluator:
    """
    Performs comprehensive statistical evaluation of anomaly detection scores
    against the expert ground truth map.
    """
    @staticmethod
    def evaluate(ground_truth_flat, anomaly_scores_flat, model_name="Model"):
        """
        Calculates ROC-AUC, PR-AUC, and adaptive F1 score.
        """
        gt = ground_truth_flat.astype(int)
        
        # Calculate ROC-AUC
        roc_auc = roc_auc_score(gt, anomaly_scores_flat)
        
        # Calculate Precision-Recall Curve & PR-AUC
        precisions, recalls, thresholds = precision_recall_curve(gt, anomaly_scores_flat)
        pr_auc = auc(recalls, precisions)
        
        # Adaptive Thresholding: 
        # Standard in remote sensing anomaly detection is to set a threshold 
        # based on a target False Alarm Rate (FAR), e.g., 1.5% or using Otsu's method.
        # Here we choose the threshold that maximizes the F1 Score (optimal oracle threshold)
        # to find the best achievable performance.
        f1_scores = []
        # Subsample thresholds to keep F1 calculation fast
        step = max(1, len(thresholds) // 500)
        test_thresholds = thresholds[::step]
        
        best_f1 = 0.0
        best_thresh = 0.5
        for t in test_thresholds:
            preds = (anomaly_scores_flat >= t).astype(int)
            f1 = f1_score(gt, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
                
        # If no optimal binarization was found, fallback to 98.5th percentile (1.5% FAR)
        if best_f1 == 0.0:
            best_thresh = np.percentile(anomaly_scores_flat, 98.5)
            preds = (anomaly_scores_flat >= best_thresh).astype(int)
            best_f1 = f1_score(gt, preds, zero_division=0)
            
        print(f"\n==========================================")
        print(f" STATISTICAL EVALUATION: {model_name.upper()}")
        print(f"==========================================")
        print(f" * ROC-AUC (Area Under ROC Curve)     : {roc_auc:.5f}")
        print(f" * PR-AUC (Precision-Recall AUC)       : {pr_auc:.5f}")
        print(f" * Maximum Achievable F1-Score        : {best_f1:.5f}")
        print(f" * Optimal Detection Threshold         : {best_thresh:.5f}")
        
        # Calculate False Alarm Rate and True Positive Rate at best threshold
        preds_best = (anomaly_scores_flat >= best_thresh).astype(int)
        tp = np.sum((preds_best == 1) & (gt == 1))
        fp = np.sum((preds_best == 1) & (gt == 0))
        fn = np.sum((preds_best == 0) & (gt == 1))
        tn = np.sum((preds_best == 0) & (gt == 0))
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f" * True Positive Rate (Sensitivity)   : {tpr:.2%}")
        print(f" * False Alarm Rate (FAR)             : {far:.2%}")
        print(f"==========================================")
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1': best_f1,
            'threshold': best_thresh,
            'predictions': preds_best,
            'roc_curve': roc_curve(gt, anomaly_scores_flat),
            'pr_curve': (recalls, precisions)
        }


# ==========================================
# STEP 5: VISUALIZATION SUITE
# ==========================================

def plot_pipeline_results(dataset, results_dict, save_path="anomaly_detection_results.png"):
    """
    Creates a comprehensive, high-resolution multi-panel matplotlib dashboard
    comparing RGB composite, ground truth, continuous anomaly heatmaps,
    binary prediction maps, and standard ROC/PR curves.
    """
    num_models = len(results_dict)
    fig = plt.figure(figsize=(16, 4 * num_models + 4), facecolor='#f8f9fa')
    plt.suptitle("Spectral-Spatial Hyperspectral Anomaly Detection Dashboard", 
                 fontsize=18, fontweight='bold', color='#1a252c', y=0.98)
    
    # 1. RGB Band Composite representation
    # Select channels close to Red (e.g. 60), Green (e.g. 30), Blue (e.g. 10)
    r_band = dataset.data_cube[:, :, min(60, dataset.bands-1)]
    g_band = dataset.data_cube[:, :, min(30, dataset.bands-1)]
    b_band = dataset.data_cube[:, :, min(10, dataset.bands-1)]
    
    # Normalize RGB representation for visualization
    rgb = np.stack([r_band, g_band, b_band], axis=2)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    
    # Plot RGB composite
    ax1 = plt.subplot2grid((num_models + 1, 4), (0, 0))
    ax1.imshow(rgb)
    ax1.set_title("1. Pseudo-RGB Composite\n(Bands: Red, Green, Blue)", fontsize=11, fontweight='semibold')
    ax1.axis('off')
    
    # Plot Ground Truth Anomalies
    ax2 = plt.subplot2grid((num_models + 1, 4), (0, 1))
    im_gt = ax2.imshow(dataset.ground_truth, cmap='inferno')
    ax2.set_title("2. Expert Ground Truth Map\n(1 = True Anomalies)", fontsize=11, fontweight='semibold')
    ax2.axis('off')
    # Add a thin green boundary highlighting targets in ground truth
    y_indices, x_indices = np.where(dataset.ground_truth == 1)
    ax2.scatter(x_indices, y_indices, s=25, facecolors='none', edgecolors='#00FF00', linewidths=0.5)

    # Plot Background visual indicator (river highlight)
    ax3 = plt.subplot2grid((num_models + 1, 4), (0, 2), colspan=2)
    # Average across all spectral bands to show background structures
    avg_img = np.mean(dataset.data_cube, axis=2)
    ax3.imshow(avg_img, cmap='bone')
    ax3.set_title("3. Spatial Background Avg Intensity\n(Natural Rivers & Transitions)", fontsize=11, fontweight='semibold')
    ax3.axis('off')

    # Plot model results row by row
    for idx, (model_name, res) in enumerate(results_dict.items(), start=1):
        # Scale score map to [0, 1] for visual display
        score_map = res['scores'].reshape(dataset.height, dataset.width)
        score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
        
        # Heatmap
        ax_heat = plt.subplot2grid((num_models + 1, 4), (idx, 0))
        im_heat = ax_heat.imshow(score_map, cmap='jet')
        ax_heat.set_title(f"{model_name}\nContinuous Anomaly Scores", fontsize=11, fontweight='semibold')
        ax_heat.axis('off')
        fig.colorbar(im_heat, ax=ax_heat, fraction=0.046, pad=0.04)
        
        # Binarized Predictions
        pred_map = res['predictions'].reshape(dataset.height, dataset.width)
        ax_pred = plt.subplot2grid((num_models + 1, 4), (idx, 1))
        ax_pred.imshow(pred_map, cmap='magma')
        ax_pred.set_title(f"{model_name}\nBinarized Detections (Max F1)", fontsize=11, fontweight='semibold')
        ax_pred.axis('off')
        
        # ROC Curve
        ax_roc = plt.subplot2grid((num_models + 1, 4), (idx, 2))
        fpr, tpr, _ = res['roc_curve']
        ax_roc.plot(fpr, tpr, color='#007acc', lw=2, label=f"ROC-AUC: {res['roc_auc']:.4f}")
        ax_roc.plot([0, 1], [0, 1], color='#a5b1b8', linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f"{model_name} ROC Curve")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, linestyle=':', alpha=0.6)
        
        # PR Curve
        ax_pr = plt.subplot2grid((num_models + 1, 4), (idx, 3))
        recalls, precisions = res['pr_curve']
        ax_pr.plot(recalls, precisions, color='#e056fd', lw=2, label=f"PR-AUC: {res['pr_auc']:.4f}")
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f"{model_name} Precision-Recall")
        ax_pr.legend(loc="lower left")
        ax_pr.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[INFO] Dashboard successfully plotted and saved to: {os.path.abspath(save_path)}")


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

def main():
    print("="*60)
    print("HYPERSPECTRAL ANOMALY DETECTION PIPELINE")
    print("="*60)
    
    # 1. Ingest/Simulate Data
    # To run on real data, replace with actual path, e.g. "SanDiego.mat"
    cube = HyperspectralDataCube(file_path=None)
    
    X_flat = cube.get_flattened_data()
    gt_flat = cube.ground_truth.flatten()
    
    results = {}
    
    # 2. RUN STATISTICAL MODEL (PCA + GLOBAL & LOCAL RX)
    print("\n--- Fitting Statistical Model (PCA + RX) ---")
    stat_detector = PCAMahalanobisDetector(n_components=8)
    stat_detector.fit_global(X_flat)
    
    # Compute Global RX
    global_rx_scores = stat_detector.compute_global_rx()
    print("[INFO] Evaluating Global RX...")
    results['PCA-Global RX'] = PipelineEvaluator.evaluate(gt_flat, global_rx_scores, "PCA Global RX")
    results['PCA-Global RX']['scores'] = global_rx_scores
    
    # Compute Local Mean RX (De-biased Spatial-Spectral RX)
    # Use window_size=11 to isolate local structures (manmade targets) 
    # while suppressing large background variations (river banks, grass/soil border)
    local_rx_scores = stat_detector.compute_local_mean_rx(cube.height, cube.width, window_size=11)
    print("[INFO] Evaluating Local Mean RX...")
    results['PCA-Local Mean RX'] = PipelineEvaluator.evaluate(gt_flat, local_rx_scores, "PCA Local Mean RX")
    results['PCA-Local Mean RX']['scores'] = local_rx_scores
    
    # 3. RUN DEEP LEARNING MODEL (PYTORCH AUTOENCODER)
    if TORCH_AVAILABLE:
        print("\n--- Training Deep Learning Model (PyTorch Autoencoder) ---")
        # Lightweight autoencoder: bottleneck is 4 components (extremely lightweight)
        ae_detector = AutoencoderDetector(in_features=cube.bands, latent_dim=4, lr=0.003, epochs=25, batch_size=128)
        ae_detector.fit(X_flat)
        
        ae_scores = ae_detector.compute_anomaly_scores(X_flat)
        print("[INFO] Evaluating Spectral Autoencoder...")
        results['Spectral Autoencoder'] = PipelineEvaluator.evaluate(gt_flat, ae_scores, "Spectral Autoencoder")
        results['Spectral Autoencoder']['scores'] = ae_scores
    else:
        print("\n[INFO] Skipping PyTorch Spectral Autoencoder (PyTorch not available).")
        
    # 4. PLOT DASHBOARD
    dashboard_filename = "hyperspectral_anomaly_dashboard.png"
    plot_pipeline_results(cube, results, save_path=dashboard_filename)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
