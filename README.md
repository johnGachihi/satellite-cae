This repository contains the implementation of Masked Autoencoders (MAE) and Context Autoencoders (CAE) for pretraining and fine-tuning on multi-spectral satellite imagery datasets, specifically for tasks such as flood mapping and land-use classification.

### 1. Project Overview
Name: Context Autoencoders for Pretraining Foundation Models for Satellite Imagery Tasks
Objective: Evaluate and compare the effectiveness of MAE and CAE pretraining strategies on multi-spectral satellite imagery for downstream tasks.
Datasets:
Pretraining: FMOW-Sentinel (Sentinel-2 imagery with 13 spectral bands).
Fine-tuning Tasks:
-Land-Use Classification: FMOW-Sentinel dataset (62 classes).
-Flood Mapping: Sen1Floods11 dataset.
###2. Repository Structure
├── data/
│   ├── fmow_sentinel/       # FMOW-Sentinel dataset files
│   ├── sen1floods11/        # Sen1Floods11 dataset files
├── models/
│   ├── mae.py               # MAE implementation
│   ├── cae.py               # CAE implementation
├── scripts/
│   ├── run_pretraining.py   # Script for pretraining
│   ├── run_finetuning.py    # Script for fine-tuning
├── utils/
│   ├── dataset_loader.py    # Dataset loading utilities
│   ├── evaluation.py        # Evaluation metrics implementation
├── results/
│   ├── mae_performance.csv  # MAE performance logs
│   ├── cae_performance.csv  # CAE performance logs
├── README.md                # Project documentation
###3.Setup
git clone https://github.com/johnGachihi/satellite-cae.git

