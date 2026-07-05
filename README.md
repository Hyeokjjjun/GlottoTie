# GlottoTie Acoustic and EGG Analysis

This repository contains MATLAB scripts for the preprocessing, analysis, and feature extraction of Electroglottograph (EGG) and Microphone (Acoustic) signals. The tools are designed to analyze both **Connected Speech (CS)** and **Sustained Vowel (SV)** recordings. 

This code is provided to ensure full transparency and reproducibility of the data preprocessing and classification model inputs described in our study.

## 📁 Repository Contents

* **`GlottoTie_CS_analysis.m`**: Main script for Connected Speech (CS) analysis.
* **`GlottoTie_SV_analysis.m`**: Main script for Sustained Vowel (SV) analysis.
* **`CS_Normal_date_time_subject1_dob.csv`**: Sample data file for CS analysis.
* **`SV_Normal_date_time_subject1_dob.csv`**: Sample data file for SV analysis.
* **(Helper Functions)**: Ensure the following dependencies are in your MATLAB path:
    * `egg_preprocessing.m`
    * `cycle_init_detection.m` (Required for SV)
    * `cycle_separation.m` (Required for SV)
    * `synchronized_egg_degg_plot.m` (Required for SV)

## ⚙️ Prerequisites

* **MATLAB** (Tested on R2021a or newer)
* **Signal Processing Toolbox** (for filtering, FFT, `pspectrum`, `filtfilt`, etc.)
* **Audio Toolbox** (for `melSpectrogram`)

## 🚀 How to Run

Both scripts feature an interactive prompt upon execution, allowing the user to choose between two analysis modes: **Segmented (`s`)** or **Full Signal (`f`)**. 

To run the analysis, open MATLAB, set this repository as your working directory, and run either script.

### 1. Connected Speech (CS) Analysis
Run `GlottoTie_CS_analysis.m`. You will be prompted to enter a mode:

* **Mode `s` (Segmented Data Export for Machine Learning):**
    * Automatically preprocesses the EGG and Microphone signals.
    * Applies a 275Hz harmonics notch filter to the audio.
    * Splits the full signal into 3-second segments.
    * Exports these segments as `.mat` files into an `extracted_segments/` directory. This replicates the data pipeline used to train our classification models.
* **Mode `f` (Full Signal Visualization):**
    * Generates comprehensive visual plots of the entire recording.
    * Outputs include: Time-domain comparisons, FFT (Original vs. Notch Filtered), Power Spectrum Density (PSD), and Mel-Spectrograms overlayed with Spectral Centroids.

### 2. Sustained Vowel (SV) Analysis
Run `GlottoTie_SV_analysis.m`. You will be prompted to enter a mode:

* **Mode `s` (Cycle-Level Segmented Analysis):**
    * Analyzes pre-defined, stable segments of the sustained vowel (e.g., 4.0s - 6.2s).
    * Calculates the fundamental frequency ($f_0$).
    * Performs EGG cycle detection and separation.
    * Generates Synchronized EGG-dEGG phase portraits and extracts volumetric/ratio parameters ($V_{tot}, V_1 - V_4$).
* **Mode `f` (Full Signal Frequency Analysis):**
    * Performs macro-level FFT and notch filtering (275Hz harmonics) over the entire unsegmented recording.
    * Plots the time-domain signals alongside the raw and filtered frequency spectrums.

## 📊 Data Format
The provided sample `.csv` files contain two columns of raw data sampled at **11,000 Hz**:
1.  **Column 1:** Raw Electroglottograph (EGG) Data
2.  **Column 2:** Raw Microphone (Acoustic) Data

*(Note: Patient-identifying information has been removed or anonymized in the sample files to comply with privacy regulations.)*

## 📝 Citation & Reproducibility
This repository is published alongside our manuscript to fulfill data availability and reproducibility requirements. If you utilize this code or preprocessing methodology in your research, please cite our paper accordingly.