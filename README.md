# Quantum Machine Learning for Tropical Cyclone Intensity Classification

This repository contains an end-to-end Python implementation of a Quantum Support Vector Machine (QSVM) pipeline for classifying the severity of tropical cyclones. Utilizing the global IBTrACS dataset, this project benchmarks specialized quantum feature maps—specifically Instantaneous Quantum Polynomial (IQP) circuits—against classical SVM algorithms, showcasing the potential of quantum-enhanced kernel methods for meteorological event classification.

## 🌪️ Project Overview

Predicting and classifying the intensity of tropical cyclones is a computationally complex problem. This project investigates whether mapping meteorological features into a high-dimensional quantum Hilbert space can provide a linear separation advantage over classical feature mapping techniques.

We classify storm conditions into three distinct categories based on maximum sustained wind speed:
- **Tropical System (TS):** < 64 knots
- **Moderate Hurricane (MH):** 64 - 95 knots
- **Severe Hurricane (SH):** ≥ 96 knots

The pipeline downloads raw IBTrACS v4 data, processes six core atmospheric features, constructs quantum feature maps manually (depth $L$ IQP, ZZ, Z), computes kernel overlap matrices natively via Qiskit's exact Statevector simulators, and finally trains multi-class Support Vector Machines to evaluate kernel-target alignment and performance.

## 🚀 Key Features

*   **Automated Data Pipeline:** `data_loader.py` securely downloads and caches the latest IBTrACS archives, intelligently selecting core observations required for modeling.
*   **Robust Preprocessing:** `preprocessing.py` implements GroupShuffle splits (grouped by Storm ID to prevent leakage), class balancing through undersampling, and zero-mean standardisation with $[\,0, \pi]\,$ mapping. 
*   **Manual IQP Construction:** `iqp_feature_map.py` builds parametrised depth-$L$ IQP (Instantaneous Quantum Polynomial) circuits gate-by-gate, bypassing deprecated library methods for complete control.
*   **Statevector Kernel Overlaps:** `quantum_kernels.py` leverages vectorized inner-product operations on statevectors simulating the quantum space, offering mathematically exact matrix formulations without shot noise.
*   **Comprehensive Baselines:** `classical_baselines.py` evaluates Linear, 3rd-degree Polynomial, and Radial Basis Function (RBF) classical kernels under robust 5-fold cross-validation grid searches.
*   **Evaluation & Alignment:** `evaluation.py` and `visualisation.py` yield macro-averaged F1/Cohen Kappa metrics, normalised heatmaps, and Frobenius inner-product measurements of Kernel-Target Alignment.

## 🛠️ Architecture

```
Quantum-cyclone-implementation/
├── requirements.txt
├── src/
│   ├── data_loader.py         # IBTrACS download & primary filtering
│   ├── preprocessing.py       # Grouped stratified split, balancing, standardization 
│   ├── iqp_feature_map.py     # Custom IQP circuit construction
│   ├── quantum_kernels.py     # Exact statevector evaluations for QSVM K-matrices
│   ├── classical_baselines.py # Baseline SVM cross-validation and hyperparameter selection
│   ├── evaluation.py          # Cohen's Kappa, Macro-F1, aligning scores
│   ├── visualisation.py       # Heatmaps & Circuit Depth evaluation plot generation
│   └── run_experiment.py      # Master execution orchestrator
├── figures/                   # Output folder for generated analysis charts
└── results/                   # JSON logs of trial runs and metrics
```

## 📦 Installation & Usage

It is recommended to run this project in a localized Python virtual environment. Python 3.10+ is required.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/quantum-cyclone-classification.git
   cd quantum-cyclone-classification
   ```

2. **Initialize Environment & Install Dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the End-to-End Experiment:**
   ```bash
   python src/run_experiment.py
   ```
   *Note: The first run will automatically download the ~250MB IBTrACS CSV dataset into a local `/data` directory.*

## 📊 Experimental Results

Using Qiskit's `AerSimulator`, exact statevectors were generated. The tests show the Quantum Support Vector classifiers correctly picking up the structure of the data compared to the Linear models.

As an example from a recently completed benchmark over 18,202 test samples (following $O(N^2)$ K-matrix construction scaling bounded subsetting):

| Method       | Depth | Accuracy | Precision | Recall | F1    | Kappa | Alignment |
|--------------|-------|----------|-----------|--------|-------|-------|-----------|
| QSVM-Z       | 2     | 96.87%   | 0.913     | 0.960  | 0.935 | 0.921 | 0.2657    |
| QSVM-IQP     | 2     | 93.62%   | 0.864     | 0.905  | 0.883 | 0.838 | 0.1679    |

### Generated Visualizations

**Normalised Confusion Matrix (QSVM-IQP L=2)**
This heatmap maps the model's predictive class probabilities across TS, MH, and SH ground truths.
![Confusion Matrix](figures/confusion_matrix.png)

**Circuit Depth Analysis**
Illustrates the effect of augmenting quantum layer depth $L$ for the Instantaneous Quantum Polynomial map.
![Depth Analysis](figures/depth_analysis.png)

## 🤝 Contributing

While this repository is primarily meant to act as a structured snapshot of a finished Quantum Machine Learning framework, contributions bridging additional quantum circuit topologies (like Hardware Efficient Ansätze) or optimizing K-matrix computation strategies (perhaps with Qiskit Primitives) are highly welcomed! Feel free to open issues or submit Pull Requests.

## 📄 License

This project is licensed under the MIT License. Data provided via [NOAA's IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive).
