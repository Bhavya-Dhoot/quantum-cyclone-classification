import os
import json
import logging
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import rbf_kernel

# Set random seed globally for reproducibility
np.random.seed(42)

from data_loader import download_data, load_and_filter_data
from preprocessing import get_processed_data
from iqp_feature_map import build_iqp_feature_map
from quantum_kernels import build_z_feature_map, build_zz_feature_map
from qiskit_aer import AerSimulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def subset_balanced_data(X, y, samples_per_class, seed):
    np.random.seed(seed)
    indices = []
    classes = np.unique(y)
    for c in classes:
        c_idxs = np.where(y == c)[0]
        chosen = np.random.choice(c_idxs, samples_per_class, replace=False)
        indices.extend(chosen)
    np.random.shuffle(indices)
    return X[indices], y[indices]

def compute_sv_gram_matrix(X_train, X_test, fm_func, **kwargs):
    # From quantum_kernels compute_statevectors logic
    n_train = len(X_train)
    n_features = X_train.shape[1]
    backend = AerSimulator(method='statevector')
    
    # Train svs
    sv_train = []
    for i in range(n_train):
        qc = fm_func(n_features, X_train[i], **kwargs)
        qc.save_statevector()
        result = backend.run(qc).result()
        sv_train.append(np.asarray(result.get_statevector(qc)))
    sv_train_matrix = np.vstack(sv_train)
    K_train_complex = sv_train_matrix @ sv_train_matrix.T.conj()
    K_train = np.abs(K_train_complex) ** 2
    K_train = (K_train + K_train.T) / 2
    
    # Test svs
    if X_test is not None and len(X_test) > 0:
        n_test = len(X_test)
        sv_test = []
        for i in range(n_test):
            qc = fm_func(n_features, X_test[i], **kwargs)
            qc.save_statevector()
            result = backend.run(qc).result()
            sv_test.append(np.asarray(result.get_statevector(qc)))
        sv_test_matrix = np.vstack(sv_test)
        K_test_complex = sv_test_matrix @ sv_train_matrix.T.conj()
        K_test = np.abs(K_test_complex) ** 2
    else:
        K_test = None
        
    return K_train, K_test

def test_stability(X_train_full, y_train_full, X_test_full, y_test_full):
    logger.info("=== 1. Statistical Stability Analysis ===")
    results = []
    samples_per_class = 200 # match original paper
    n_runs = 10
    
    # Pre-select test set to keep evaluation time bounded. Original had ~46k test points, 
    # we'll use a balanced random subset of the test set for the 10 iterations to save kernel compute time.
    X_test_sub, y_test_sub = subset_balanced_data(X_test_full, y_test_full, 100, 42) 
    
    for i in range(n_runs):
        seed = 42 + i
        X_train_q, y_train_q = subset_balanced_data(X_train_full, y_train_full, samples_per_class, seed)
        
        K_train, K_test = compute_sv_gram_matrix(X_train_q, X_test_sub, build_iqp_feature_map, L=2)
        
        clf = SVC(kernel='precomputed')
        clf.fit(K_train, y_train_q)
        y_pred = clf.predict(K_test)
        
        acc = accuracy_score(y_test_sub, y_pred)
        f1 = f1_score(y_test_sub, y_pred, average='macro')
        results.append((acc, f1))
        logger.info(f"Run {i+1}: Acc={acc:.4f}, F1={f1:.4f}")
        
    accs = [r[0] for r in results]
    f1s = [r[1] for r in results]
    
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    ci_acc = 1.96 * std_acc / np.sqrt(n_runs)
    
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    ci_f1 = 1.96 * std_f1 / np.sqrt(n_runs)
    
    logger.info(f"Stability Results -> Mean Acc: {mean_acc:.4f} \u00b1 {ci_acc:.4f}")
    logger.info(f"Stability Results -> Mean F1: {mean_f1:.4f} \u00b1 {ci_f1:.4f}")
    
    with open("results/statistical_stability.txt", "w") as f:
        f.write(f"Mean Accuracy: {mean_acc:.4f} +/- {ci_acc:.4f} (95% CI)\n")
        f.write(f"Std Accuracy: {std_acc:.4f}\n")
        f.write(f"Mean F1: {mean_f1:.4f} +/- {ci_f1:.4f} (95% CI)\n")

def plot_kernel_heatmaps(X_train_full, y_train_full):
    logger.info("=== 2. Kernel Matrix Visualization ===")
    X_sub, y_sub = subset_balanced_data(X_train_full, y_train_full, 30, 42) # 90x90 matrix
    
    # Compute RBF
    gamma = 1.0 / (X_sub.shape[1] * X_sub.var())
    K_rbf = rbf_kernel(X_sub, X_sub, gamma=gamma)
    
    # Compute quantum kernels
    K_z, _ = compute_sv_gram_matrix(X_sub, None, build_z_feature_map)
    K_zz, _ = compute_sv_gram_matrix(X_sub, None, build_zz_feature_map)
    K_iqp, _ = compute_sv_gram_matrix(X_sub, None, build_iqp_feature_map, L=2)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 11))
    kernels = [("RBF", K_rbf), ("QSVM-Z", K_z), ("QSVM-ZZ", K_zz), ("QSVM-IQP L=2", K_iqp)]
    
    for i, (name, K) in enumerate(kernels):
        ax = axs[i//2, i%2]
        im = ax.imshow(K, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"{name} Kernel Matrix")
        fig.colorbar(im, ax=ax)
        
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/kernel_heatmaps.png", dpi=300)
    plt.close()
    
    return K_rbf, K_z, K_zz, K_iqp

def plot_eigenvalue_spectrum(K_rbf, K_z, K_zz, K_iqp):
    logger.info("=== 3. Kernel Eigenvalue Spectrum ===")
    
    def get_sorted_eigs(K):
        # eps = 1e-10 to avoid log(0) if any eigenvalues are exactly 0, np.linalg.eigvalsh returns real parts
        eigs = np.linalg.eigvalsh(K)
        # return sorted descending
        return np.sort(eigs)[::-1]
        
    eigs_rbf = get_sorted_eigs(K_rbf)
    eigs_z = get_sorted_eigs(K_z)
    eigs_zz = get_sorted_eigs(K_zz)
    eigs_iqp = get_sorted_eigs(K_iqp)
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.maximum(eigs_rbf, 1e-15), label="RBF", marker='o', alpha=0.7)
    plt.plot(np.maximum(eigs_z, 1e-15), label="QSVM-Z", marker='s', alpha=0.7)
    plt.plot(np.maximum(eigs_zz, 1e-15), label="QSVM-ZZ", marker='^', alpha=0.7)
    plt.plot(np.maximum(eigs_iqp, 1e-15), label="QSVM-IQP L=2", marker='d', alpha=0.7)
    
    plt.yscale('log')
    plt.ylabel('Eigenvalue (Log Scale)')
    plt.xlabel('Index')
    plt.title('Kernel Eigenvalue Spectra Analysis')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig("figures/eigenvalue_spectrum.png", dpi=300)
    plt.close()

def test_ablation(X_train_full, y_train_full, X_test_full, y_test_full):
    logger.info("=== 4. Feature Ablation Study ===")
    features = ['Wind Speed', 'Pressure', 'Latitude', 'Longitude', 'RMW', 'Translational Velocity']
    
    X_train_q, y_train_q = subset_balanced_data(X_train_full, y_train_full, 200, 42)
    X_test_sub, y_test_sub = subset_balanced_data(X_test_full, y_test_full, 100, 42)
    
    baseline_K_train, baseline_K_test = compute_sv_gram_matrix(X_train_q, X_test_sub, build_iqp_feature_map, L=2)
    clf = SVC(kernel='precomputed')
    clf.fit(baseline_K_train, y_train_q)
    baseline_acc = accuracy_score(y_test_sub, clf.predict(baseline_K_test))
    
    ablation_results = {}
    for i, feature_name in enumerate(features):
        logger.info(f"Ablating feature: {feature_name}")
        X_train_ablated = X_train_q.copy()
        X_test_ablated = X_test_sub.copy()
        
        # Ablation = mean substitution. Data is standardized (mean=0), so setting to 0 acts as mean substitution.
        X_train_ablated[:, i] = 0.0
        X_test_ablated[:, i] = 0.0
        
        K_train, K_test = compute_sv_gram_matrix(X_train_ablated, X_test_ablated, build_iqp_feature_map, L=2)
        clf = SVC(kernel='precomputed')
        clf.fit(K_train, y_train_q)
        acc = accuracy_score(y_test_sub, clf.predict(K_test))
        ablation_results[feature_name] = acc
        
    labels = list(ablation_results.keys())
    drops = [baseline_acc - ablation_results[k] for k in labels]
    
    # Sort by drop impact
    sorted_idx = np.argsort(drops)[::-1]
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_drops = [drops[i] for i in sorted_idx]
    
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_labels, sorted_drops, color='coral')
    plt.ylabel('Accuracy Drop (Baseline - Ablated)')
    plt.title('Feature Ablation Importance (QSVM-IQP L=2)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/feature_ablation.png", dpi=300)
    plt.close()
    
    with open("results/ablation_table.json", "w") as f:
        json.dump(ablation_results, f, indent=4)

def test_scaling(X_train_full, y_train_full, X_test_full, y_test_full):
    logger.info("=== 5. Training Size Scaling Analysis ===")
    sizes = [50, 100, 200, 300]
    results_acc = []
    
    X_test_sub, y_test_sub = subset_balanced_data(X_test_full, y_test_full, 100, 42)
    
    for size in sizes:
        logger.info(f"Evaluating Training Size: {size} per class ({size*3} total)")
        X_train_q, y_train_q = subset_balanced_data(X_train_full, y_train_full, size, 42)
        
        K_train, K_test = compute_sv_gram_matrix(X_train_q, X_test_sub, build_iqp_feature_map, L=2)
        clf = SVC(kernel='precomputed')
        clf.fit(K_train, y_train_q)
        acc = accuracy_score(y_test_sub, clf.predict(K_test))
        results_acc.append(acc)
        
    plt.figure(figsize=(8, 6))
    plt.plot([s * 3 for s in sizes], results_acc, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Total Training Size')
    plt.ylabel('Accuracy')
    plt.title('Training Size Scaling (QSVM-IQP L=2)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("figures/training_scaling.png", dpi=300)
    plt.close()
    
    with open("results/scaling_table.json", "w") as f:
        json.dump({str(s): acc for s, acc in zip(sizes, results_acc)}, f, indent=4)

def main():
    download_data()
    raw_df = load_and_filter_data()
    X_train_full, y_train_full, X_test_full, y_test_full = get_processed_data(raw_df, random_state=42)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    test_stability(X_train_full, y_train_full, X_test_full, y_test_full)
    K_rbf, K_z, K_zz, K_iqp = plot_kernel_heatmaps(X_train_full, y_train_full)
    plot_eigenvalue_spectrum(K_rbf, K_z, K_zz, K_iqp)
    test_ablation(X_train_full, y_train_full, X_test_full, y_test_full)
    test_scaling(X_train_full, y_train_full, X_test_full, y_test_full)
    
    logger.info("EPJ Validation Completed Successfully.")

if __name__ == "__main__":
    main()
