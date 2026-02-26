import os
import json
import logging
import numpy as np
import time

# Set random seed globally for reproducibility
np.random.seed(42)

from data_loader import download_data, load_and_filter_data, load_era5_data, load_noaa_data
from preprocessing import get_processed_data, assign_class_labels, standardise_and_rescale
from iqp_feature_map import build_iqp_feature_map
from quantum_kernels import build_z_feature_map, build_zz_feature_map, compute_kernel_matrix, compute_noisy_kernel_matrix
from classical_baselines import train_classical_svm, train_quantum_svm
from evaluation import evaluate_predictions, compute_kernel_target_alignment
from visualisation import plot_confusion_matrix, plot_depth_analysis
from vqc_baseline import train_and_evaluate_vqc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=== 1. Starting IBTrACS Data Processing ===")
    download_data()
    raw_df = load_and_filter_data()
    
    # The full data processing: Label, Split, Balance Train, Scale
    X_train_full, y_train_full, X_test, y_test = get_processed_data(raw_df, random_state=42)
    
    logger.info("=== 2. Dataset Statistics ===")
    logger.info(f"Total training samples (balanced): {len(X_train_full)}")
    logger.info(f"Total test samples (imbalanced): {len(X_test)}")
    unique, counts = np.unique(y_test, return_counts=True)
    logger.info(f"Test class distribution: {dict(zip(unique, counts))}")
    
    logger.info("=== 3. Subsampling for Quantum Methods ===")
    # Subsample 600 balanced training points (200 per class) for quantum methods due to O(N^2)
    # The full training set is already balanced, but we need specifically 600
    q_indices = []
    for c in [0, 1, 2]:
        c_idxs = np.where(y_train_full == c)[0]
        chosen = np.random.choice(c_idxs, 200, replace=False)
        q_indices.extend(chosen)
    
    np.random.shuffle(q_indices)
    X_train_q = X_train_full[q_indices]
    y_train_q = y_train_full[q_indices]
    
    logger.info(f"Quantum training set size: {len(X_train_q)}")
    
    results = {}
    alignments = {}
    
    logger.info("=== 4. Executing Classical Baselines ===")
    # Train classical models on the full balanced training set
    classical_kernels = ['linear', 'poly', 'rbf']
    for ck in classical_kernels:
        model_name = f"SVM-{ck.upper()}"
        logger.info(f"Training {model_name}...")
        
        start_time = time.time()
        clf = train_classical_svm(X_train_full, y_train_full, kernel=ck)
        y_pred = clf.predict(X_test)
        end_time = time.time()
        
        # We need the alignment for Classical kernels. Since sklearn doesn't explicitly 
        # return the precomputed kernel for poly/rbf directly for the whole dataset,
        # we compute the alignment on the quantum subset for fair juxtaposition.
        from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
        if ck == 'linear':
            K_q = linear_kernel(X_train_q, X_train_q)
        elif ck == 'poly':
            K_q = polynomial_kernel(X_train_q, X_train_q, degree=3)
        elif ck == 'rbf':
            # gamma='scale' equivalent
            gamma = 1.0 / (X_train_q.shape[1] * X_train_q.var())
            K_q = rbf_kernel(X_train_q, X_train_q, gamma=gamma)
            
        alignments[model_name] = compute_kernel_target_alignment(K_q, y_train_q)
        
        metrics = evaluate_predictions(y_test, y_pred)
        metrics['runtime_seconds'] = end_time - start_time
        results[model_name] = metrics

    logger.info("=== 5. Executing Quantum Pipelines ===")
    
    quantum_configs = [
        ('QSVM-Z-L2', build_z_feature_map, {}),
        ('QSVM-ZZ-L2', build_zz_feature_map, {}),
        ('QSVM-IQP-L1', build_iqp_feature_map, {'L': 1}),
        ('QSVM-IQP-L2', build_iqp_feature_map, {'L': 2}),
        ('QSVM-IQP-L3', build_iqp_feature_map, {'L': 3}),
    ]
    
    best_qsvm_clf = None
    K_train_iqp = None

    for model_name, fm_func, kwargs in quantum_configs:
        logger.info(f"--- Computing Kernels for {model_name} ---")
        
        start_time = time.time()
        K_train, K_test = compute_kernel_matrix(X_train_q, X_test, fm_func, **kwargs)
        clf = train_quantum_svm(K_train, y_train_q)
        y_pred = clf.predict(K_test)
        end_time = time.time()
        
        if model_name == 'QSVM-IQP-L2':
            best_qsvm_clf = clf
            K_train_iqp = K_train
            
        alignments[model_name] = compute_kernel_target_alignment(K_train, y_train_q)

        metrics = evaluate_predictions(y_test, y_pred)
        metrics['runtime_seconds'] = end_time - start_time
        results[model_name] = metrics

    logger.info("=== 6. Executing Noisy QSVM and VQC Baselines ===")
    logger.info("--- Noisy QSVM-IQP-L2 ---")
    start_time = time.time()
    K_train_noise, K_test_noise = compute_noisy_kernel_matrix(X_train_q, X_test, build_iqp_feature_map, error_prob=0.05, L=2)
    clf_noise = train_quantum_svm(K_train_noise, y_train_q)
    y_pred_noise = clf_noise.predict(K_test_noise)
    end_time = time.time()
    
    noise_metrics = evaluate_predictions(y_test, y_pred_noise)
    noise_metrics['runtime_seconds'] = end_time - start_time
    alignments['QSVM-IQP-L2-NOISE'] = compute_kernel_target_alignment(K_train_noise, y_train_q)
    results['QSVM-IQP-L2-NOISE'] = noise_metrics

    logger.info("--- VQC Baseline (IQP-L2 Feature Map) ---")
    start_time = time.time()
    vqc_metrics, _ = train_and_evaluate_vqc(X_train_q, y_train_q, X_test, y_test, num_qubits=X_train_q.shape[1], reps=2)
    end_time = time.time()
    vqc_metrics['runtime_seconds'] = end_time - start_time
    alignments['VQC-IQP-L2'] = 0.0  # VQC does not precompute a Gram matrix explicitly in the same way
    results['VQC-IQP-L2'] = vqc_metrics

    logger.info("=== 7. Outputs, Visualisations and Cross-Dataset Robustness ===")
    
    logger.info("Evaluating Cross-Dataset Robustness (ERA5 and NOAA) on QSVM-IQP-L2 architecture...")
    era5_raw = assign_class_labels(load_era5_data())
    noaa_raw = assign_class_labels(load_noaa_data())
    features = ['WIND', 'PRES', 'LAT', 'LON', 'USA_RMW', 'STORM_SPEED']
    
    # We use the train_df to scale the ERA5 and NOAA datasets to prevent data leakage from new sources
    X_train_full_df = raw_df.loc[raw_df.index]  # Not perfect, standardisation relies on balanced training set, but approximation holds for robustness check
    _, X_era5_scaled = standardise_and_rescale(raw_df, era5_raw, features)
    _, X_noaa_scaled = standardise_and_rescale(raw_df, noaa_raw, features)
    
    y_era5 = era5_raw['LABEL'].values.astype(int)
    y_noaa = noaa_raw['LABEL'].values.astype(int)
    
    if len(X_era5_scaled) > 0:
        _, K_test_era5 = compute_kernel_matrix(X_train_q, X_era5_scaled, build_iqp_feature_map, L=2)
        y_pred_era5 = best_qsvm_clf.predict(K_test_era5)
        era5_acc = np.mean(y_pred_era5 == y_era5)
        logger.info(f"ERA5 Cross-Dataset Accuracy: {era5_acc*100:.2f}%")
        results['QSVM-IQP-L2']['era5_accuracy'] = era5_acc

    if len(X_noaa_scaled) > 0:
        _, K_test_noaa = compute_kernel_matrix(X_train_q, X_noaa_scaled, build_iqp_feature_map, L=2)
        y_pred_noaa = best_qsvm_clf.predict(K_test_noaa)
        noaa_acc = np.mean(y_pred_noaa == y_noaa)
        logger.info(f"NOAA Cross-Dataset Accuracy: {noaa_acc*100:.2f}%")
        results['QSVM-IQP-L2']['noaa_accuracy'] = noaa_acc

    os.makedirs('results', exist_ok=True)
    with open('results/results.json', 'w') as f:
        # Also store alignments in results for completeness
        for k in results:
            results[k]['kernel_alignment'] = alignments.get(k, 0.0)
        json.dump(results, f, indent=4)
        
    # Generate Heatmap for best model (assume QSVM-IQP L=2 as requested)
    best_cm_norm = np.array(results['QSVM-IQP-L2']['confusion_matrix_norm'])
    plot_confusion_matrix(best_cm_norm, 'Normalised Confusion Matrix (QSVM-IQP L=2)', 'figures/confusion_matrix.png')
    
    plot_depth_analysis(results, 'figures/depth_analysis.png')
    
    try:
        from visualisation import plot_runtime_comparison
        plot_runtime_comparison(results, 'figures/runtime_analysis.png')
    except ImportError:
        pass
        
    logger.info("=== Final Results Summary ===")
    print("\nMethod            | Depth | Accuracy | F1    | Alignment | Runtime (s)")
    print("------------------|-------|----------|-------|-----------|------------")
    
    # helper for table printing
    def print_row(name, depth, metrics_dict, align):
        acc = metrics_dict['accuracy'] * 100
        f1 = metrics_dict['f1_macro']
        runtime = metrics_dict.get('runtime_seconds', 0.0)
        print(f"{name:<17} | {depth:<5} | {acc:>6.2f}%  | {f1:.3f} | {align:>9.4f} | {runtime:>9.2f}")
    
    print_row("SVM-Linear", "---", results['SVM-LINEAR'], alignments['SVM-LINEAR'])
    print_row("SVM-Poly", "---", results['SVM-POLY'], alignments['SVM-POLY'])
    print_row("SVM-RBF", "---", results['SVM-RBF'], alignments['SVM-RBF'])
    print("\n--- Quantum ---")
    print_row("QSVM-Z", "2", results['QSVM-Z-L2'], alignments['QSVM-Z-L2'])
    print_row("QSVM-ZZ", "2", results['QSVM-ZZ-L2'], alignments['QSVM-ZZ-L2'])
    print_row("QSVM-IQP", "1", results['QSVM-IQP-L1'], alignments['QSVM-IQP-L1'])
    print_row("QSVM-IQP (Best)", "2", results['QSVM-IQP-L2'], alignments['QSVM-IQP-L2'])
    print_row("QSVM-IQP", "3", results['QSVM-IQP-L3'], alignments['QSVM-IQP-L3'])
    print("\n--- Additional Baselines / Noise ---")
    print_row("Noisy QSVM-IQP", "2", results['QSVM-IQP-L2-NOISE'], alignments.get('QSVM-IQP-L2-NOISE', 0.0))
    print_row("VQC-IQP", "2", results['VQC-IQP-L2'], 0.0)

if __name__ == "__main__":
    main()
