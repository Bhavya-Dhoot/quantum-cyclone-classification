import os
import json
import logging
import numpy as np

# Set random seed globally for reproducibility
np.random.seed(42)

from data_loader import download_data, load_and_filter_data
from preprocessing import get_processed_data
from iqp_feature_map import build_iqp_feature_map
from quantum_kernels import build_z_feature_map, build_zz_feature_map, compute_kernel_matrix
from classical_baselines import train_classical_svm, train_quantum_svm
from evaluation import evaluate_predictions, compute_kernel_target_alignment
from visualisation import plot_confusion_matrix, plot_depth_analysis

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
        clf = train_classical_svm(X_train_full, y_train_full, kernel=ck)
        
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
        
        y_pred = clf.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        results[model_name] = metrics

    logger.info("=== 5. Executing Quantum Pipelines ===")
    
    quantum_configs = [
        ('QSVM-Z-L2', build_z_feature_map, {}),
        ('QSVM-ZZ-L2', build_zz_feature_map, {}),
        ('QSVM-IQP-L1', build_iqp_feature_map, {'L': 1}),
        ('QSVM-IQP-L2', build_iqp_feature_map, {'L': 2}),
        ('QSVM-IQP-L3', build_iqp_feature_map, {'L': 3}),
    ]
    
    for model_name, fm_func, kwargs in quantum_configs:
        logger.info(f"--- Computing Kernels for {model_name} ---")
        K_train, K_test = compute_kernel_matrix(X_train_q, X_test, fm_func, **kwargs)
        
        alignments[model_name] = compute_kernel_target_alignment(K_train, y_train_q)
        
        clf = train_quantum_svm(K_train, y_train_q)
        y_pred = clf.predict(K_test)
        metrics = evaluate_predictions(y_test, y_pred)
        results[model_name] = metrics

    logger.info("=== 6. Outputs and Visualisations ===")
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
    
    logger.info("=== Final Results Summary ===")
    print("\nMethod       | Depth | Accuracy | Precision | Recall | F1    | Kappa | Alignment")
    print("-------------|-------|----------|-----------|--------|-------|-------|----------")
    
    # helper for table printing
    def print_row(name, depth, metrics_dict, align):
        acc = metrics_dict['accuracy'] * 100
        prec = metrics_dict['precision_macro']
        rec = metrics_dict['recall_macro']
        f1 = metrics_dict['f1_macro']
        kappa = metrics_dict['kappa']
        print(f"{name:<12} | {depth:<5} | {acc:>6.2f}%  | {prec:.3f}     | {rec:.3f}  | {f1:.3f} | {kappa:.3f} | {align:.4f}")
    
    print_row("SVM-Linear", "---", results['SVM-LINEAR'], alignments['SVM-LINEAR'])
    print_row("SVM-Poly", "---", results['SVM-POLY'], alignments['SVM-POLY'])
    print_row("SVM-RBF", "---", results['SVM-RBF'], alignments['SVM-RBF'])
    print_row("QSVM-Z", "2", results['QSVM-Z-L2'], alignments['QSVM-Z-L2'])
    print_row("QSVM-ZZ", "2", results['QSVM-ZZ-L2'], alignments['QSVM-ZZ-L2'])
    print_row("QSVM-IQP", "1", results['QSVM-IQP-L1'], alignments['QSVM-IQP-L1'])
    print_row("QSVM-IQP", "2", results['QSVM-IQP-L2'], alignments['QSVM-IQP-L2'])
    print_row("QSVM-IQP", "3", results['QSVM-IQP-L3'], alignments['QSVM-IQP-L3'])

if __name__ == "__main__":
    main()
