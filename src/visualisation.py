import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def plot_confusion_matrix(cm_norm: np.ndarray, title: str, filename: str):
    """
    Plot and save a normalised confusion matrix heatmap.
    Uses blue colormap.
    """
    logger.info(f"Generating confusion matrix plot: {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=['TS (0)', 'MH (1)', 'SH (2)'],
                yticklabels=['TS (0)', 'MH (1)', 'SH (2)'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_depth_analysis(results_dict: dict, filename: str):
    """
    Plot circuit depth vs accuracy plot.
    x-axis: depth L in {1, 2, 3}
    Plot lines for: QSVM-IQP, QSVM-ZZ (horizontal baseline or just point at L=2), SVM-RBF (horizontal baseline)
    In practice, ZZ only has L=2 (2 reps), but we can plot it as a point or horizontal line for comparison.
    """
    logger.info(f"Generating depth analysis plot: {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    depths = [1, 2, 3]
    iqp_accs = [
        results_dict.get('QSVM-IQP-L1', {}).get('accuracy', 0),
        results_dict.get('QSVM-IQP-L2', {}).get('accuracy', 0),
        results_dict.get('QSVM-IQP-L3', {}).get('accuracy', 0)
    ]
    
    rbf_acc = results_dict.get('SVM-RBF', {}).get('accuracy', 0)
    zz_acc = results_dict.get('QSVM-ZZ-L2', {}).get('accuracy', 0)
    z_acc = results_dict.get('QSVM-Z-L2', {}).get('accuracy', 0)
    
    plt.figure(figsize=(8, 6))
    
    # Plot IQP
    plt.plot(depths, iqp_accs, marker='o', label='QSVM-IQP', color='b', linestyle='-', linewidth=2)
    
    # Plot Baselines as horizontal lines to compare across all depths
    plt.axhline(y=rbf_acc, color='r', linestyle='--', label=f'SVM-RBF Baseline ({rbf_acc:.3f})')
    plt.axhline(y=zz_acc, color='g', linestyle='-.', label=f'QSVM-ZZ L=2 ({zz_acc:.3f})')
    plt.axhline(y=z_acc, color='m', linestyle=':', label=f'QSVM-Z L=2 ({z_acc:.3f})')
    
    plt.title('Circuit Depth vs Test Accuracy')
    plt.xlabel('IQP Layer Depth (L)')
    plt.ylabel('Accuracy')
    plt.xticks(depths)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
