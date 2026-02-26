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

def plot_runtime_comparison(results_dict: dict, filename: str):
    """
    Plot computational runtime comparison across models.
    """
    logger.info(f"Generating runtime comparison plot: {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    models = []
    runtimes = []
    colors = []
    
    for k, v in results_dict.items():
        if 'runtime_seconds' in v:
            models.append(k)
            runtimes.append(v['runtime_seconds'])
            if 'SVM' in k and 'QSVM' not in k:
                colors.append('skyblue')
            elif 'NOISE' in k:
                colors.append('salmon')
            elif 'VQC' in k:
                colors.append('mediumpurple')
            else:
                colors.append('lightgreen')
                
    # Sort by runtime descending
    sorted_indices = np.argsort(runtimes)[::-1]
    models = [models[i] for i in sorted_indices]
    runtimes = [runtimes[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(models, runtimes, color=colors, edgecolor='black')
    
    plt.xlabel('Runtime (Seconds)')
    plt.ylabel('Model')
    plt.title('Computational Runtime Comparison')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add text labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}s', 
                 va='center', ha='left', fontsize=9)
                 
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
