import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes all required standard multi-class metrics on the test set.
    """
    logger.info("Evaluating predictions...")
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Per-class metrics
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    # Normalised confusion matrix (rows sum to 1)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    metrics = {
        'accuracy': acc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'kappa': kappa,
        'precision_per_class': prec_per_class.tolist(),
        'recall_per_class': rec_per_class.tolist(),
        'confusion_matrix_norm': cm_norm.tolist()
    }
    
    return metrics

def compute_kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    A(K, y) = <K, yy^T>_F / (||K||_F * ||yy^T||_F)
    Uses one-versus-rest encoding for y, averages alignment across all 3 classes.
    y: array of class labels 0, 1, 2
    K: NxN precomputed kernel matrix
    """
    logger.info("Computing kernel-target alignment...")
    classes = np.unique(y)
    alignments = []
    
    for c in classes:
        # OVR encoding for class c: +1 if c, else -1
        y_c = np.where(y == c, 1, -1).astype(float)
        
        # Outer product target matrix: T = yy^T
        T = np.outer(y_c, y_c)
        
        # Frobenius inner product: <A, B>_F = trace(A.T * B) or sum(A * B)
        inner_prod = np.sum(K * T)
        
        norm_K = np.linalg.norm(K, 'fro')
        norm_T = np.linalg.norm(T, 'fro')
        
        if norm_K == 0 or norm_T == 0:
            a_c = 0.0
        else:
            a_c = inner_prod / (norm_K * norm_T)
        
        alignments.append(a_c)
        
    avg_alignment = np.mean(alignments)
    logger.info(f"Average Kernel-Target Alignment: {avg_alignment:.4f}")
    return avg_alignment
