from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import logging
import numpy as np

logger = logging.getLogger(__name__)

def train_classical_svm(X_train: np.ndarray, y_train: np.ndarray, kernel: str):
    """
    Train a classical SVM using exhaustive grid search for hyperparameter C.
    The Grid space for C is {0.1, 1, 10, 100}. Multi-class uses ovr.
    kernel can be 'linear', 'poly' (degree 3), or 'rbf' (gamma='scale' auto).
    """
    logger.info(f"Training classical SVM with kernel='{kernel}'")
    param_grid = {'C': [0.1, 1, 10, 100]}
    
    if kernel == 'poly':
        svc = SVC(kernel='poly', degree=3, decision_function_shape='ovr', random_state=42)
    elif kernel == 'rbf':
        svc = SVC(kernel='rbf', gamma='scale', decision_function_shape='ovr', random_state=42)
    elif kernel == 'linear':
        svc = SVC(kernel='linear', decision_function_shape='ovr', random_state=42)
    else:
        raise ValueError(f"Unsupported classical kernel: {kernel}")

    clf = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    logger.info("Running 5-fold CV to select C...")
    clf.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {clf.best_params_}, CV Score: {clf.best_score_:.4f}")
    return clf.best_estimator_

def train_quantum_svm(K_train: np.ndarray, y_train: np.ndarray):
    """
    Train a quantum SVM using a precomputed training kernel matrix.
    Selects C via 5-fold cross-validation.
    """
    logger.info("Training Quantum SVM with precomputed kernel")
    param_grid = {'C': [0.1, 1, 10, 100]}
    
    svc = SVC(kernel='precomputed', decision_function_shape='ovr', random_state=42)
    clf = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    logger.info("Running 5-fold CV to select C for QSVM...")
    clf.fit(K_train, y_train)
    
    logger.info(f"Best QSVM parameters: {clf.best_params_}, CV Score: {clf.best_score_:.4f}")
    return clf.best_estimator_
