import logging
import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from iqp_feature_map import build_iqp_feature_map
from evaluation import evaluate_predictions

logger = logging.getLogger(__name__)

def train_and_evaluate_vqc(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, num_qubits: int, reps: int = 2):
    """
    Train a Variational Quantum Classifier (VQC) with IQP Feature Map and RealAmplitudes ansatz.
    """
    logger.info("Building VQC model...")
    x_params = ParameterVector('x', length=num_qubits)
    # 1. Feature Map (IQP L=2 for consistency)
    feature_map = build_iqp_feature_map(num_qubits, x_params, L=reps)
    
    # 2. Ansatz
    ansatz = RealAmplitudes(num_qubits, reps=reps)
    
    # 3. Optimizer
    optimizer = COBYLA(maxiter=100)
    
    # Since VQC handles multi-class naturally but needs one-hot labels,
    # we convert scalar labels to one-hot for VQC, or let VQC handle it if possible.
    # Qiskit VQC supports categorical inputs directly if we use CrossEntropyLoss and one-hot encoding.
    n_classes = len(np.unique(y_train))
    y_train_one_hot = np.eye(n_classes)[y_train]
    y_test_one_hot = np.eye(n_classes)[y_test]
    
    vqc = VQC(
        num_qubits=num_qubits,
        feature_map=feature_map,
        ansatz=ansatz,
        loss="cross_entropy",
        optimizer=optimizer,
    )
    
    logger.info("Training VQC via COBYLA optimizer (this might take a while)...")
    vqc.fit(X_train, y_train_one_hot)
    
    logger.info("Training complete. Evaluating VQC...")
    y_pred_one_hot = vqc.predict(X_test)
    y_pred = np.argmax(y_pred_one_hot, axis=1)
    
    # Evaluate predictions
    metrics = evaluate_predictions(y_test, y_pred)
    return metrics, vqc
