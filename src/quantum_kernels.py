import numpy as np
import logging
from typing import Callable, List
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

logger = logging.getLogger(__name__)

def build_z_feature_map(n: int, x: np.ndarray) -> QuantumCircuit:
    """
    QSVM-Z: Single-qubit rotations only, NO entanglement.
    Apply H then R_Z(2 * x_j) on each qubit, repeat 2 times.
    """
    qc = QuantumCircuit(n)
    for rep in range(2):
        for j in range(n):
            qc.h(j)
            qc.rz(2 * x[j], j)
    return qc

def build_zz_feature_map(n: int, x: np.ndarray) -> QuantumCircuit:
    """
    QSVM-ZZ: 2 repetitions, full entanglement.
    Each rep: H on all qubits, R_Z(2 * x_j), then CNOT-RZ-CNOT on all pairs.
    """
    qc = QuantumCircuit(n)
    for rep in range(2):
        # 1. H on all
        for j in range(n):
            qc.h(j)
        # 2. R_Z on all
        for j in range(n):
            qc.rz(2 * x[j], j)
        # 3. Entanglement
        for j in range(n):
            for k in range(j + 1, n):
                qc.cx(j, k)
                qc.rz(2 * (np.pi - x[j]) * (np.pi - x[k]), k)
                qc.cx(j, k)
    return qc

def compute_statevectors(X: np.ndarray, feature_map_method: Callable, **kwargs) -> List[np.ndarray]:
    """
    Computes the exact statevector |sv_i> = U(x_i)|0> for each input sample x_i in X.
    Uses qiskit_aer.AerSimulator(method='statevector').
    """
    n_samples, n_features = X.shape
    statevectors = []
    
    # Use the simulator in statevector mode
    backend = AerSimulator(method='statevector')
    
    logger.info("Computing statevectors for all samples...")
    for i in tqdm(range(n_samples), desc="Building and running circuits"):
        # Construct the circuit
        qc = feature_map_method(n_features, X[i], **kwargs)
        
        # We need to save the statevector at the end of the circuit
        qc.save_statevector()
        
        # Execute the circuit
        result = backend.run(qc).result()
        sv_i = result.get_statevector(qc)
        statevectors.append(np.asarray(sv_i))
        
    return statevectors

def compute_kernel_matrix(X_train: np.ndarray, X_test: np.ndarray, feature_map_method: Callable, **kwargs):
    """
    Computes K_train (N x N) and K_test (M x N).
    Where K_ij = |<sv_i | sv_j>|^2
    """
    logger.info("Starting quantum kernel computation.")
    
    sv_train = compute_statevectors(X_train, feature_map_method, **kwargs)
    
    # Instead of nested loops O(N^2), use NumPy matrix multiplication
    # Statevectors are complex arrays of shape (2^n,)
    # sv_matrix: shape (N, 2^n)
    sv_train_matrix = np.vstack(sv_train)
    
    logger.info("Computing training kernel matrix via inner products...")
    # Gram matrix = |A * A.H|^2
    K_train_complex = sv_train_matrix @ sv_train_matrix.T.conj()
    K_train = np.abs(K_train_complex) ** 2
    
    # Enforce symmetry and PSD explicitly
    K_train = (K_train + K_train.T) / 2
    
    logger.info("Computing test kernel matrix...")
    if len(X_test) > 0:
        sv_test = compute_statevectors(X_test, feature_map_method, **kwargs)
        sv_test_matrix = np.vstack(sv_test)
        
        K_test_complex = sv_test_matrix @ sv_train_matrix.T.conj()
        K_test = np.abs(K_test_complex) ** 2
    else:
        K_test = None

    return K_train, K_test
