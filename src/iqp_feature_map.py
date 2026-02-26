from qiskit import QuantumCircuit
import numpy as np

def build_iqp_feature_map(n: int, x: np.ndarray, L: int = 1) -> QuantumCircuit:
    """
    Manually constructs the Instantaneous Quantum Polynomial (IQP) feature map circuit.
    
    n: number of qubits (features)
    x: 1D numpy array of features for a single sample, shape (n,)
    L: circuit depth
    """
    if len(x) != n:
        raise ValueError(f"Input dimension mismatch. Expected {n}, got {len(x)}")
        
    qc = QuantumCircuit(n)
    
    alpha = 1.0
    beta = 1.0
    
    for layer in range(L):
        # 1. Hadamard layer on all qubits
        for j in range(n):
            qc.h(j)
            
        # 2. Single-qubit Z-rotations: R_Z(alpha * x_j)
        for j in range(n):
            qc.rz(alpha * x[j], j)
            
        # 3. Two-qubit controlled-phase gates CP(beta * x_j * x_k)
        for j in range(n):
            for k in range(j + 1, n):
                qc.cp(beta * x[j] * x[k], j, k)
                
    # 4. Final closing Hadamard layer
    for j in range(n):
        qc.h(j)
        
    return qc
