import numpy as np
from qiskit.providers.models import BackendProperties

def noise_amplification_factors(vqe_result, use_depth=False):
    """ 
    Calibrate the noise amplification factors in ZNE 
    based on hardware properties at the time of execution
    """
    properties = BackendProperties.from_dict(vqe_result['VQE_spec']['qubit_data'])
    noise_factors = []

    for qc in vqe_result['VQE_spec']['circuits']['noise_amplified_parallel']['0'].values():
        factor = 0
        if use_depth:
            factor = qc.depth()
        else:
            for op in qc.data:
                gate = op.operation.name
                qbts = [q.index for q in op.qubits]
                if gate in ['cx', 'cz']:
                    factor += properties.gate_error(gate, qbts)
                elif gate not in ['barrier', 'measure']:
                    factor += properties.gate_error(gate, qbts[0])
        noise_factors.append(factor)
        
    return np.array(noise_factors)/noise_factors[0] # division normalizes at lambda=1

def average_without_outliers(data):
    """ 
    Return the mean and standard deviation of input data after removal 
    of outliers (data lying three standard deviations outside the mean)
    """
    std = np.std(data, axis=1).reshape(-1,1)
    avg = np.mean(data, axis=1).reshape(-1,1)
    mask_outliers = abs(data-avg) > std * 3
    return [
        (np.mean(d[~m]), np.std(d[~m])) 
        for d,m in zip(data, mask_outliers)
    ]