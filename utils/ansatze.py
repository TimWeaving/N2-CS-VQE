from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np

# 0         ref = |11000>
# 1,2,3     ref = |11000>
# 4,5,6,7,8 ref = |10000>
# 9         ref = |00000>

def ansatz_0():
    pvec = ParameterVector('P', 5)
    qc = QuantumCircuit(5)
    qc.sx(4); qc.sx(2); qc.x(3); qc.sx(1); qc.sx(0)
    qc.rz(np.pi/2, 0); qc.rz(np.pi/2, 1); qc.rz(np.pi/2, 2); qc.rz(-np.pi/2, 4)
    qc.cx(0,1); qc.rz(pvec[2], 1); qc.cx(0,1)
    qc.cx(2,4); qc.rz(pvec[3], 4); qc.cx(2,4)
    #qc.barrier()
    qc.rz(pvec[0], 0); qc.sx(0)#; qc.rz(np.pi, 0)
    qc.rz(np.pi/2, 1); qc.sx(1); qc.rz(np.pi/2, 1)
    qc.rz(np.pi/2, 2); qc.sx(2); qc.rz(np.pi/2, 2)
    qc.rz(np.pi/2, 4); qc.sx(4)#; qc.rz(np.pi, 4)
    return qc

def ansatz_123():
    """
    reference:   |11000>
    excitations:
    """
    pvec = ParameterVector('P', 2)
    qc = QuantumCircuit(5)
    qc.sx(4); qc.sx(2); qc.sx(3); qc.sx(1); qc.sx(0)
    qc.rz(np.pi/2, 0); qc.rz(np.pi/2, 1); qc.rz(np.pi/2, 2); qc.rz(-np.pi/2, 3); qc.rz(-np.pi/2, 4)
    qc.cx(0,1); qc.cx(1,4); qc.rz(pvec[0], 4);  qc.cx(1,4); qc.cx(0,1)
    qc.cx(2,3); qc.rz(pvec[1], 3); qc.cx(2,3)
    #qc.barrier()
    qc.rz(np.pi/2, 0); qc.sx(0)#; qc.rz(- np.pi, 0)
    qc.rz(np.pi/2, 1); qc.sx(1); qc.rz(np.pi/2, 1)
    qc.rz(np.pi/2, 2); qc.sx(2)#; qc.rz(- np.pi, 2)
    qc.rz(np.pi/2, 3); qc.sx(3); qc.rz(np.pi/2, 3)
    qc.rz(np.pi/2, 4); qc.sx(4); qc.rz(np.pi/2, 4)
    return qc

# bond length = 
def ansatz_45678():
    """
    reference:   |10000>
    excitations: IIXYI + XYIII + IIIIY + YYYII + IIIYI
    """
    pvec = ParameterVector('P', 5)
    qc = QuantumCircuit(5)
    qc.sx(4); qc.sx(2); qc.sx(3); qc.sx(1); qc.sx(0)
    qc.rz(np.pi/2, 1); qc.rz(np.pi/2, 2); qc.rz(np.pi/2, 3); qc.rz(-np.pi/2, 4)
    qc.cx(1,2); qc.rz(pvec[1], 2); qc.cx(1,2)
    qc.cx(3,4); qc.rz(pvec[2], 4); qc.cx(3,4)
    qc.rz(-np.pi,2); qc.sx(2); qc.rz(-np.pi,2)
    qc.rz(-np.pi,4); qc.sx(4); qc.rz(-np.pi,4)
    qc.cx(2,3); qc.cx(3,4); qc.rz(pvec[4], 4); qc.cx(3,4); qc.cx(2,3)
    #qc.barrier()
    qc.rz(pvec[0], 0); qc.sx(0)#; qc.rz(np.pi, 0)
    qc.rz(pvec[3], 1); qc.sx(1)#; qc.rz(np.pi, 1)
    qc.rz(np.pi/2, 2); qc.sx(2)#; qc.rz(np.pi, 2)
    qc.rz(np.pi/2, 3); qc.sx(3)#; qc.rz(np.pi, 3)
    qc.rz(np.pi/2, 4); qc.sx(4)#; qc.rz(np.pi, 4)
    return qc

def ansatz_9():
    """
    reference:   |00000> 
    excitations:
    """
    pvec = ParameterVector('P', 6)
    qc = QuantumCircuit(5)
    qc.sx(0); qc.sx(1); qc.sx(2); qc.sx(3); qc.sx(4)
    qc.rz(pvec[0], 0); qc.rz(pvec[2], 2); qc.rz(pvec[4], 4)
    qc.rz(np.pi/2, 1); qc.rz(np.pi/2, 3) 
    qc.sx(0); qc.sx(2); qc.sx(4)
    qc.cx(0,1); qc.cx(2,3)
    qc.rz(pvec[1], 1); qc.rz(pvec[3], 3) 
    qc.cx(0,1); qc.cx(2,3)
    qc.cx(1, 4)
    qc.rz(pvec[5], 4)
    qc.cx(1, 4)
    #qc.barrier()
    qc.rz(np.pi/2, 0); qc.sx(0); qc.rz(np.pi/2, 0)
    qc.rz(np.pi/2, 1); qc.sx(1)#; qc.rz(- np.pi, 1)
    qc.rz(np.pi/2, 2); qc.sx(2); qc.rz(np.pi/2, 2)
    qc.rz(np.pi/2, 3); qc.sx(3)#; qc.rz(- np.pi, 3)
    qc.rz(np.pi/2, 4); qc.sx(4); qc.rz(np.pi/2, 4)
    return qc