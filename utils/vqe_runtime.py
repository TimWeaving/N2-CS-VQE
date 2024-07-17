import mthree as m3
import numpy as np
from scipy.optimize import minimize
from time import time
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, transpile
from qiskit.providers.ibmq.runtime import UserMessenger, RuntimeEncoder
from qiskit.opflow import PauliSumOp
from qiskit.circuit.library import XGate
from qiskit.transpiler import PassManager, InstructionDurations
#from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
from functools import reduce
from typing import List, Dict, Tuple

def CNOT_root_product(
        n_qubits: int, 
        c: int, 
        t: int, 
        noise_parameter: int = 1
    ) -> QuantumCircuit:
    """ Decomposition of the CNOT gate into repeated CPhase gates, which are each transpiled into 2 CNOTs
    Therefore, a noise parameter k results in 2k CNOT gates that implement a single CNOT.
    """
    qc_block = QuantumCircuit(n_qubits)
    qc_block.h(t)
    for j in range(noise_parameter):
        qc_block.cp(np.pi/noise_parameter,c,t)
    qc_block.h(t)
    return qc_block

def _replace_CNOTs(qc: QuantumCircuit, noise_parameter:int=1) -> QuantumCircuit:
    """ Strip single CNOT gates from the imput circuit and replace with the CPhase decomposition above
    """
    qc_copy = qc.copy()
    # index CNOT gates within the circuit data
    where_CX = np.where(np.array([i.operation.name=='cx' for i in qc_copy.data]))[0]
    if where_CX.size == 0:
        # if no CNOTs remaining, return the circuit
        return qc_copy
    else:
        # otherwise, take the first CNOT index in the list
        first_CX = where_CX[0]
        # extract the control and target qubit indices
        control, target = qc_copy.data[first_CX].qubits
        control, target = control.index, target.index
        # get the CNOT decomposition into CPhase gates
        noisy_CNOT = CNOT_root_product(qc.num_qubits, control, target, noise_parameter=noise_parameter)
        # remove the CNOT from the circuit...
        qc_copy.data.pop(first_CX)
        # ... and insert in its place the corresponding noise-amplified block
        for insert_operation in noisy_CNOT[::-1]:
            qc_copy.data.insert(first_CX, insert_operation)
        # then recurse until there are no CNOT gates remaining
        return _replace_CNOTs(qc_copy, noise_parameter=noise_parameter)

def hex_to_bin(hexdict: Dict[str, int], n_qubits:int):
    """ NOte: Qiskit returns its quantum measurement data in hex
    This function converts the dictionary keys from hex -> binary
    """
    return {
        np.binary_repr(int(hexstr, 16), n_qubits):count 
        for hexstr, count in hexdict.items()
    }

def expectation_value(op: PauliSumOp, counts:Dict[str, int]) -> float:
    """ Given an observable and measurement outcome from a quantum 
    experiment, compute the corresponding expectation value.

    Note: it is assumed the circuit from which the counts are derived was mapped 
    into the correct Pauli basis, else this expectation value will be incorrect!
    """
    # mask where there are non-identity qubit posistions 
    # (equivalent to rotating onto Pauli Z's via the appropriate basis change)
    non_I_mask = op.primitive.paulis.x[:, ::-1] | op.primitive.paulis.z[:, ::-1]
    measurements, coeffs = zip(*counts.items())
    # convert the measurements into a binary array
    measurement_mask = np.array([[int(s) for s in binstr] for binstr in measurements], dtype=bool)
    
    expval = 0
    # loop over the terms of the observable
    for h_coeff, symp in zip(op.coeffs, non_I_mask):
        # sign flips are defined by a binary AND between the measurement data and 
        # where there is a corresponding Pauli Z operator in the observable term
        signs = (-1) ** np.sum(symp & measurement_mask, axis=1)
        expval += np.sum(coeffs * signs) * h_coeff
    return expval
    
def energy_estimate(ops: List[PauliSumOp], results: List[Dict[str, int]]) -> float:
    """ Given a list of observables (usually qubit-wise commuting) 
    and measurement sets, calculate the separate expectation values 
    and then sum to get the overall energy estimate
    """
    assert len(ops)==len(results), f'Incompatible number of observables {len(ops)} versus measurement sets {len(results)}'
    return sum(map(lambda x:expectation_value(x[0],x[1]), zip(ops, results))).real

class VQE_Driver:
    """ Runtime program for performing VQE routines.
    """
    n_shots = 2**10
    use_ZNE = True
    n_dd_reps = 1
    checkpoint = 1
    start_time = time()
    
    def __init__(self,
        backend,
        user_messenger,
        circuit: QuantumCircuit= None,
        QWC_cliques: Dict[int, PauliSumOp] = None,
        ZNE_factors: List[int] = [1,2,3,4],
        parallelize: bool = True,
        circuit_optimization_level: int = 3
        ) -> None:
        """
        backend: The backend on which quantum circuits will be executed, either a noise model or real quantum device
        user_messenger: allows interim messaging during the VQE routine
        circuit: the ansatz circuit over which we shall optimize in the VQE routine
        QWC_cliques: a dictionary of qubit-wise commuting observables
        ZNE_factors: a list of noise-amplification factors for the zero-noise extrapolation procedure
        parallelize: flag that determines whether the circuit should be parallelized accross the backend
        circuit_optimization_level: intensity of transpilation circuit optimizer
        """
        self.user_messenger = user_messenger
        
        self.runtime_checkpoint(message='Initializing...')
        self.backend       = backend
        self.faulty_qubits = backend.properties().faulty_qubits()
        self.durations     = InstructionDurations.from_backend(backend)
        self.constraints   = backend.configuration().timing_constraints
        self.circuit       = circuit
        if isinstance(QWC_cliques, PauliSumOp):
            QWC_cliques = {0:QWC_cliques}
        self.QWC_cliques = {int(k):v for k,v in QWC_cliques.items()} # since in RuntimeEncoder integers are stringified i->'i'
        self.ZNE_factors = ZNE_factors
        self.parallelize = parallelize
        self.circuit_optimization_level = circuit_optimization_level
        self.param_indices = list(range(self.circuit.num_parameters))
        self.param_names = [p.name for p in self.circuit.parameters]

        self.runtime_checkpoint(message='Building circuits')
        self.circuits = {}
        self.zne_circuits = {}
        # for each qubit-wise commuting (QWC) cliuqe, determine the relevant basis transformation
        for index, clique in self.QWC_cliques.items():
            # Hadamard gates wherever we have a Pauli X or Y
            H_pos = reduce(lambda a,b:a|b, clique.primitive.paulis.x[:,::-1])
            # S^\dag gate wherever there is a Pauli Y
            S_pos = reduce(lambda a,b:a|b, clique.primitive.paulis.z[:,::-1]) & H_pos
            # reverse the qubit indices due to Qiskit's ordering!
            H_pos = self.circuit.num_qubits-1-np.where(H_pos)[0]
            S_pos = self.circuit.num_qubits-1-np.where(S_pos)[0]
            # circuits with change-of-basis (COB) blocks included
            circ = self.get_parallel_cob_circuit(X_indices=H_pos, Y_indices=S_pos)
            self.circuits[index] = circ
            self.zne_circuits[index] = {
                n:self.replace_CNOTs(qc=circ, noise_parameter=n) 
                for n in self.ZNE_factors
            }
        # effective number of qubits is the number of active qubits 
        # in the ansatz times the number of parallel copies
        self.n_qubits = self.circuit.num_qubits * self.n_tiles
        
        self.runtime_checkpoint(message='Initializing M3 class')
        self.mem = m3.M3Mitigation(backend)
        self.runtime_checkpoint(message='Extracting measurement calibration data')
        self.mem.cals_from_system()
        self.runtime_checkpoint(message='Getting the final measurement map')
        # the measurement map for use in measurement-error mitigation
        self.mapping = m3.utils.final_measurement_mapping(self.circuits[0])

        self.runtime_checkpoint(message='Successfully Initialized the VQE Driver :)')

    def runtime_checkpoint(self, message = None):
        """ Publish checkpoints for debugging purposes
        """
        end_time = time()
        duration = end_time - self.start_time
        message = f'Checkpoint {self.checkpoint} completed after {duration} seconds: {message}'
        self.user_messenger.publish(message)
        self.checkpoint += 1
        self.start_time = time()
    
    def parallelize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """ Tile as many copies of the ansatz circuit as possible 
        accross the available qubits in the backend
        """
        backend_n_q = self.backend.configuration().num_qubits
        n_tiles = backend_n_q // circuit.num_qubits
        n_fault = len(self.faulty_qubits) # number of faulty qubits that would cause the runtime job to error
        stacked_circuit = QuantumCircuit()
        for i in range(n_tiles-n_fault):
            # define the quantum/classical registers for each circuit tile
            qr = QuantumRegister(circuit.num_qubits, f'qreg{i}')
            cr = ClassicalRegister(circuit.num_qubits, f'creg{i}')
            # add these to the full, parallelized circuit
            stacked_circuit.add_register(qr)
            stacked_circuit.add_register(cr)
            qc = QuantumCircuit(qr, cr)
            qc.compose(circuit, inplace=True)
            # and insert the corresponding circuit tile
            stacked_circuit.compose(qc, qubits=qr, inplace=True)
        
        stacked_circuit.barrier()
        
        return stacked_circuit, n_tiles
    
    def get_parallel_cob_circuit(self, 
            X_indices: List[int] = [], 
            Y_indices: List[int] = []
        ) -> QuantumCircuit:
        """ Given the indices of X and Y Paulis in some QWC clique,
        apply Hadamard and S^\dag gates to rotate onto the Pauli-Z 
        basis for subsequent measurement
        """
        circuit_cob = self.circuit.copy()
        # S^\dag H where we have Pauli Y's
        for i in Y_indices:
            circuit_cob.sdg(i)
            circuit_cob.h(i)
        # H where we have Pauli X's
        for i in X_indices:
            circuit_cob.h(i)

        # also, parallelize the circuit at this point if necessary
        if self.parallelize:
            parallel_circuit, self.n_tiles = self.parallelize_circuit(circuit_cob)
            for i in range(self.n_tiles):
                # meaure each circuit tile to the corresponding classical register
                parallel_circuit.measure(parallel_circuit.qregs[i], parallel_circuit.cregs[i])
        else:
            self.n_tiles=1
            parallel_circuit = circuit_cob
            parallel_circuit.measure_all()
        # finally, transpile the circuit given a backend and optimization level
        return transpile(
            parallel_circuit, 
            backend=self.backend, 
            optimization_level=self.circuit_optimization_level
        )
    
    def replace_CNOTs(self, 
            qc: QuantumCircuit, 
            noise_parameter:int=1
        ) -> QuantumCircuit:
        """ Replace CNOT gates with decomposition over CPhases for noise amplification
        """
        return transpile(
            _replace_CNOTs(qc, noise_parameter=noise_parameter),
            optimization_level=0,
            backend=self.backend
        )

    def split_measurements(self, 
            data_in: Dict[str, int], 
            data_type: str = 'bin'
        ):
        """ If the circuit has been parallelized, we need to split the 
        resulting measurement output accross the classical registers, yielding
        circuit.num_qubits*n_tiles -> circuit.num_qubits
        """
        n_qubits_per_partition = self.circuit.num_qubits
        data_out = {}
        if data_type == 'bin':
            bin_data = data_in 
        elif data_type == 'hex':
            bin_data = hex_to_bin(data_in, self.n_qubits)
        else:
            raise ValueError('Unrecognised data type, must be bin or hex.')
        # loop over the parallel measurement outcomes
        for binstr, coeff in bin_data.items():
            # split across each circuit tile
            for i in range(self.n_tiles):
                binstr_part = binstr[i*n_qubits_per_partition:(i+1)*n_qubits_per_partition]
                data_out[binstr_part] = data_out.get(binstr_part, 0) + coeff/self.n_tiles
        return data_out
    
    def insert_DD_gates(self, qc_bound):
        """ Insert sequence of X-X gates for dynamical decoupling
        Note circuit must have already been bound with parameters
        """
        dd_sequence = [XGate()]*2*self.n_dd_reps
        pm = PassManager([ALAPScheduleAnalysis(self.durations),
                        PadDynamicalDecoupling(self.durations, dd_sequence, 
                        pulse_alignment=self.constraints['pulse_alignment'])])
        qc_dd = pm.run(qc_bound)
        # ensure scheduling consistent with the device backend, else can result in error
        #qc_dd = transpile(qc_dd, backend=self.backend, scheduling_method='alap')
        return qc_dd 

    def bind_parameters_to_circuits(self, qc, params):
        """ Bind paramters to the input circuit and insert dynamical decoupling sequence
        """
        qc_bound = qc.bind_parameters(params)
        qc_dd    = self.insert_DD_gates(qc_bound=qc_bound)
        return qc_dd
    
    def get_counts(self, 
            circuits: List[QuantumCircuit], 
            params: np.array
        ) -> List[Dict[str, int]]:
        """ Given a list of circuits and an array of parameters to bind, 
        submit the circuits to the backend and apply measurement-error 
        mitigation to the resulting measurement data.
        """
        assert len(circuits) == len(self.QWC_cliques), 'Must supply same number of circuits as there are qubit-wise commuting cliques'
        # bind parameters to the circuits
        bound_circuits = list(map(lambda qc:self.bind_parameters_to_circuits(qc, params), circuits.values()))
        # execute the bound circuits on the backend
        results = self.backend.run(bound_circuits, shots=self.n_shots).result().results
        # apply measurement-error mitigation with mthree
        results = [
            self.mem.apply_correction(hex_to_bin(r.data.counts, n_qubits=self.n_qubits), self.mapping
                ).nearest_probability_distribution() for r in results
        ]
        # split the parallelized measurements across circuit tiles
        measurements = {i:self.split_measurements(r) for i,r in enumerate(results)}
        return measurements

    def get_energy_estimate(self, x: np.array, override_ZNE=False) -> Tuple[float, dict]:
        """ Execute the quantum circuits at parametrization x and calculate
        energy estimate. Also returns all the raw data for further postprocessing 
        if desired (e.g. error mitigation such as symmetry verification)
        """
        aux_data = {'measurements':{}}
        if self.use_ZNE and not override_ZNE:
            data_to_extrapolate = []
            aux_data['noise_factors'] = self.ZNE_factors
            for n in self.ZNE_factors:
                self.n_dd_reps = 2*n
                noise_amp_circs = {i:self.zne_circuits[i][n] for i in self.QWC_cliques.keys()}
                results = self.get_counts(noise_amp_circs, x)
                data_to_extrapolate.append(energy_estimate(self.QWC_cliques.values(), results.values()))
                aux_data['measurements'][n] = results
            aux_data['data_for_extrapolation'] = data_to_extrapolate
            energy_out = np.polyfit(self.ZNE_factors, data_to_extrapolate, deg=1)[-1]
        else:
            self.n_dd_reps = 2
            results = self.get_counts(self.circuits, x)
            aux_data['measurements'] = results
            energy_out = energy_estimate(self.QWC_cliques.values(), results.values())

        return energy_out, aux_data
    
    def get_partial_derivative_estimate(self, x: np.array, pd_index:int) -> Tuple[float, dict]:
        """ Compute partial derivate for parameter indexed by pd_index via the parameter-shift rule 
        """
        x_upper = x.copy(); x_upper[pd_index]+=np.pi/4
        x_lower = x.copy(); x_lower[pd_index]-=np.pi/4
        
        nrg_upper, aux_upper = self.get_energy_estimate(x_upper, override_ZNE=True)
        nrg_lower, aux_lower = self.get_energy_estimate(x_lower, override_ZNE=True)
        aux_data_out = {'upper': aux_upper, 'lower':aux_lower}
        
        return nrg_upper - nrg_lower, aux_data_out

    def get_gradient_estimate(self, x: np.array) -> Tuple[float, dict]:
        """ Return the vector of partial derivatives computed for each parameter index
        """
        derivative_by_parameter = [self.get_partial_derivative_estimate(x, i) for i in self.param_indices]
        pd, data = zip(*derivative_by_parameter) 
        return np.array(pd), dict(zip(self.param_names, data))
    
def main(backend, user_messenger, **kwargs):
    """ The main runtime program entry-point.

    All the heavy-lifting is handled by the VQE_Driver class

    Returns:
        - vqe_result (Dict[str,Union[int, float, bool, array]]):
            The optimizer output
        - interim_values (Dict[str,List[Union[float, array]]]):
            The interim energy, parameter and gradient values
    """    
    circuit     = kwargs.get("circuit", None)
    QWC_cliques = kwargs.get("QWC_cliques", None)
    parallelize = kwargs.get("parallelize", True)
    circuit_optimization_level = kwargs.get("circuit_optimization_level", 1)
    
    vqe = VQE_Driver(
        circuit=circuit, 
        QWC_cliques=QWC_cliques, 
        backend=backend,
        user_messenger=UserMessenger(),
        parallelize=parallelize,
        circuit_optimization_level=circuit_optimization_level
    )
    vqe.n_shots = kwargs.get("n_shots", 2**7)
    vqe.use_ZNE = kwargs.get("use_ZNE", True)
    opt_setting = kwargs.get("opt_setting", {"maxiter":20})
    init_params = kwargs.get("init_params", np.random.random(circuit.num_parameters))
    optimizer   = kwargs.get("optimizer", 'BFGS')
    
    vqe_spec = {
        'hardware_spec':backend.configuration().to_dict(),
        'qubit_data':backend.properties().to_dict(),
        'circuits':{
            'standard':circuit, 
            'standard_parallel':vqe.circuits, 
            'noise_amplified_parallel':vqe.zne_circuits,
            'n_parallel':vqe.n_tiles
        },
        'mthree_mapping':vqe.mapping,
        'QWC_cliques':vqe.QWC_cliques,
        'parallelize':parallelize,
        'faulty_qubits':vqe.faulty_qubits,
        'circuit_optimization_level':circuit_optimization_level
    }
    encoder = RuntimeEncoder()
    vqe_spec_encoded = encoder.encode(vqe_spec)
    user_messenger.publish(vqe_spec_encoded)

    data_out = {
        'parameter_history':{},
        'energy_history':{'values':{}, 'data':{}},
        'gradient_history':{'values':{}, 'data':{}},
    }

    global counter
    counter = -1
    def get_counter(increment=True):
        global counter
        if increment:
            counter += 1
        return counter
    
    def fun(x):    
        counter = get_counter(increment=True)
        nrg, data = vqe.get_energy_estimate(x)
        #user_messenger.publish(f'Optimization step {counter: <2} energy: {nrg}')
        data_out['parameter_history'][counter] = x
        data_out['energy_history']['values'][counter] = nrg
        data_out['energy_history']['data'][counter] = data
        data_out_encoded = encoder.encode(data_out)
        user_messenger.publish(data_out_encoded)
        return nrg

    def jac(x):
        counter = get_counter(increment=False)
        grad, data = vqe.get_gradient_estimate(x)
        #user_messenger.publish(f'Optimization step {counter: <2} gradient norm: {np.linalg.norm(grad)}')
        data_out['gradient_history']['values'][counter] = grad
        data_out['gradient_history']['data'][counter] = data
        return grad

    user_messenger.publish('VQE simulation commencing...')
    opt_out = minimize(
        fun=fun, jac=jac, x0=init_params, 
        method=optimizer, options=opt_setting
    )
    data_out['optimizer_output'] = opt_out
    
    return {
        'VQE_data':data_out,
        'VQE_spec':vqe_spec
    }