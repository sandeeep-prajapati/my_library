A **quantum circuit** is a model for quantum computation that consists of a sequence of quantum operations (or gates) applied to quantum bits (qubits). Quantum circuits leverage the principles of quantum mechanics, such as superposition and entanglement, to perform calculations that are often exponentially faster than classical circuits for certain problems.

### Components of a Quantum Circuit
1. **Qubits**: 
   - The fundamental unit of quantum information, analogous to classical bits.
   - Qubits can exist in a state of `|0⟩`, `|1⟩`, or any superposition of these states, represented mathematically as:
     \[
     |\psi\rangle = \alpha |0\rangle + \beta |1\rangle
     \]
     where \(|\alpha|^2 + |\beta|^2 = 1\).
   - Qubits are initialized to a base state (usually `|0⟩`) before any operations are performed.

2. **Gates**:
   - Quantum gates are operations that change the state of qubits. They are the building blocks of quantum circuits, similar to logic gates in classical circuits.
   - Gates can be single-qubit gates, like the **Hadamard gate (H)**, which creates superposition, or multi-qubit gates, like the **CNOT gate**, which entangles qubits.
   - Some commonly used gates include:
     - **Hadamard Gate (H)**: Creates superposition.
     - **Pauli-X Gate (X)**: Flips the state of a qubit (`|0⟩` to `|1⟩` and vice versa).
     - **CNOT Gate**: Flips the target qubit if the control qubit is `|1⟩`, creating entanglement between qubits.
     - **Phase Gates (S, T)**: Introduce a phase shift to the qubit's state.
   - Quantum gates are represented as matrices, and their action on qubits can be described using matrix multiplication.

3. **Measurement**:
   - Measurement is the process of observing the state of qubits, collapsing their superposition into one of the basis states (`|0⟩` or `|1⟩`).
   - When measuring a qubit, the probability of measuring `|0⟩` or `|1⟩` is given by the square of the amplitudes:
     \[
     P(0) = |\alpha|^2, \quad P(1) = |\beta|^2
     \]
   - After measurement, the qubit's state is determined, and any subsequent operations will use this new state.

### How Quantum Circuits Work in Qiskit
In Qiskit, quantum circuits can be easily created and manipulated using its high-level abstractions. Here’s how you can work with quantum circuits in Qiskit:

#### Step 1: Import Qiskit Modules
You begin by importing the necessary modules:
```python
from qiskit import QuantumCircuit, Aer, execute
```

#### Step 2: Create a Quantum Circuit
You can create a quantum circuit by specifying the number of qubits and classical bits (for measurement):
```python
qc = QuantumCircuit(2, 2)  # 2 qubits and 2 classical bits
```

#### Step 3: Add Gates to the Circuit
Add quantum gates to manipulate the qubits. For example:
```python
qc.h(0)            # Apply Hadamard gate to qubit 0
qc.cx(0, 1)       # Apply CNOT gate with qubit 0 as control and qubit 1 as target
```

#### Step 4: Measurement
Measure the qubits to store the results in classical bits:
```python
qc.measure([0, 1], [0, 1])  # Measure qubits 0 and 1 into classical bits 0 and 1
```

#### Step 5: Execute the Circuit
You can run the circuit on a simulator or actual quantum hardware. For simulation:
```python
backend = Aer.get_backend('qasm_simulator')  # Choose a simulator backend
job = execute(qc, backend, shots=1024)        # Execute the circuit 1024 times
result = job.result()                          # Get the results
counts = result.get_counts(qc)                # Get the measurement counts
print("Measurement results:", counts)
```

### Summary
A quantum circuit in Qiskit consists of qubits, quantum gates that manipulate these qubits, and measurement operations that extract classical information from the quantum states. Here’s a recap of the process:
1. **Qubits**: Basic units of quantum information, able to exist in superposition.
2. **Gates**: Operations applied to qubits to change their states, enabling complex computations.
3. **Measurement**: Observing the state of qubits to produce classical output.

Quantum circuits enable the execution of quantum algorithms that can exploit the principles of quantum mechanics to achieve significant computational advantages over classical algorithms for specific tasks. Qiskit provides a straightforward interface to construct and simulate these circuits, facilitating exploration and experimentation in quantum computing.