Multi-qubit gates are quantum gates that operate on two or more qubits simultaneously. They are essential for creating entangled states and performing operations that involve interactions between qubits. In Qiskit, several multi-qubit gates are commonly used, including the CNOT gate, SWAP gate, and controlled-U gates. Below, we'll explore each of these gates and provide example circuits.

### 1. CNOT Gate (Controlled-NOT Gate)
The CNOT gate is a two-qubit gate that flips the state of the target qubit (second qubit) if the control qubit (first qubit) is in state \(|1\rangle\). This gate is crucial for creating entanglement.

#### Example Circuit with CNOT Gate
```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Create a quantum circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# Apply the H gate to the first qubit to create superposition
qc.h(0)  
# Apply the CNOT gate (0 is the control qubit, 1 is the target qubit)
qc.cx(0, 1)

# Measure both qubits
qc.measure([0, 1], [0, 1])

# Execute the circuit
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
result = job.result()

# Get and plot the results
counts = result.get_counts()
print("Measurement results:", counts)
plot_histogram(counts)
plt.show()
```

### 2. SWAP Gate
The SWAP gate exchanges the states of two qubits. It can be useful for moving qubits around in quantum circuits or for certain algorithms.

#### Example Circuit with SWAP Gate
```python
# Create a quantum circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# Initialize the qubits to a specific state
qc.x(0)  # Set the first qubit to |1⟩

# Apply the SWAP gate to swap the states of the two qubits
qc.swap(0, 1)

# Measure both qubits
qc.measure([0, 1], [0, 1])

# Execute the circuit
job = execute(qc, backend, shots=1024)
result = job.result()

# Get and plot the results
counts = result.get_counts()
print("Measurement results:", counts)
plot_histogram(counts)
plt.show()
```

### 3. Controlled-U Gate
Controlled-U gates are a family of gates where a unitary operation \( U \) is applied to the target qubit based on the state of the control qubit. You can define a custom unitary operation for U.

#### Example Circuit with Controlled-U Gate
```python
import numpy as np
from qiskit.circuit import Gate

# Define a unitary operation (e.g., a rotation about the Z axis)
theta = np.pi / 2
U = np.array([[np.cos(theta/2), -1j * np.sin(theta/2)],
              [-1j * np.sin(theta/2), np.cos(theta/2)]])

# Create a quantum circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# Apply the H gate to create superposition on the first qubit
qc.h(0)

# Define the controlled-U gate
qc.append(Gate('CU', 1, [0, 1], params=[U]))

# Measure both qubits
qc.measure([0, 1], [0, 1])

# Execute the circuit
job = execute(qc, backend, shots=1024)
result = job.result()

# Get and plot the results
counts = result.get_counts()
print("Measurement results:", counts)
plot_histogram(counts)
plt.show()
```

### Summary of Multi-Qubit Gates

| Gate     | Description                                                  |
|----------|--------------------------------------------------------------|
| **CNOT** | Flips the target qubit if the control qubit is \(|1\rangle\). Creates entanglement. |
| **SWAP** | Exchanges the states of two qubits.                          |
| **Controlled-U** | Applies a specified unitary operation to the target qubit based on the control qubit. |

### Conclusion
Multi-qubit gates play a vital role in quantum circuits by enabling operations that involve interactions between multiple qubits. Using Qiskit, you can easily implement these gates and visualize the results of the operations. The examples provided demonstrate how to work with CNOT, SWAP, and controlled-U gates, showcasing their functionality in creating complex quantum algorithms.