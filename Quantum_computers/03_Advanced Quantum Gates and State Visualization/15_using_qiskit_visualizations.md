Qiskit provides several powerful visualization tools for understanding quantum states and the results of quantum computations. Here are some commonly used visualizations with examples:

---

### **1. Plot Quantum State Vectors**
The quantum state vector represents the amplitudes and phases of a quantum state in the computational basis.

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_city

# Create a quantum circuit and prepare a superposition state
qc = QuantumCircuit(1)
qc.h(0)  # Apply a Hadamard gate
state = Statevector.from_instruction(qc)

# Plot the state vector
plot_state_city(state)
```

This will generate a 3D bar chart showing the real and imaginary parts of the state vector.

---

### **2. Plot Bloch Sphere**
The Bloch sphere provides a geometric representation of a single qubit’s state.

```python
from qiskit.visualization import plot_bloch_multivector

# Plot the Bloch vector for the state
plot_bloch_multivector(state)
```

The Bloch sphere visualization will show the qubit's state as a point on or inside the sphere, indicating the relative contributions of \( |0\rangle \) and \( |1\rangle \).

---

### **3. Plot Measurement Histograms**
Histograms are used to visualize the measurement outcomes of a quantum circuit.

```python
from qiskit import Aer, execute

# Add a measurement to the circuit
qc.measure_all()

# Simulate the circuit
simulator = Aer.get_backend('aer_simulator')
result = execute(qc, simulator, shots=1024).result()
counts = result.get_counts()

# Plot the measurement histogram
from qiskit.visualization import plot_histogram
plot_histogram(counts)
```

This will create a histogram showing the distribution of measurement results, with probabilities corresponding to the amplitudes of the quantum state.

---

### **Summary of Tools**

| **Visualization Tool**       | **Function**                                           |
|-------------------------------|-------------------------------------------------------|
| `plot_state_city()`           | Visualizes the quantum state vector in a 3D bar chart. |
| `plot_bloch_multivector()`    | Represents the quantum state on the Bloch sphere.      |
| `plot_histogram()`            | Displays measurement outcomes as a histogram.          |

These visualizations help bridge the gap between abstract quantum mechanics and intuitive understanding. They are especially useful for debugging and presenting results of quantum algorithms.