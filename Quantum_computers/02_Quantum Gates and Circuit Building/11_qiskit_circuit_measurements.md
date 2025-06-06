Measurement in quantum computing is a crucial process that translates the quantum state of qubits into classical bits. When a measurement is performed on a quantum state, the state collapses to one of the possible outcomes, and the probability of collapsing to each outcome is determined by the state of the system before measurement. This concept is essential for extracting meaningful information from quantum computations.

### Importance of Measurement
1. **State Collapse**: Measurement causes the quantum state to collapse to one of its eigenstates. Before measurement, a qubit can be in a superposition of states; after measurement, it will be in one definite state.
2. **Probabilistic Outcomes**: The outcome of a measurement is probabilistic, and the probabilities are determined by the amplitudes of the quantum state.
3. **Information Extraction**: Measurement is the primary means by which we extract information from a quantum system, allowing us to observe the effects of quantum operations and algorithms.

### Measurement in Qiskit
In Qiskit, the measurement process involves:
- Specifying which qubits to measure.
- Storing the results in classical bits.
- Using the `measure()` method to perform the measurement.

### Example: Demonstrating Measurement Collapse

Let's create a simple quantum circuit that illustrates measurement collapse. We will:
1. Create a superposition of states using a Hadamard gate.
2. Measure the qubit to demonstrate the collapse to either \(|0\rangle\) or \(|1\rangle\).
3. Execute the circuit multiple times to observe the probabilistic outcomes.

Here’s the complete code to demonstrate this:

```python
# Import necessary libraries
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Step 1: Create a quantum circuit with 1 qubit and 1 classical bit
qc = QuantumCircuit(1, 1)

# Step 2: Apply a Hadamard gate to the qubit to create superposition
qc.h(0)  # This creates the superposition state (|0> + |1>)/sqrt(2)

# Step 3: Measure the qubit
qc.measure(0, 0)

# Visualize the circuit
print("Quantum Circuit Demonstrating Measurement Collapse:")
print(qc.draw())

# Step 4: Execute the circuit
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)  # Execute with 1024 shots
result = job.result()

# Step 5: Get and visualize the measurement results
counts = result.get_counts()
print("Measurement results:", counts)

# Plot the histogram of results
plot_histogram(counts)
plt.title("Measurement Results")
plt.show()
```

### Explanation of the Code

1. **Create the Quantum Circuit**:
   - We initialize a quantum circuit with 1 qubit and 1 classical bit for measurement.

2. **Apply the Hadamard Gate**:
   - The Hadamard gate (H) is applied to the qubit, placing it into a superposition state of \(|0\rangle\) and \(|1\rangle\) with equal probabilities.

3. **Measurement**:
   - We then measure the qubit, which will collapse the superposition into either \(|0\rangle\) or \(|1\rangle\). The result is stored in the classical bit.

4. **Execution**:
   - The circuit is executed using the QASM simulator with a specified number of shots (repetitions of the experiment). Here, we use 1024 shots to get a statistical distribution of outcomes.

5. **Results Visualization**:
   - The measurement results are collected and displayed as a histogram, showing the frequency of each outcome (|0⟩ and |1⟩) after multiple measurements.

### Expected Output
When you run the code, you should see the circuit diagram and a histogram showing the measurement results. Given the superposition created by the Hadamard gate, you will typically see approximately equal counts for \(|0\rangle\) and \(|1\rangle\) in the histogram, illustrating the probabilistic nature of quantum measurement.

### Conclusion
This example illustrates how measurement works in Qiskit and why it is important in quantum computing. The measurement process is essential for extracting classical information from quantum states, allowing us to observe the outcomes of quantum algorithms and operations. By demonstrating measurement collapse, we can see how quantum states change upon measurement and how probabilistic outcomes arise from superposition.