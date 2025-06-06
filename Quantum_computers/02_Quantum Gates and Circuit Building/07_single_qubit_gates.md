To use single-qubit gates in Qiskit, you can create a quantum circuit that applies the X, Y, Z, and H gates sequentially on a single qubit. After applying these gates, you can measure the output state of the qubit. Below is a step-by-step guide to creating such a circuit, including code examples.

### Step 1: Import Necessary Libraries
You need to import the required libraries from Qiskit:
```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
```

### Step 2: Create a Quantum Circuit
You can create a quantum circuit with one qubit and one classical bit for measurement:
```python
# Create a quantum circuit with 1 qubit and 1 classical bit
qc = QuantumCircuit(1, 1)
```

### Step 3: Apply Single-Qubit Gates
You can apply the X, Y, Z, and H gates to the qubit. Here’s the order of the gates applied:
1. **X Gate**: Flip the qubit.
2. **Y Gate**: Introduce a phase flip.
3. **Z Gate**: Apply another phase flip.
4. **H Gate**: Create superposition.
```python
# Apply the X gate
qc.x(0)

# Apply the Y gate
qc.y(0)

# Apply the Z gate
qc.z(0)

# Apply the H gate
qc.h(0)
```

### Step 4: Measure the Qubit
After applying the gates, you need to measure the qubit and store the result in the classical bit:
```python
# Measure the qubit into the classical bit
qc.measure(0, 0)
```

### Step 5: Execute the Circuit
You can use a simulator backend to execute the circuit and obtain the measurement results:
```python
# Use the statevector simulator to execute the circuit
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
result = job.result()
```

### Step 6: Get and Visualize Results
Finally, you can get the counts of the measurement results and visualize them using a histogram:
```python
# Get the counts of the measurement results
counts = result.get_counts()

# Print the counts
print("Measurement results:", counts)

# Plot the histogram of results
plot_histogram(counts)
```

### Complete Example Code
Here’s the complete code, combining all the steps:
```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Step 1: Create a quantum circuit with 1 qubit and 1 classical bit
qc = QuantumCircuit(1, 1)

# Step 2: Apply single-qubit gates
qc.x(0)  # Apply the X gate
qc.y(0)  # Apply the Y gate
qc.z(0)  # Apply the Z gate
qc.h(0)  # Apply the H gate

# Step 3: Measure the qubit
qc.measure(0, 0)

# Step 4: Execute the circuit
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
result = job.result()

# Step 5: Get and visualize the results
counts = result.get_counts()
print("Measurement results:", counts)

# Plot the histogram
plot_histogram(counts)
plt.show()
```

### Explanation of the Circuit
1. **X Gate**: Flips the initial state of the qubit (|0⟩ to |1⟩).
2. **Y Gate**: Flips the state and introduces a phase change, altering the probabilities.
3. **Z Gate**: Applies a phase flip, further modifying the state.
4. **H Gate**: Transforms the qubit into a superposition state, creating equal probabilities for |0⟩ and |1⟩.

### Expected Output
When you run the complete example code, you should see the measurement results printed in the console, and a histogram will visualize the distribution of results. The results may vary due to the inherent probabilistic nature of quantum mechanics, especially after the Hadamard gate is applied.

By executing this code, you will understand how to manipulate a single qubit using common quantum gates in Qiskit.