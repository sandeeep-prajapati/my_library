Rotation gates in Qiskit (\( R_X \), \( R_Y \), \( R_Z \)) are single-qubit gates that rotate the qubit state around the \( X \), \( Y \), and \( Z \) axes of the Bloch sphere. These gates are parameterized by an angle \( \theta \), which determines the amount of rotation.

### **Mathematical Representations**

1. **\( R_X(\theta) \)**:
   Rotates the qubit around the \( X \)-axis by \( \theta \) radians.
   \[
   R_X(\theta) = \begin{bmatrix}
   \cos(\theta/2) & -i\sin(\theta/2) \\
   -i\sin(\theta/2) & \cos(\theta/2)
   \end{bmatrix}
   \]

2. **\( R_Y(\theta) \)**:
   Rotates the qubit around the \( Y \)-axis by \( \theta \) radians.
   \[
   R_Y(\theta) = \begin{bmatrix}
   \cos(\theta/2) & -\sin(\theta/2) \\
   \sin(\theta/2) & \cos(\theta/2)
   \end{bmatrix}
   \]

3. **\( R_Z(\theta) \)**:
   Rotates the qubit around the \( Z \)-axis by \( \theta \) radians.
   \[
   R_Z(\theta) = \begin{bmatrix}
   e^{-i\theta/2} & 0 \\
   0 & e^{i\theta/2}
   \end{bmatrix}
   \]

---

### **Implementation in Qiskit**

Below is a Qiskit example of how to use \( R_X \), \( R_Y \), and \( R_Z \) gates, visualize the circuit, and observe the effects on the Bloch sphere.

#### **Code Example**

```python
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

# Create a quantum circuit with one qubit
qc = QuantumCircuit(1)

# Apply rotation gates
qc.rx(3.14 / 2, 0)  # RX rotation of π/2
qc.ry(3.14 / 4, 0)  # RY rotation of π/4
qc.rz(3.14 / 6, 0)  # RZ rotation of π/6

# Visualize the circuit
print(qc)

# Simulate the statevector
state = Statevector.from_instruction(qc)

# Plot the Bloch vector
bloch_vector = plot_bloch_multivector(state)
plt.show()
```

---

### **Step-by-Step Explanation**

1. **Circuit Construction**:
   - A quantum circuit is created with one qubit.
   - Rotation gates (\( RX, RY, RZ \)) are applied with specific angles.

2. **State Simulation**:
   - Use the `Statevector` class to calculate the qubit's state after applying the rotation gates.
   - The statevector represents the quantum state as a vector in the complex vector space.

3. **Bloch Sphere Visualization**:
   - The `plot_bloch_multivector()` function generates a Bloch sphere representation of the statevector, showing the effect of the rotation gates.

---

### **Visualizing Effects on the Bloch Sphere**

- **\( R_X(\theta) \)**:
  Rotates the state around the \( X \)-axis, changing the relative weights of the \( |0\rangle \) and \( |1\rangle \) components symmetrically.
  
- **\( R_Y(\theta) \)**:
  Rotates the state around the \( Y \)-axis, affecting the phase difference between \( |0\rangle \) and \( |1\rangle \).

- **\( R_Z(\theta) \)**:
  Rotates the state around the \( Z \)-axis, modifying the relative phase without altering the probabilities of measurement outcomes in the \( |0\rangle \) or \( |1\rangle \) basis.

---

### **Circuit Visualization**

Running the code above would generate:

1. **Circuit Diagram**:
   ```
        ┌──────────┐ ┌──────────┐ ┌──────────┐
    q_0: ┤ RX(π/2) ├─┤ RY(π/4) ├─┤ RZ(π/6) ├
         └──────────┘ └──────────┘ └──────────┘
   ```

2. **Bloch Sphere Output**:
   - A graphical representation of the final state of the qubit after applying the rotation gates.

This illustrates the flexibility of rotation gates in manipulating qubit states in quantum circuits.