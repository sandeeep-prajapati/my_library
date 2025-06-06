Quantum gates are the building blocks of quantum circuits, analogous to classical logic gates. In Qiskit, several common quantum gates manipulate the states of qubits. Here’s an overview of the most commonly used gates: **X**, **Y**, **Z**, **H**, and **CNOT**. 

### 1. X Gate (Pauli-X Gate)
- **Function**: The X gate, also known as the NOT gate or Pauli-X gate, flips the state of a qubit. If the qubit is in the state \(|0\rangle\), it will become \(|1\rangle\) and vice versa.
- **Matrix Representation**:
  \[
  X = \begin{pmatrix}
  0 & 1 \\
  1 & 0
  \end{pmatrix}
  \]
- **Application**:
```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(1)  # Create a circuit with 1 qubit
qc.x(0)                 # Apply the X gate to qubit 0
print(qc.draw())
```

### 2. Y Gate (Pauli-Y Gate)
- **Function**: The Y gate is another Pauli gate that flips the qubit and introduces a phase of \(\pi\) (180 degrees). It transforms \(|0\rangle\) to \(i|1\rangle\) and \(|1\rangle\) to \(-i|0\rangle\).
- **Matrix Representation**:
  \[
  Y = \begin{pmatrix}
  0 & -i \\
  i & 0
  \end{pmatrix}
  \]
- **Application**:
```python
qc = QuantumCircuit(1)  # Create a circuit with 1 qubit
qc.y(0)                 # Apply the Y gate to qubit 0
print(qc.draw())
```

### 3. Z Gate (Pauli-Z Gate)
- **Function**: The Z gate introduces a phase flip to the qubit. It leaves \(|0\rangle\) unchanged but flips the phase of \(|1\rangle\).
- **Matrix Representation**:
  \[
  Z = \begin{pmatrix}
  1 & 0 \\
  0 & -1
  \end{pmatrix}
  \]
- **Application**:
```python
qc = QuantumCircuit(1)  # Create a circuit with 1 qubit
qc.z(0)                 # Apply the Z gate to qubit 0
print(qc.draw())
```

### 4. H Gate (Hadamard Gate)
- **Function**: The Hadamard gate creates superposition. It transforms \(|0\rangle\) into \(|+\rangle\) and \(|1\rangle\) into \(|-\rangle\).
- **Matrix Representation**:
  \[
  H = \frac{1}{\sqrt{2}} \begin{pmatrix}
  1 & 1 \\
  1 & -1
  \end{pmatrix}
  \]
- **Application**:
```python
qc = QuantumCircuit(1)  # Create a circuit with 1 qubit
qc.h(0)                 # Apply the H gate to qubit 0
print(qc.draw())
```

### 5. CNOT Gate (Controlled-NOT Gate)
- **Function**: The CNOT gate is a two-qubit gate that flips the state of the target qubit if the control qubit is \(|1\rangle\). It creates entanglement between qubits.
- **Matrix Representation**:
  \[
  \text{CNOT} = \begin{pmatrix}
  1 & 0 & 0 & 0 \\
  0 & 1 & 0 & 0 \\
  0 & 0 & 0 & 1 \\
  0 & 0 & 1 & 0
  \end{pmatrix}
  \]
- **Application**:
```python
qc = QuantumCircuit(2)  # Create a circuit with 2 qubits
qc.cx(0, 1)             # Apply the CNOT gate (0 is control, 1 is target)
print(qc.draw())
```

### Summary
Here's a recap of the common quantum gates:

| Gate | Symbol | Description |
|------|--------|-------------|
| X    | \(X\)  | Flips the state of a qubit (NOT gate). |
| Y    | \(Y\)  | Flips the state and introduces a phase of \(\pi\). |
| Z    | \(Z\)  | Flips the phase of \(|1\rangle\). |
| H    | \(H\)  | Creates superposition from a basis state. |
| CNOT | \(CX\) | Flips the target qubit if the control qubit is \(|1\rangle\). |

These gates can be combined to form complex quantum circuits that enable quantum algorithms to leverage superposition and entanglement, fundamentally differentiating them from classical computation.