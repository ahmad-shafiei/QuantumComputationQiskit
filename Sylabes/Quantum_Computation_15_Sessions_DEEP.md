# Quantum Computation – Full 15-Session Course (Deep & Detailed, Markdown)

This document provides a **complete, teaching-ready syllabus** for a 15-session course on **Quantum Computation**.  
Each session includes:

- Learning Objectives  
- Core Concepts / Outline  
- Theory & Intuition  
- Examples  
- In-Class Exercises  
- Homework  
- Python/Qiskit Coding Suggestions  
- Suggested Figures (Bloch sphere, circuits, etc.)  

You can use this as:
- A printable handout for students  
- A base to create slides  
- A structure for designing assignments and exams  

---

## Session 1: Introduction & Mathematical Foundations


### Learning Objectives
By the end of this session, students should be able to:
- Explain why quantum computation is interesting and potentially powerful.
- Distinguish classical bits from quantum bits (qubits).
- Describe the notion of a quantum state as a complex vector.
- Understand basic linear algebra notions: vectors, norms, inner products.
- Install Python and Qiskit, and run a minimal quantum circuit.

### Core Concepts / Outline
- Motivation: limitations of classical computing, Moore’s law, new applications.
- Bits vs. Qubits.
- State vectors and the 2D complex vector space for a single qubit.
- Normalization and probabilities.
- Tensor products (conceptual preview).
- Overview of the circuit model of quantum computation.
- Tooling: Python, Qiskit, and simulators.

### Theory & Intuition
Classical computation is based on **bits** that can take values 0 or 1. A classical register of \(n\) bits can be in exactly one of \(2^n\) possible states at any time.  

In contrast, a **qubit** is described by a **normalized complex 2D vector**:
\[
\lvert \psi \rangle = \alpha \lvert 0 \rangle + \beta \lvert 1 \rangle,\quad \text{with } |\alpha|^2 + |\beta|^2 = 1.
\]
Here, \(\lvert 0 \rangle = \begin{pmatrix}1 \\ 0\end{pmatrix}\) and \(\lvert 1 \rangle = \begin{pmatrix}0 \\ 1\end{pmatrix}\).  

The squared magnitudes \(|\alpha|^2\) and \(|\beta|^2\) represent measurement probabilities in the computational basis. The fact that a qubit can be in a **superposition** of basis states enables new kinds of computation.

### Example
Consider the state:
\[
\lvert \psi \rangle = \frac{1}{\sqrt{2}} \lvert 0 \rangle + \frac{1}{\sqrt{2}} \lvert 1 \rangle.
\]
The probabilities of measuring 0 or 1 are both \(1/2\). This is analogous to a fair coin, but represented as a quantum state.

### In-Class Exercises
1. Represent the states \(|0\rangle\), \(|1\rangle\), and \((|0\rangle + |1\rangle)/\sqrt{2}\) as column vectors.  
2. Verify normalization for several example states.  
3. Discuss qualitatively how a 2-qubit system would be represented (preview of tensor products).

### Homework
1. Normalize the state
   \[
   \lvert \phi \rangle = 2 \lvert 0 \rangle + 3 \lvert 1 \rangle,
   \]
   and compute measurement probabilities.  
2. Write a short paragraph explaining, in your own words, the difference between a classical bit and a qubit.  
3. Install Python and Qiskit on your computer and run the sample code below. Take a screenshot of the output.

### Python/Qiskit Coding
```python
# Minimal Qiskit example: single-qubit Hadamard and measurement
from qiskit import QuantumCircuit, Aer, execute

# Create a circuit with 1 qubit and 1 classical bit
qc = QuantumCircuit(1, 1)

# Apply a Hadamard gate to create superposition
qc.h(0)

# Measure the qubit
qc.measure(0, 0)

# Simulate the circuit
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)
result = job.result()
counts = result.get_counts()

print("Circuit:")
print(qc)
print("Measurement counts:", counts)
```

### Suggested Figures
- **Conceptual diagram**: Classical bit vs. qubit (bit as a two-state system, qubit as a vector in a 2D complex space).  
- **Vector diagram**: Show \(|0\rangle\) and \(|1\rangle\) as orthogonal basis vectors.  
- **Simple circuit**: One-qubit circuit with an H gate followed by a measurement:
  ```text
  q0: ──H──■──
           │
  c0: ─────┼──
  ```


---

## Session 2: Linear Algebra, Dirac Notation & Operators


### Learning Objectives
By the end of this session, students should be able to:
- Recall the main concepts of linear algebra used in quantum mechanics.
- Work with inner products, norms, and orthonormal bases.
- Understand Dirac (bra-ket) notation.
- Recognize unitary and Hermitian matrices and their roles in quantum computing.

### Core Concepts / Outline
- Vector spaces and basis vectors.
- Inner product and norm.
- Orthonormal bases.
- Dirac notation: kets, bras, and bra-ket products.
- Unitary and Hermitian operators.
- Global phase vs. relative phase.

### Theory & Intuition
Quantum states live in **complex vector spaces** (Hilbert spaces). An **inner product** allows us to define angles and lengths in this space:
\[
\langle \phi | \psi \rangle = \sum_i \phi_i^* \psi_i.
\]

In **Dirac notation**, kets \(|\psi\rangle\) represent column vectors, while bras \(\langle \psi|\) represent row vectors of complex conjugates. The inner product is written as \(\langle \phi|\psi\rangle\).

**Unitary operators** \(U\) (with \(U^\dagger U = I\)) represent valid quantum evolutions; they preserve norms and thus total probability. **Hermitian operators** are observables (measurable quantities) in quantum mechanics.

A key point is that **global phase** has no physical effect:
\[
|\psi\rangle \text{ and } e^{i\theta}|\psi\rangle
\]
represent the same physical state.

### Example
Show that the Hadamard gate
\[
H = \frac{1}{\sqrt{2}}\begin{pmatrix}1 & 1 \\ 1 & -1\end{pmatrix}
\]
is unitary by verifying \(H^\dagger H = I\).

### In-Class Exercises
1. Compute \(\langle 0|1\rangle\), \(\langle 1|1\rangle\), and \(\langle +|+\rangle\), where \(|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}\).  
2. Given a 2×2 matrix, determine whether it is unitary.  
3. Show that multiplying a state by a global phase does not affect measurement probabilities.

### Homework
1. For an arbitrary normalized qubit state \(|\psi\rangle = \alpha |0\rangle + \beta |1\rangle\), compute  
   \(\langle \psi|\psi \rangle\) and confirm that it equals 1.  
2. Read a short section on Dirac notation from your main textbook and summarize key points in half a page.

### Python/Qiskit Coding
```python
import numpy as np

# Define basis states
zero = np.array([[1], [0]], dtype=complex)
one  = np.array([[0], [1]], dtype=complex)

# Define Hadamard
H = (1/np.sqrt(2)) * np.array([[1, 1],
                               [1, -1]], dtype=complex)

# Verify unitarity: H† H
identity_check = H.conj().T @ H
print("H^† H =")
print(identity_check)
```

### Suggested Figures
- A diagram showing bras and kets as row and column vectors.  
- Illustration of orthonormal basis vectors (like x and y axes in 2D).  
- A box summarizing properties of unitary and Hermitian matrices.


---

## Session 3: Qubits and the Bloch Sphere


### Learning Objectives
By the end of this session, students should be able to:
- Describe the Bloch sphere representation of a single qubit.
- Map a general qubit state to Bloch sphere coordinates (θ, φ).
- Understand the geometric interpretation of quantum states.
- Relate rotations on the Bloch sphere to quantum gates.

### Core Concepts / Outline
- General form of a 1-qubit state:
  \(|\psi\rangle = \alpha|0\rangle + \beta|1\rangle\).
- Parametrization:
  \(|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle\).
- Bloch sphere: mapping to (x, y, z) on the unit sphere.
- Poles and equator: |0⟩, |1⟩, |+⟩, |−⟩.
- Geometric action of single-qubit gates as rotations.

### Theory & Intuition
Any normalized qubit state can be written as:
\[
|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle,
\]
for some real angles \(\theta \in [0,\pi]\) and \(\phi \in [0, 2\pi)\), ignoring global phase.  
We define Bloch coordinates:
\[
x = \sin\theta \cos\phi,\quad
y = \sin\theta \sin\phi,\quad
z = \cos\theta.
\]
These coordinates describe a point on the **unit sphere**, called the **Bloch sphere**.  

Common states:
- \(|0\rangle\): north pole (0, 0, 1)  
- \(|1\rangle\): south pole (0, 0, −1)  
- \(|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}\): point on the +x axis  
- \(|-\rangle = (|0\rangle - |1\rangle)/\sqrt{2}\): point on the −x axis

Single-qubit gates effect rotations of this Bloch vector.

### Example
Take \(|\psi\rangle = (|0\rangle + i|1\rangle)/\sqrt{2}\).  
Express \(|\psi\rangle\) in the parametrized form and find its Bloch sphere coordinates.

### In-Class Exercises
1. Write down Bloch sphere coordinates for \(|0\rangle\), \(|1\rangle\), \(|+\rangle\), \(|-\rangle\), and \((|0\rangle + i|1\rangle)/\sqrt{2}\).  
2. Discuss intuitively what applying H, X, Z does to points on the Bloch sphere.

### Homework
1. Prove that every pure qubit state corresponds to exactly one point on the Bloch sphere (up to global phase).  
2. Derive the expression for x, y, z in terms of α and β.

### Python/Qiskit Coding
```python
from qiskit.visualization import plot_bloch_vector
import matplotlib.pyplot as plt
import numpy as np

# Example: |+> state
theta = np.pi/2
phi = 0
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

fig = plot_bloch_vector([x, y, z])
plt.show()
```

### Suggested Figures
- **Bloch sphere diagram**:
  - North pole labeled |0⟩, south pole |1⟩.
  - Equator showing |+⟩ and |−⟩.
  - A generic state |ψ⟩ with angles θ and φ indicated.
- **Gate action sketches**:
  - X gate: 180° rotation around x-axis.
  - Z gate: 180° rotation around z-axis.


---

## Session 4: Single-Qubit Gates and Rotations


### Learning Objectives
By the end of this session, students should be able to:
- Identify and use standard single-qubit gates (X, Y, Z, H, S, T).
- Represent these gates as matrices and understand their effects.
- Relate certain gates to rotations on the Bloch sphere.
- Build basic 1-qubit circuits and reason about their outputs.

### Core Concepts / Outline
- Pauli gates: X, Y, Z.
- Hadamard gate H and its role in creating superposition.
- Phase gates S and T.
- Rotation gates: Rx(θ), Ry(θ), Rz(θ).
- Gate identities and simple compositions (e.g., HZH = X).

### Theory & Intuition
Single-qubit gates correspond to **unitary 2×2 matrices**. For example:
\[
X = \begin{pmatrix}0 & 1 \\ 1 & 0\end{pmatrix},\quad
Z = \begin{pmatrix}1 & 0 \\ 0 & -1\end{pmatrix}.
\]
The Hadamard gate creates superposition:
\[
H |0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}},\quad
H |1\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}.
\]

Rotation gates such as \(R_x(\theta)\) correspond to rotations of the Bloch sphere around a given axis by angle θ.

### Example
Verify the identity:
\[
HZH = X.
\]
Compute HZH and show it equals the X matrix.

### In-Class Exercises
1. Apply H followed by X to \(|0\rangle\) and compute the resulting state.  
2. Compute the effect of S and T on the states |+⟩ and |−⟩.  
3. Discuss how a rotation gate Rx(π/2) moves a state on the Bloch sphere.

### Homework
1. Prove that the Pauli matrices X, Y, Z are Hermitian and unitary.  
2. Show that H^2 = I.  
3. Construct a sequence of gates that maps |0⟩ to |1⟩ via an intermediate superposition.

### Python/Qiskit Coding
```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(1, 1)

# Prepare |0>, apply H then Z then H
qc.h(0)
qc.z(0)
qc.h(0)
qc.measure(0, 0)

print(qc)
```

### Suggested Figures
- Matrix table listing X, Y, Z, H, S, T.
- Bloch sphere diagrams showing how X, Y, Z act as 180° rotations around axes.
- A simple circuit diagram illustrating sequences like H → Z → H.


---

## Session 5: Multi-Qubit States, Entanglement and Bell States


### Learning Objectives
By the end of this session, students should be able to:
- Work with multi-qubit systems using tensor products.
- Describe entangled states, especially Bell states.
- Understand the difference between separable and entangled states.
- Construct simple entangling circuits using CNOT and H gates.

### Core Concepts / Outline
- Tensor products of Hilbert spaces.
- Computational basis for multiple qubits (|00⟩, |01⟩, |10⟩, |11⟩, ...).
- Separable vs. entangled states.
- Bell states and their preparation circuits.
- Two-qubit gates: CNOT, CZ, SWAP.

### Theory & Intuition
For two qubits, the state space is 4-dimensional. The basis states are:
\[
|00\rangle, |01\rangle, |10\rangle, |11\rangle.
\]
A state is **separable** if it can be written as \(|\psi\rangle_A \otimes |\phi\rangle_B\). States that cannot be written this way are **entangled**.  

The **Bell states** are maximally entangled two-qubit states, such as:
\[
|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}.
\]

### Example
Prepare the Bell state \(|\Phi^+\rangle\) by:
1. Starting with |00⟩.
2. Applying H on qubit 0.
3. Applying CNOT with control qubit 0 and target qubit 1.

### In-Class Exercises
1. Show that \(|\Phi^+\rangle\) is not separable.  
2. Write the states for other Bell states: \(|\Phi^-\rangle\), \(|\Psi^+\rangle\), \(|\Psi^-\rangle\).  
3. Use the CNOT gate to create different Bell states by changing the input.

### Homework
1. Prove that no single-qubit gates acting independently on each qubit can create entanglement starting from a product state.  
2. Compute measurement correlations for Bell states in the computational basis.

### Python/Qiskit Coding
```python
from qiskit import QuantumCircuit, Aer, execute

qc = QuantumCircuit(2, 2)
qc.h(0)       # Create superposition on qubit 0
qc.cx(0, 1)   # Entangle qubit 0 and 1
qc.measure([0,1], [0,1])

backend = Aer.get_backend('qasm_simulator')
result = execute(qc, backend, shots=1024).result()
print("Counts:", result.get_counts())
```

### Suggested Figures
- Circuit diagram for generating |Φ⁺⟩:
  ```text
  q0: ──H──■──
           │
  q1: ─────X──
  ```
- Diagram showing entangled pairs as “linked” qubits.
- Table comparing separable and entangled states.


---

## Session 6: Measurement, Born Rule and Noise


### Learning Objectives
By the end of this session, students should be able to:
- Explain the Born rule and measurement outcomes.
- Describe projective measurements in the computational basis.
- Understand the concept of decoherence and basic noise models.
- Use Qiskit to simulate noisy quantum circuits.

### Core Concepts / Outline
- Born rule for measurement probabilities.
- Projective measurements and post-measurement states.
- Measurement in different bases.
- Noise channels: bit-flip, phase-flip, depolarizing (conceptual).
- Decoherence and T1/T2 times (high level).

### Theory & Intuition
Measurement in quantum mechanics is probabilistic. For state:
\[
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle,
\]
measuring in the computational basis yields 0 with probability \(|\alpha|^2\) and 1 with probability \(|\beta|^2\). After measurement, the state collapses to the corresponding basis state.  

Real quantum systems are open and interact with their environment, causing **decoherence**. Noise can be modeled with channels like bit-flip or depolarizing channels.

### Example
Simulate repeated measurements of the state (|0⟩ + |1⟩)/√2 and show that frequencies approach 50–50 as shots increase.

### In-Class Exercises
1. Given a state, compute measurement probabilities and post-measurement states.  
2. Discuss qualitatively how decoherence affects superpositions and entanglement.

### Homework
1. Read about T1 and T2 times for qubits and summarize in half a page.  
2. Consider a simple noise model (bit-flip with probability p) and reason its impact on a single-qubit state.

### Python/Qiskit Coding
```python
from qiskit import QuantumCircuit, Aer, execute

qc = QuantumCircuit(1,1)
qc.h(0)
qc.measure(0,0)

backend = Aer.get_backend('qasm_simulator')
result = execute(qc, backend, shots=2048).result()
print(result.get_counts())
```

(Optionally extend with Qiskit Aer noise models if desired.)

### Suggested Figures
- Probability tree diagrams showing measurement outcomes.
- Bloch sphere with arrows showing “shrinking” of the vector due to decoherence.
- Circuit with measurement symbol at the end of a wire.


---

## Session 7: Quantum Circuit Model and Universality


### Learning Objectives
By the end of this session, students should be able to:
- Describe the quantum circuit model of computation.
- Understand how complex unitaries are built from basic gate sets.
- Know what it means for a gate set to be universal.
- Express simple algorithms as quantum circuits.

### Core Concepts / Outline
- Quantum circuit as a sequence of gates.
- Wires as qubits, boxes as gates, measurement symbols.
- Universality and standard gate sets {H, T, CNOT} etc.
- Circuit depth and width.
- Decomposition of multi-qubit unitaries into 1- and 2-qubit gates (conceptual).

### Theory & Intuition
The **circuit model** is the central paradigm of quantum computation. It is analogous to classical logic circuits but with unitary gates operating on qubits. A finite set of gates is **universal** if any unitary can be approximated by some circuit built from those gates.

### Example
Show that {H, T, CNOT} is a universal gate set (conceptually: H and T generate arbitrary single-qubit unitaries, plus CNOT for entanglement).

### In-Class Exercises
1. Draw the circuit for a simple algorithm (e.g., preparing a Bell state).  
2. Discuss how you would implement a 3-qubit unitary using only 1- and 2-qubit gates.

### Homework
1. Read a section about universality in your main text and write a summary.  
2. Sketch a circuit that takes |00⟩ to (|00⟩ + |11⟩)/√2 and then measures both qubits.

### Python/Qiskit Coding
```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0,1],[0,1])
print(qc.draw())
```

### Suggested Figures
- General block diagram of a quantum circuit: input state → sequence of gates → measurement.
- Example circuits for Bell state and small algorithms.


---

## Session 8: Deutsch and Deutsch–Jozsa Algorithms


### Learning Objectives
By the end of this session, students should be able to:
- State the Deutsch problem and the Deutsch–Jozsa problem.
- Describe the oracle model of computation.
- Implement the Deutsch–Jozsa algorithm as a circuit.
- Understand the advantage over classical deterministic algorithms.

### Core Concepts / Outline
- Oracle functions f: {0,1}^n → {0,1}.
- Constant vs. balanced functions.
- Deutsch algorithm (n = 1 case).
- Deutsch–Jozsa algorithm (general n).
- Query complexity and quantum advantage.

### Theory & Intuition
The Deutsch–Jozsa algorithm determines whether a function f is constant or balanced with **one oracle query** in the quantum model, whereas any deterministic classical algorithm may require up to \(2^{n-1} + 1\) queries.

The algorithm:
1. Prepare |0...0⟩|1⟩.  
2. Apply H to all qubits.  
3. Apply oracle U_f.  
4. Apply H again to the first n qubits.  
5. Measure the first n qubits.

### Example
Implement the Deutsch algorithm for a single-bit function and verify that measurement outcomes distinguish constant and balanced functions.

### In-Class Exercises
1. Write down the circuit for Deutsch–Jozsa with n = 2.  
2. Discuss the number of oracle queries required in quantum vs. classical settings.

### Homework
1. Show mathematically that constant functions yield measurement outcome |0...0⟩ in the Deutsch–Jozsa algorithm.  
2. Implement a 2-qubit version in Qiskit with a chosen oracle.

### Python/Qiskit Coding
```python
from qiskit import QuantumCircuit, Aer, execute

def deutsch_oracle(is_constant):
    qc = QuantumCircuit(2)
    if is_constant:
        qc.x(1)  # Example constant oracle
    else:
        qc.cx(0,1)  # Example balanced oracle
    return qc.to_gate(label="Uf")

qc = QuantumCircuit(2,1)
qc.x(1)
qc.h([0,1])

oracle = deutsch_oracle(is_constant=False)
qc.append(oracle, [0,1])

qc.h(0)
qc.measure(0,0)

backend = Aer.get_backend('qasm_simulator')
result = execute(qc, backend, shots=1024).result()
print(result.get_counts())
```

### Suggested Figures
- Circuit diagram for Deutsch algorithm (n=1).  
- Circuit diagram for Deutsch–Jozsa (n general) with oracle block U_f.  
- Table showing constant vs. balanced examples.


---

## Session 9: Grover’s Search Algorithm


### Learning Objectives
By the end of this session, students should be able to:
- Explain the unstructured search problem.
- Understand the idea of amplitude amplification.
- Describe the Grover iterate (oracle + diffusion).
- Implement Grover’s algorithm for small N using Qiskit.

### Core Concepts / Outline
- Search problem over N = 2^n items.
- Oracle marking the “good” state(s).
- Grover operator G = (2|s⟩⟨s| − I) · (I − 2|ω⟩⟨ω|).
- Optimal number of iterations O(√N).
- Behavior of amplitudes over iterations.

### Theory & Intuition
Grover’s algorithm provides a **quadratic speedup** over classical search. It repeatedly “rotates” the state vector in the 2D subspace spanned by the marked state and the uniform superposition, increasing the amplitude of the target state.

### Example
Apply Grover’s algorithm to a 2-qubit system (N = 4) with exactly one marked element, and track the amplitudes by hand or numerically.

### In-Class Exercises
1. For N = 4, compute how many Grover iterations are optimal.  
2. Analyze how amplitudes change during one full Grover iteration.

### Homework
1. Implement Grover’s algorithm in Qiskit for N = 4 and verify success probabilities.  
2. Write a short explanation of why the algorithm fails if you over-iterate.

### Python/Qiskit Coding
```python
from qiskit import QuantumCircuit, Aer, execute

# Example: Grover for 2-qubit search space with state |11> marked
qc = QuantumCircuit(2,2)

# Step 1: prepare uniform superposition
qc.h([0,1])

# Step 2: oracle marking |11>
qc.cz(0,1)  # phase flip for |11>

# Step 3: diffusion operator
qc.h([0,1])
qc.x([0,1])
qc.h(1)
qc.cx(0,1)
qc.h(1)
qc.x([0,1])
qc.h([0,1])

qc.measure([0,1],[0,1])

backend = Aer.get_backend('qasm_simulator')
result = execute(qc, backend, shots=1024).result()
print(result.get_counts())
```

### Suggested Figures
- Geometric picture: rotation in a 2D subspace between |s⟩ and |ω⟩.  
- Circuit diagram for Grover’s algorithm with oracle block and diffusion block.  
- Plot of success probability vs. number of iterations (conceptual).


---

## Session 10: Shor’s Algorithm and Period Finding


### Learning Objectives
By the end of this session, students should be able to:
- Understand the high-level structure of Shor’s algorithm.
- Explain the role of period finding and modular exponentiation.
- Recognize the use of the Quantum Fourier Transform (QFT).
- Describe why Shor’s algorithm threatens classical public-key cryptography.

### Core Concepts / Outline
- Integer factorization problem.
- Period finding as the core quantum subroutine.
- Modular exponentiation circuit (black-box view).
- QFT and its role in extracting the period.
- Classical post-processing to obtain factors.

### Theory & Intuition
Shor’s algorithm factors a composite integer N in polynomial time, using a quantum subroutine to find the period r of the function:
\[
f(x) = a^x \mod N.
\]
The quantum part prepares a superposition over x, computes f(x) in a second register, and uses the QFT to extract information about r. Classical algorithms then compute the gcd to derive non-trivial factors of N.

### Example
Work through a toy example of factoring N = 15 with a chosen base a (e.g., a = 2), tracing the period-finding idea (without fully building the full circuit).

### In-Class Exercises
1. Given N and a, compute f(x) = a^x mod N for small x and guess the period r.  
2. Discuss qualitatively why period finding can be done efficiently quantumly but not classically (known algorithms).

### Homework
1. Read a simplified explanation of Shor’s algorithm from a textbook or online source and write a 1-page summary.  
2. Implement the QFT on a small number of qubits (3 or 4) in Qiskit as a separate exercise (see next session).

### Python/Qiskit Coding
(Focus on QFT building blocks, see Session 11 for concrete code.)

### Suggested Figures
- Block diagram of Shor’s algorithm showing classical and quantum components.  
- Illustration of the function f(x) = a^x mod N as a periodic signal.  
- Circuit diagram showing registers and QFT block (high level).


---

## Session 11: Quantum Fourier Transform (QFT)


### Learning Objectives
By the end of this session, students should be able to:
- Define the Quantum Fourier Transform (QFT).
- Write down the QFT circuit for n qubits.
- Implement QFT and its inverse in Qiskit.
- Understand the role of QFT in algorithms like Shor’s and phase estimation.

### Core Concepts / Outline
- Definition of QFT on N = 2^n states.
- Decomposition of QFT into Hadamard and controlled-phase gates.
- Complexity O(n^2) of QFT circuit.
- Inverse QFT and its implementation.

### Theory & Intuition
The QFT on N = 2^n basis states is defined by:
\[
\text{QFT} \lvert x \rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi i xk/N} \lvert k \rangle.
\]
The circuit can be decomposed into a sequence of H and controlled-phase gates, along with swaps to reverse qubit order.

### Example
Write the QFT on 2 qubits explicitly and compare with the matrix decomposition.

### In-Class Exercises
1. Draw the QFT circuit for n = 3 qubits.  
2. Discuss how to implement the inverse QFT by reversing the order of gates and conjugating phase angles.

### Homework
1. Implement QFT for n = 3 qubits in Qiskit and test it on computational basis states.  
2. Explore approximating QFT by omitting some small-angle controlled-phase gates.

### Python/Qiskit Coding
```python
from qiskit import QuantumCircuit
import numpy as np

def qft(n):
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j+1, n):
            qc.cp(np.pi / (2**(k-j)), k, j)
    # Swap qubits to reverse order
    for i in range(n//2):
        qc.swap(i, n-1-i)
    return qc

qc = qft(3)
print(qc.draw())
```

### Suggested Figures
- Circuit diagram of QFT for 3 qubits.  
- Table comparing classical FFT and quantum QFT (structure and complexity).


---

## Session 12: Quantum Error Correction (QEC)


### Learning Objectives
By the end of this session, students should be able to:
- Explain why quantum error correction is necessary.
- Describe the basic idea behind redundancy and encoding.
- Understand simple codes such as the 3-qubit bit-flip code conceptually.
- Recognize the notions of syndrome measurement and recovery.

### Core Concepts / Outline
- Fragility of quantum information and no-cloning theorem.
- Bit-flip and phase-flip errors.
- Encoding logical qubits into multiple physical qubits.
- Syndrome measurement using ancilla qubits.
- High-level description of Shor code or Steane code.

### Theory & Intuition
Quantum error correction encodes a logical qubit into multiple physical qubits, so that errors affecting a subset of the qubits can be detected and corrected. Although we cannot measure the state directly without collapsing it, we can measure **syndromes**—parity checks that reveal where an error occurred without revealing the logical state.

### Example
Discuss the 3-qubit bit-flip code which encodes:
\[
|0_L\rangle = |000\rangle,\quad
|1_L\rangle = |111\rangle.
\]
Explain how majority voting can correct a single bit-flip error.

### In-Class Exercises
1. Show how the bit-flip code corrects an X error on any one qubit.  
2. Discuss why phase-flip errors also need to be corrected, and outline the phase-flip code.

### Homework
1. Read about Shor’s 9-qubit code and summarize its structure.  
2. Write a conceptual explanation of syndrome measurement without collapsing the logical state.

### Python/Qiskit Coding
(Qiskit Ignis or newer tools may be used; for simplicity, you may just simulate errors and majority voting in Python.)

### Suggested Figures
- Encoding diagram for the 3-qubit bit-flip code.  
- Syndrome measurement circuit sketch.  
- Conceptual block diagram of an error-correcting cycle.


---

## Session 13: Variational Quantum Algorithms: VQE & QAOA


### Learning Objectives
By the end of this session, students should be able to:
- Explain the idea behind variational (hybrid) quantum-classical algorithms.
- Describe the Variational Quantum Eigensolver (VQE) and its workflow.
- Understand QAOA as an example of a variational algorithm for optimization.
- Implement a simple VQE-like circuit in Qiskit.

### Core Concepts / Outline
- NISQ era and limitations of deep circuits.
- Parameterized quantum circuits (ansätze).
- Classical optimizer loop (gradient-free or gradient-based).
- VQE for approximating ground-state energies.
- QAOA for combinatorial optimization.

### Theory & Intuition
Variational algorithms use parameterized circuits:
\[
|\psi(\vec{\theta})\rangle = U(\vec{\theta})|0...0\rangle,
\]
and a classical optimizer that updates parameters \(\vec{\theta}\) to minimize a cost function, often an expectation value of a Hamiltonian.

### Example
Build a simple 1- or 2-qubit variational circuit with a few rotation gates and use a classical optimizer to minimize the expectation of Z on one qubit.

### In-Class Exercises
1. Sketch the workflow of VQE (quantum subroutine + classical optimizer).  
2. Discuss why variational circuits can be more noise-resilient than deep non-parameterized circuits.

### Homework
1. Implement a simple variational circuit in Qiskit and manually tune parameters to minimize a simple cost.  
2. Read a short article or paper on VQE and summarize it.

### Python/Qiskit Coding
```python
from qiskit import QuantumCircuit, Aer, execute
import numpy as np

def ansatz(theta):
    qc = QuantumCircuit(1,1)
    qc.ry(theta, 0)
    qc.measure(0,0)
    return qc

backend = Aer.get_backend('qasm_simulator')

def expectation(theta):
    qc = ansatz(theta)
    result = execute(qc, backend, shots=1024).result()
    counts = result.get_counts()
    # Expectation of Z: P(0) - P(1)
    p0 = counts.get('0', 0) / 1024
    p1 = counts.get('1', 0) / 1024
    return p0 - p1
```

### Suggested Figures
- Workflow diagram: quantum device ↔ classical optimizer loop.  
- Simple parameterized circuit with RY(θ) gates.  
- Conceptual figure illustrating a cost landscape and parameter updates.


---

## Session 14: Quantum Hardware and NISQ Era


### Learning Objectives
By the end of this session, students should be able to:
- Identify main physical platforms for quantum hardware.
- Describe superconducting qubits, trapped ions, and photonic qubits at a high level.
- Understand what NISQ means and its implications.
- Recognize hardware constraints: decoherence times, gate errors, connectivity.

### Core Concepts / Outline
- Superconducting qubits (e.g., transmons).
- Trapped ion qubits.
- Photonic qubits.
- NISQ devices: noisy, intermediate-scale quantum.
- Hardware metrics: T1, T2, gate fidelity, connectivity graphs.

### Theory & Intuition
Quantum hardware implementations differ in:
- How qubits are physically realized.
- How gates are implemented (microwave pulses, laser interactions, optical elements).
- Their strengths and weaknesses: scalability, coherence times, gate speeds.

The NISQ era is characterized by devices with tens to a few hundred qubits, limited coherence, and noise—too small and noisy for full fault-tolerant computation, but large enough to explore interesting algorithms.

### Example
Compare superconducting qubits and trapped ions in terms of:
- Typical coherence times.
- Gate speed.
- Connectivity between qubits.

### In-Class Exercises
1. Students research one hardware platform and present a short summary.  
2. Discuss what kinds of algorithms might be feasible on NISQ hardware.

### Homework
1. Choose a specific quantum hardware vendor or platform (e.g., IBM, IonQ, Xanadu) and write a short report about their technology.  
2. Explain why error correction is challenging yet essential for scalable quantum computers.

### Python/Qiskit Coding
```python
from qiskit import IBMQ

# (If using IBM Quantum Experience)
# IBMQ.load_account()
# provider = IBMQ.get_provider()
# backend = provider.get_backend('ibmq_qasm_simulator')
# print(backend.configuration())
```

### Suggested Figures
- Comparison table of different hardware platforms.  
- Conceptual diagram of a superconducting qubit circuit or ion trap.  
- Illustration of a connectivity graph for a real device.


---

## Session 15: Final Project Design and Presentations


### Learning Objectives
By the end of this session, students should be able to:
- Design a small quantum algorithm or experiment from scratch.
- Translate a problem statement into a quantum circuit and/or variational ansatz.
- Implement, simulate, and analyze results using Qiskit.
- Present their project clearly and critically discuss limitations.

### Core Concepts / Outline
- Project design process:
  - Choose a problem.
  - Select appropriate quantum model/algorithm.
  - Design circuit and simulation strategy.
- Analysis and interpretation of results.
- Limitations due to noise, qubit count, and depth.
- Future directions and open questions.

### Theory & Intuition
This is a synthesis session where students bring together:
- Theoretical understanding of qubits, gates, circuits, and algorithms.  
- Practical skills in Qiskit and simulation.  
- Awareness of hardware constraints and noise.

Projects can be algorithm-focused (e.g., implementing Grover or VQE for a particular toy problem) or concept-focused (e.g., exploring the effect of noise on entanglement).

### Example Project Ideas
- Implement Grover’s algorithm for a specific search problem of size N = 8 or 16.  
- Build a small VQE circuit to approximate the ground-state energy of a 2-qubit Hamiltonian.  
- Explore decoherence by applying simple noise models and measuring entanglement decay.

### In-Class Activities
- Students present short project proposals and receive feedback.  
- Work in small groups to refine algorithms and circuits.  
- Live debugging of circuits and code with instructor guidance.

### Homework / Final Assignment
- Complete a mini-project:
  - 4–8 pages report in LaTeX or Markdown, including motivation, method, results, and discussion.
  - Qiskit code as an appendix or GitHub repository.
- Prepare a short presentation (5–10 minutes) summarizing the project.

### Python/Qiskit Coding
(Project-dependent; students are expected to integrate all previous knowledge and examples.)

### Suggested Figures
- Flowchart of quantum project workflow: Problem → Algorithm → Circuit → Simulation → Results.  
- Circuit diagrams from students’ projects (e.g., Grover, VQE, Bell tests).  
- Plots of measurement statistics or cost function vs. iteration for variational algorithms.


---

