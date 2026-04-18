
# 1. Computer Architecture & Microarchitecture

This studies **how CPUs are designed internally**.

Key ideas:

* Instruction pipelines
* Superscalar execution
* Out-of-order execution
* Branch prediction
* Speculative execution

You learn:

* Why CPU performance differs
* How instructions are executed at the hardware level

Example study:

* Build a **simple CPU simulator**

---

# 2. Instruction Set Architecture (ISA)

ISA defines **the language between software and hardware**.

Examples:

* x86
* ARM
* RISC-V

Topics:

* Instruction encoding
* Addressing modes
* Registers
* Privilege levels

Project idea:

* Implement a **RISC-V emulator**.

---

# 3. Operating System Kernel Internals

This is the **heart of system-level programming**.

Key areas:

* Process scheduling
* Virtual memory
* Page tables
* System calls
* Kernel threads
* Interrupt handling

Real study:

* Read **Linux kernel source code**.

Project idea:

* Write a **mini operating system kernel**.

---

# 4. Memory Systems & Cache Hierarchies

Modern CPUs rely heavily on **memory hierarchy**.

Topics:

* L1 / L2 / L3 cache
* Cache coherence protocols
* NUMA architecture
* TLB (Translation Lookaside Buffer)

You learn:

* Why memory latency affects performance.

---

# 5. Compilers & Code Generation

How **high-level languages become machine code**.

Stages:

1. Lexical analysis
2. Parsing
3. Intermediate representation
4. Optimization
5. Code generation

Advanced topics:

* SSA (Static Single Assignment)
* Register allocation
* Loop optimizations

Project:

* Build a **toy compiler**.

---

# 6. Concurrency & Parallelism

Modern computers rely on **multi-core CPUs**.

Topics:

* Memory models
* Lock-free programming
* Atomic operations
* Thread scheduling
* Synchronization primitives

Example:

* Build a **lock-free queue**.

---

# 7. Distributed Systems Internals

Large systems run across **multiple machines**.

Concepts:

* Consensus algorithms
* Fault tolerance
* Replication
* Distributed transactions

Algorithms:

* Paxos
* Raft

Project:

* Build a **distributed key-value store**.

---

# 8. Networking Stack Internals

Instead of using HTTP libraries, understand the **TCP/IP stack inside the OS**.

Topics:

* Packet routing
* Congestion control
* TCP handshake
* Kernel networking

Project:

* Implement a **TCP stack in user space**.

---

# 9. Storage Systems & File Systems

How data is **stored on disk and SSDs**.

Topics:

* Journaling file systems
* Log-structured storage
* B-trees
* SSD wear leveling

Project:

* Build a **custom file system**.

---

# 10. Virtualization & Hypervisors

Running multiple OS instances on the same machine.

Topics:

* Hardware virtualization
* Hypervisors
* Virtual memory translation
* Container vs VM architecture

Examples:

* KVM
* Xen
* VMware

Project:

* Build a **simple virtual machine monitor**.

---

# 11. Hardware Security

Understanding vulnerabilities at the hardware level.

Topics:

* Side-channel attacks
* Spectre
* Meltdown
* Rowhammer

These attacks exploit **CPU microarchitecture**.

---

# 12. Reverse Engineering & Binary Analysis

Understanding programs **without source code**.

Tools:

* Ghidra
* IDA
* Radare2

Topics:

* Binary disassembly
* Malware analysis
* Memory inspection

Project:

* Reverse engineer a compiled program.

---

# What Makes These “MIT-Level”

These topics require understanding across **multiple layers**:

```
Hardware
   ↓
Instruction Set
   ↓
Operating System
   ↓
Compilers
   ↓
Applications
```

Most engineers only understand the **top layer**.

Deep engineers understand **the entire stack**.

---
