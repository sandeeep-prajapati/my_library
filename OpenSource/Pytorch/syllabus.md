
---

🔍 Understanding PyTorch Internals (Folder Structure, Architecture)

1. 🧠 Explore the role of torch/ vs aten/ vs c10/ directories

> Prompt: “What is the purpose of each major directory inside the PyTorch repo?”



torch/: Python API

aten/: C++ backend kernels (tensor operations)

c10/: Core utilities, type definitions, and dispatcher



---

2. ⚙️ Trace what happens when you call torch.add(x, y)

> Prompt: “How does PyTorch dispatch a call like torch.add() down to a CUDA or CPU kernel?”



Explore: Python frontend → dispatcher → backend op in aten/src/ATen/native/



---

3. 🔌 Understand the Dispatcher (C10 Dispatcher)

> Prompt: “How does PyTorch decide whether to use the CPU or CUDA kernel for an operation?”



Learn about: RegisterOperators.cpp, native_functions.yaml, dispatch keys



---

4. 📦 Study native_functions.yaml and its role in op registration

> Prompt: “How do I define a new PyTorch operation using native_functions.yaml?”




---

5. 🧪 Understand how PyTorch tests core tensor ops (test/test_torch.py)

> Prompt: “Where and how are low-level ops like add, reshape, view tested in PyTorch?”




---

🚀 Contribution-Specific Prompts

6. 🔧 Find and fix docstring issues or missing parameter descriptions in modules

> Prompt: “Can I improve or fix inconsistencies in the docstring for torch.nn.Conv2d?”




---

7. 🧪 Add unit tests for under-tested functions in torch.nn.functional

> Prompt: “Which functions in torch.nn.functional are poorly tested, and can I add edge cases?”




---

8. ✍️ Write tutorials using features like torch.fx, torch.compile, or torch.utils.benchmark

> Prompt: “Can I add tutorials for new features or performance tools introduced in PyTorch 2.x?”




---

9. 🧵 Participate in triaging open issues (labeling, reproduction, comments)

> Prompt: “Can I help verify and reproduce unconfirmed bugs on open GitHub issues?”




---

10. 🧰 Improve setup/build documentation for new contributors

> Prompt: “Can I update CONTRIBUTING.md to add missing setup instructions for Windows/M1 Mac?”




---

💎 Good Practices PyTorch Maintainers Follow

11. ✅ Learn their Pull Request Standards

> Prompt: “What does a high-quality PR look like in PyTorch? What are the expectations?”



Well-scoped

CI passing

Contains tests and benchmarks

Clear PR title like: fix: wrong output shape in torch.nn.Unfold

PR linked to GitHub Issue



---

12. 🧪 Follow testing practices: write deterministic, reproducible tests

> Prompt: “How does PyTorch ensure its tests are device-agnostic (CPU/CUDA) and reproducible?”




---

13. 📊 Use pytest, assertEqual, and torch’s test decorators properly

> Prompt: “How do I write @onlyCUDA or @skipIfNoMPS tests like PyTorch maintainers?”




---

14. 🔎 Review how code is benchmarked before being merged

> Prompt: “What benchmarking tools do PyTorch contributors use to measure performance regression?”



> Explore: torch.utils.benchmark, nvprof, timeit, pytest-benchmark




---

🏗️ Bigger Problem Preparation Prompts

15. 📌 Pick a real kernel bug (e.g., torch.matmul overflow) and reproduce it

> Prompt: “How can I reproduce and trace a RuntimeError in a CUDA kernel in ATen?”




---

16. 🧰 Create a new operator with autograd and add test cases

> Prompt: “How do I define a new tensor op with forward/backward support?”



> Files to touch:



native_functions.yaml

aten/src/ATen/native/

torch/autograd/

test/test_autograd.py



---

17. ⚡ Trace the FX Graph for a model and modify it manually

> Prompt: “How can I use torch.fx to inspect and transform a ResNet model?”




---

18. 🔄 Understand PT2 (torch.compile) and TorchInductor flow

> Prompt: “What happens internally when I run torch.compile(model) in PyTorch 2.0+?”




---

19. 🧠 Contribute to TorchEval or TorchTNT by adding new metrics/trainers

> Prompt: “Can I add a new training loop metric (e.g. Top-3 Accuracy) in TorchEval?”




---

20. 🚦 Fix or rewrite failing tests in low-level ops like scatter, index_add, etc.

> Prompt: “Can I trace why an indexing op like index_add fails with specific dtype or shape?”




---

✅ Final Tips for Solving Bigger Problems

Start small but deep: Fix a test, then learn how it connects to ATen, dispatch, and autograd.

Join PyTorch forums or dev-discussions: https://dev-discuss.pytorch.org/

Follow the @oncall GitHub labels (e.g., oncall: pt2, oncall: distributed) to see real core issues.

Subscribe to PyTorch PRs and learn how senior engineers write code.



---

📌 Want Me To Help You?

I can:

Show you a beginner issue in torch.nn.functional or aten now

Help you fork, clone, and set up PyTorch

Guide you in your first real contribution (step-by-step)