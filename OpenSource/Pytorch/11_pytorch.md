A **high-quality PR** in PyTorch follows strict guidelines to ensure maintainability, performance, and compatibility. Here’s what the core team expects, with actionable checklists and examples:

---

### **1. PR Requirements Checklist**
#### **(A) Before Submission**
| Requirement          | Example |
|----------------------|---------|
| **Linked Issue** | Fixes #12345 (or "Addresses #12345" for partial fixes) |
| **Latest Base** | Rebased on `main` (`git pull upstream main`) |
| **Tests Added** | New `OpInfo` tests for operators, edge cases for bug fixes |
| **Benchmarks** | For performance PRs (use `torch.utils.benchmark`) |
| **Documentation** | Updated docstrings or tutorials (if API changes) |

#### **(B) Code Quality**
| Expectation          | Bad Example | Good Example |
|----------------------|-------------|--------------|
| **Modularity** | Monolithic 300-line function | Split into smaller helper functions |
| **Performance** | Unnecessary `.cpu()` calls | Use `torch.device`-agnostic code |
| **Style** | Inconsistent indentation | Follow PyTorch’s `.clang-format`/`flake8` |
| **Comments** | `# Fix bug here` | `# Needed for stride computation (see Issue #12345)` |

---

### **2. Key Expectations**
#### **(A) Testing**
- **Coverage**: 100% for new features (verified by Codecov).
- **Device/Dtype**: Test all combinations (CPU/CUDA, float32/int64).
- **Autograd**: Use `gradcheck` if applicable.
- **Edge Cases**: Empty tensors, non-contiguous inputs.

Example test for a new op:
```python
@ops(op_db, dtypes=OpDTypes.supported)
def test_my_op(self, device, dtype, op):
    samples = op.sample_inputs(device, dtype)
    for sample in samples:
        self.assertEqual(op(sample.input, *sample.args), expected_output)
```

#### **(B) Performance**
For optimizations:
- Include **before/after benchmarks**:
  ```python
  from torch.utils.benchmark import Timer
  t = Timer("new_op(x)", setup="x=torch.randn(100, device='cuda')")
  print(t.timeit(100))  # Compare with old implementation
  ```
- Avoid **regressions** in memory usage (check with `torch.cuda.memory_stats()`).

#### **(C) Documentation**
- **Docstrings**: Update if API changes (Google style).
- **Tutorials**: For user-facing features (e.g., `torch.compile` guides).
- **Release Notes**: Add to `docs/source/notes/*.rst`.

---

### **3. PR Submission Workflow**
1. **Draft PR**: Early feedback for complex changes.
2. **CI Passing**: All checks (CircleCI, ROCm) must pass.
3. **Review Comments**: Address every comment with fixes or discussions.
4. **Squash Commits**: Before merging (e.g., `git rebase -i main`).

---

### **4. Common Rejection Reasons**
| Reason          | Fix |
|----------------------|---------|
| **Lack of Tests** | Add `OpInfo` or unit tests |
| **Performance Regressions** | Optimize further or revert |
| **BC-breaking** | Add deprecation warnings |
| **Overly Complex** | Split into smaller PRs |

---

### **5. Example High-Quality PRs**
1. **[Add `torch.fft.ihfft2`](https://github.com/pytorch/pytorch/pull/70291)**:
   - Full test coverage (CPU/CUDA, autograd).
   - Docstring with LaTeX math.
2. **[Optimize `nn.LSTM` CUDA](https://github.com/pytorch/pytorch/pull/63544)**:
   - Benchmarks showing 2x speedup.
   - No memory regression.

---

### **6. Pro Tips**
- **Small PRs**: <500 lines (easier to review).
- **Labels**: Tag `module: nn` or `triaged` if applicable.
- **Reviewers**: Mention domain experts (e.g., `@pytorch/vision` for image ops).

---

### **7. Template for PR Description**
```markdown
## Description
Fixes #12345. Adds `torch.my_op` for [purpose].

## Changes
- Implemented core logic in `aten/src/ATen/native/MyOp.cpp`
- Added `OpInfo` tests (100% coverage)
- Updated docs in `docs/source/torch.rst`

## Benchmark (CUDA)
| Before | After |
|--------|-------|
| 10ms   | 5ms   |

## Checklist
- [x] Tests pass
- [x] Documentation updated
- [x] Benchmarks included
```

---

Aim for **clarity, testability, and performance**—your PR will likely be merged faster! 🚀 Let me know if you need help structuring a specific PR.