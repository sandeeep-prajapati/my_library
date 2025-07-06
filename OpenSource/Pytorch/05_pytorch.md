PyTorch rigorously tests low-level ops (like `add`, `reshape`, `view`) across **devices** (CPU/CUDA), **dtypes**, and **edge cases**. Here’s where and how these tests are implemented:

---

### **1. Test File Locations**
Low-level ops are tested primarily in these directories:
- **`test/test_torch.py`** – Core ops (e.g., `add`, `mul`, `reshape`).
- **`test/test_ops.py`** – OpInfo-based tests (comprehensive dtype/device coverage).
- **`test/test_view_ops.py`** – Tests for `view`, `reshape`, `transpose`, etc.
- **`test/test_unary_ufuncs.py`** – Unary ops (e.g., `sin`, `exp`).
- **`test/test_binary_ufuncs.py`** – Binary ops (e.g., `add`, `mul`).
- **`test/test_cuda.py`** – CUDA-specific behavior.

---

### **2. Testing Mechanisms**
#### **(A) OpInfo-Based Testing (Most Comprehensive)**
PyTorch uses **`OpInfo`** (in `common_methods_invocations.py`) to systematically test ops:
- **Dtype/Device Coverage**: Tests all combinations (e.g., `float32` on CPU, `int64` on CUDA).
- **Gradient Checks**: Verifies autograd support via `gradcheck`.
- **Reference Numerics**: Compares against NumPy or manual implementations.

Example for `add`:
```python
# test/test_ops.py
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_methods_invocations import op_db

class TestAdd(TestCase):
    @ops(op_db, allowed_dtypes=(torch.float32,))
    def test_add(self, device, dtype, op):
        a = torch.tensor([1.0], device=device, dtype=dtype)
        b = torch.tensor([2.0], device=device, dtype=dtype)
        self.assertEqual(op(a, b), torch.tensor([3.0], device=device, dtype=dtype))
```

Key `OpInfo` Fields:
| Field          | Purpose                                                                 |
|----------------|-------------------------------------------------------------------------|
| `dtypes`       | Supported dtypes (e.g., `float32`, `int64`).                           |
| `supports_autograd` | Whether the op supports backward passes.                          |
| `sample_inputs` | Predefined input tensors for edge cases (e.g., empty tensors, NaN). |

---

#### **(B) Direct Unit Tests (Legacy)**
Some tests explicitly check edge cases:
```python
# test/test_torch.py
def test_add(self):
    # Basic
    self.assertEqual(torch.add(torch.tensor(1), torch.tensor(1), 2)
    # Type promotion
    self.assertEqual(torch.add(torch.tensor(1, dtype=torch.int32), torch.tensor(1.0)).dtype, torch.float32)
    # In-place
    x = torch.tensor(1.0)
    torch.add(x, 1, out=x)
    self.assertEqual(x, 2.0)
```

---

#### **(C) View Ops (`view`, `reshape`)**
Special tests for memory-sharing behavior:
```python
# test/test_view_ops.py
def test_view(self):
    x = torch.randn(4, 4)
    y = x.view(16)
    self.assertTrue(y._is_view() and y._base is x)  # Checks memory sharing
    # Test stride computation
    self.assertEqual(y.stride(), (1,))
```

---

### **3. Device-Specific Tests**
- **CUDA vs. CPU Consistency**: Ops are tested to ensure identical results across devices.
- **CUDA-Specific Edge Cases**: E.g., non-contiguous tensors, pinned memory.

Example:
```python
# test/test_cuda.py
def test_add_cuda(self):
    x_cpu = torch.tensor([1.0])
    x_cuda = x_cpu.cuda()
    self.assertEqual(torch.add(x_cpu, x_cpu), torch.add(x_cuda, x_cuda).cpu())
```

---

### **4. Autograd/Gradient Testing**
- **`gradcheck`/`gradgradcheck`**: Numerically verifies gradients.
- **Autograd-specific edge cases**: E.g., non-differentiable inputs.

```python
# test/test_autograd.py
def test_add_autograd(self):
    x = torch.randn(3, requires_grad=True)
    y = torch.randn(3, requires_grad=True)
    torch.autograd.gradcheck(lambda x, y: x + y, (x, y))
```

---

### **5. Performance Testing**
Benchmarks (not correctness tests) live in:
- `benchmarks/` – Checks op speed across devices/dtypes.

---

### **6. How to Run Tests**
```bash
# Run all tests
pytest test/test_torch.py -v

# Run a specific test
pytest test/test_torch.py::TestAdd -v

# Run with CUDA
PYTORCH_TEST_WITH_CUDA=1 pytest test/test_cuda.py -v
```

---

### **7. Example PRs for Reference**
- [Add OpInfo for `reshape`](https://github.com/pytorch/pytorch/pull/58761)
- [Fix `view` op stride computation](https://github.com/pytorch/pytorch/pull/49112)

---

### **Key Takeaways**
1. **OpInfo-based tests** are the gold standard (dtype/device/autograd coverage).
2. **Legacy unit tests** handle edge cases explicitly.
3. **View ops** require special memory-sharing checks.
4. **Always test across devices** (CPU/CUDA) and dtypes.

Contributing a new op? Start by adding an `OpInfo` entry and follow the patterns above! 🛠️