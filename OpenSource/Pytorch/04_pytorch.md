Defining a new PyTorch operation via `native_functions.yaml` is a structured way to add core operations that integrate with PyTorch’s dispatcher (CPU/CUDA/autograd support). Here’s a step-by-step guide:

---

### **1. Key Files Involved**
- **`native_functions.yaml`**: Declares the operator’s schema and metadata (located in `aten/src/ATen/native/`).
- **`Declarations.yaml`**: (Optional) Additional declarations for code generation.
- **Implementation Files**: C++/CUDA code in `aten/src/ATen/native/`.

---

### **2. Step-by-Step Process**

#### **Step 1: Define the Operator in `native_functions.yaml`**
Add an entry specifying:
- **Operator name** (e.g., `my_op`)
- **Schema** (input/output types, mutability)
- **Dispatch keys** (CPU, CUDA, Autograd, etc.)

Example:
```yaml
- func: my_op(Tensor self, Tensor other) -> Tensor
  dispatch:
    CPU: my_op_cpu
    CUDA: my_op_cuda
  autogen: my_op.out  # (Optional) Generates out= variant
```

Key Fields:
| Field       | Purpose                                                                 |
|-------------|-------------------------------------------------------------------------|
| `func`      | Function signature (PyTorch-style type annotations).                    |
| `dispatch`  | Maps backend (CPU/CUDA) to kernel names.                                |
| `variants`  | `function` (default), `method`, or `out` (for out= variants).           |
| `autogen`   | Automatically generates boilerplate (e.g., `out=` variants).            |

---

#### **Step 2: Implement Kernels**
Place implementations in:
- **CPU**: `aten/src/ATen/native/MyOp.cpp`
- **CUDA**: `aten/src/ATen/native/cuda/MyOp.cu`

Example (CPU):
```cpp
// MyOp.cpp
#include <ATen/ATen.h>

namespace at {
namespace native {

Tensor my_op_cpu(const Tensor& self, const Tensor& other) {
  // Your CPU implementation
  Tensor output = ...;
  return output;
}

}} // namespace at::native
```

Example (CUDA):
```cpp
// MyOp.cu
#include <ATen/ATen.h>

namespace at {
namespace native {

Tensor my_op_cuda(const Tensor& self, const Tensor& other) {
  // Launch CUDA kernel
  dim3 blocks(...);
  my_op_kernel<<<blocks, threads>>>(...);
  return output;
}

}} // namespace at::native
```

---

#### **Step 3: Register Kernels**
Use `TORCH_LIBRARY_IMPL` to link the YAML declaration to your kernels:
```cpp
// In MyOp.cpp
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("my_op", TORCH_FN(my_op_cpu));
}

// In MyOp.cu
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("my_op", TORCH_FN(my_op_cuda));
}
```

---

#### **Step 4: Build and Test**
1. **Recompile PyTorch**:
   ```bash
   python setup.py develop --cmake  # Debug build
   ```
2. **Test in Python**:
   ```python
   import torch
   x = torch.randn(3, device='cpu')
   y = torch.randn(3, device='cuda')
   torch.ops.aten.my_op(x, x)  # CPU
   torch.ops.aten.my_op(y, y)  # CUDA
   ```

---

### **3. Special Cases**
#### **(A) Autograd Support**
To enable autograd:
1. Add `autograd` key in `native_functions.yaml`:
   ```yaml
   dispatch:
     Autograd: my_op_autograd
   ```
2. Implement a **autograd Function** in `tools/autograd/derivatives.yaml`:
   ```yaml
   - name: my_op(Tensor self, Tensor other) -> Tensor
     self: grad * other
     other: grad * self
   ```

#### **(B) Out= Variant**
If your op supports an `out=` parameter:
```yaml
- func: my_op.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CPU: my_op_out_cpu
    CUDA: my_op_out_cuda
```

---

### **4. Debugging Tips**
- **Codegen Output**: Check generated code in `build/aten/src/ATen/RegisterCPU.cpp`.
- **Dispatch Tracing**:
  ```python
  torch._C._set_print_dispatcher(True)
  torch.ops.aten.my_op(...)
  ```

---

### **5. Example PRs**
- [Add a simple operator](https://github.com/pytorch/pytorch/pull/43921)
- [Add an operator with autograd](https://github.com/pytorch/pytorch/pull/49171)

---

### **Summary**
1. **Declare** in `native_functions.yaml`.
2. **Implement** CPU/CUDA kernels.
3. **Register** with `TORCH_LIBRARY_IMPL`.
4. **Test** with `torch.ops.aten`.

This ensures your op works seamlessly with PyTorch’s dispatcher, autograd, and serialization. Need help with a specific op design? 🛠️