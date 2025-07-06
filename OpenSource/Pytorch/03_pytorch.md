PyTorch dynamically selects between **CPU** and **CUDA** kernels for operations like `torch.add()` based on the **device** and **dtype** of input tensors. Here’s a detailed breakdown of the decision-making process:

---

### **1. Key Mechanism: Dispatch by Device**
When you call an operation (e.g., `torch.add(x, y)`), PyTorch’s **dispatcher** resolves the correct kernel using the following steps:

#### **Step 1: Inspect Input Tensor Properties**
- The dispatcher checks the **`device`** (`x.device`) and **`dtype`** (`x.dtype`) of input tensors.
  - Example: If `x` is a `CUDA` tensor, the dispatcher looks for a CUDA kernel.
  - If inputs are on different devices (e.g., `x` on CPU, `y` on CUDA), PyTorch raises an error (unless explicitly handled, like with `.to(device)`).

#### **Step 2: Dispatch Key Resolution**
- Each tensor has a **`DispatchKeySet`** (a set of "tags" like `CPU`, `CUDA`, `Autograd`, etc.).
- The dispatcher selects the **highest-priority kernel** registered for the combination of:
  - **Operator name** (e.g., `add`).
  - **Dispatch keys** (e.g., `CUDA` + `Autograd` if `requires_grad=True`).

#### **Step 3: Kernel Execution**
- The selected kernel (e.g., `add_cuda_kernel`) is executed.
- If no kernel is found for the exact device/dtype, PyTorch may:
  - **Promote dtypes** (e.g., `int32` → `float32`).
  - **Move tensors to the same device** (if possible, but usually requires explicit `.to(device)`).

---

### **2. Behind the Scenes: How Kernels Are Registered**
PyTorch registers kernels for specific backends at compile time. For example:
```cpp
// CUDA kernel registration (in aten/src/ATen/native/cuda/Add.cu)
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("add", add_cuda_kernel);
}

// CPU kernel registration (in aten/src/ATen/native/Add.cpp)
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("add", add_cpu_kernel);
}
```
- The `TORCH_LIBRARY_IMPL` macro binds the `add` operation to implementations for `CPU` or `CUDA`.

---

### **3. Special Cases**
#### **(A) Autograd (Gradient Tracking)**
- If any input has `requires_grad=True`, the dispatcher first routes the call to the **Autograd** key.
- Autograd records the operation for backpropagation and then invokes the actual kernel (e.g., `CUDA`).

#### **(B) Mixed Devices**
- PyTorch **does not** automatically transfer data between devices. For example:
  ```python
  x_cpu = torch.tensor([1.0])
  y_cuda = torch.tensor([2.0], device="cuda")
  torch.add(x_cpu, y_cuda)  # Error: "Expected all tensors to be on the same device"
  ```
  - Fix: Explicitly move tensors to the same device:
    ```python
    torch.add(x_cpu.to("cuda"), y_cuda)  # Works
    ```

#### **(C) Fallback Behavior**
- If a CUDA kernel isn’t available, PyTorch may:
  - **Fall back to CPU** (rare, usually fails instead).
  - Raise an error (e.g., "Operator `add` not implemented for CUDA").

---

### **4. Debugging Dispatch**
To see how PyTorch selects kernels:
```python
# Enable dispatch tracing
torch._C._set_print_dispatcher(True)

x = torch.tensor([1.0], device="cuda")
y = torch.tensor([2.0], device="cuda")
torch.add(x, y)  # Prints dispatch traces (e.g., "CPU", "CUDA", "Autograd")
```

---

### **5. Why This Matters for Contributors**
- **Adding a new operator?** You must:
  1. Implement kernels for all supported devices (CPU/CUDA/etc.).
  2. Register them with the dispatcher.
- **Optimizing performance?** Ensure the correct kernel is chosen (e.g., avoid unnecessary device transfers).

---

### **Summary**
PyTorch’s dispatcher:
1. **Checks tensor properties** (device/dtype).
2. **Selects the highest-priority kernel** registered for the operator + dispatch key.
3. **Fails fast** if no valid kernel exists (no silent fallbacks across devices).

This design ensures efficient, device-aware execution while maintaining flexibility. Would you like to explore how custom ops are dispatched? 🚀