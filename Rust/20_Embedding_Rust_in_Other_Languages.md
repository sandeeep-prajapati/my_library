
Rust was *designed* to integrate with other languages safely — whether that’s **C**, **Python**, or **JavaScript** — through tools like **FFI (Foreign Function Interface)** and **WebAssembly (WASM)**.

Let’s explore this in three major sections 👇

---

## 🧩 1. Integrating **Rust with C** (via FFI)

Rust and C are both **low-level languages** that compile to native machine code.
The **Foreign Function Interface (FFI)** lets Rust call C functions — or expose Rust functions to be called from C.

---

### ✅ A. Calling C Code from Rust

**Step 1:** Create a C header file (`mathlib.h`)

```c
// mathlib.h
int add(int a, int b);
```

**Step 2:** Implement it (`mathlib.c`)

```c
// mathlib.c
int add(int a, int b) {
    return a + b;
}
```

**Step 3:** Call it from Rust using `extern "C"`

```rust
// main.rs
#[link(name = "mathlib")]
extern "C" {
    fn add(a: i32, b: i32) -> i32;
}

fn main() {
    unsafe {
        println!("Result from C: {}", add(5, 7));
    }
}
```

**Step 4:** Compile and link

```bash
gcc -c mathlib.c -o libmathlib.a
cargo build
```

✅ **Note:** `unsafe` is required because Rust can’t verify the safety of foreign code.

---

### ✅ B. Calling Rust from C

**Step 1:** Write Rust code and export functions

```rust
// src/lib.rs
#[no_mangle]
pub extern "C" fn multiply(a: i32, b: i32) -> i32 {
    a * b
}
```

* `#[no_mangle]` prevents Rust from renaming the function.
* `extern "C"` ensures C-compatible calling conventions.

**Step 2:** Build a static or shared library

```bash
cargo build --release
```

You’ll find:

```
target/release/libyourlib.a   (static)
target/release/libyourlib.so  (shared, on Linux)
```

**Step 3:** Use it in C

```c
#include <stdio.h>

int multiply(int a, int b); // Declare Rust function

int main() {
    printf("Result: %d\n", multiply(3, 4));
    return 0;
}
```

✅ Works seamlessly for performance-critical native modules (like image processing or cryptography).

---

## 🐍 2. Integrating **Rust with Python**

Rust can act as a **Python extension module** — enabling you to write performance-critical code in Rust and call it directly from Python.

There are two popular approaches:

### 🦀 Option 1: Using `pyo3` (Most Common)

#### Step 1: Add dependencies

```bash
cargo add pyo3 --features extension-module
```

#### Step 2: Write Rust code

```rust
use pyo3::prelude::*;

#[pyfunction]
fn add(a: i32, b: i32) -> PyResult<i32> {
    Ok(a + b)
}

#[pymodule]
fn rustmath(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}
```

#### Step 3: Build the Python module

```bash
maturin develop
```

> You’ll need to install [`maturin`](https://github.com/PyO3/maturin):
> `pip install maturin`

#### Step 4: Use in Python

```python
import rustmath
print(rustmath.add(3, 5))  # Output: 8
```

✅ **Benefits:**

* No unsafe code needed.
* Integrates seamlessly with Python’s packaging system.
* Great for AI/ML performance optimization.

---

### 🐍 Option 2: Using `cffi` (More Manual)

You can expose C-style functions from Rust and call them from Python using the built-in `ctypes` or `cffi` modules.

Rust:

```rust
#[no_mangle]
pub extern "C" fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

Python:

```python
from ctypes import cdll
lib = cdll.LoadLibrary("./target/release/librustlib.so")
print(lib.add(3, 7))
```

✅ Simpler setup, but less Pythonic than `pyo3`.

---

## 🌐 3. Integrating **Rust with JavaScript** (via WebAssembly)

Rust can compile directly to **WebAssembly (WASM)** — allowing it to run safely inside web browsers or Node.js.

---

### 🦀 Step 1: Add Tooling

Install WebAssembly target and wasm-pack:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

---

### 🦀 Step 2: Create a Library

```bash
cargo new --lib wasm_greet
cd wasm_greet
```

`Cargo.toml`:

```toml
[dependencies]
wasm-bindgen = "0.2"
```

---

### 🦀 Step 3: Write Rust Code

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Hello, {} from Rust!", name)
}
```

---

### 🦀 Step 4: Build for WebAssembly

```bash
wasm-pack build --target web
```

It generates a `pkg/` folder containing:

* `wasm_greet_bg.wasm` → compiled WebAssembly module
* `wasm_greet.js` → JS bindings for easy use

---

### 🦀 Step 5: Use in JavaScript (Browser Example)

```html
<script type="module">
import init, { greet } from "./pkg/wasm_greet.js";

async function run() {
    await init();
    console.log(greet("Sandeep"));
}
run();
</script>
```

✅ Output in browser console:

```
Hello, Sandeep from Rust!
```

---

### 🧠 Bonus: Node.js Integration

You can target Node instead:

```bash
wasm-pack build --target nodejs
```

Then in Node:

```js
import { greet } from './pkg/wasm_greet.js';
console.log(greet("Rust + Node.js"));
```

---

## 🚀 4. Summary Table

| Integration       | Tool/Crate                  | Communication Type    | Use Case                        |
| ----------------- | --------------------------- | --------------------- | ------------------------------- |
| C ↔ Rust          | FFI (`extern "C"`)          | Native ABI            | System libraries, OS-level work |
| Python ↔ Rust     | `pyo3`, `maturin`, `cffi`   | C-ABI, native module  | Fast Python extensions          |
| JavaScript ↔ Rust | `wasm-bindgen`, `wasm-pack` | WebAssembly           | Web, Node.js modules            |
| Other Languages   | `cbindgen`, `uniffi`        | Language-specific FFI | Kotlin, Swift, C#, etc.         |

---

## ⚠️ 5. Common Pitfalls

| Pitfall                   | Description                             | Best Practice                                        |
| ------------------------- | --------------------------------------- | ---------------------------------------------------- |
| Memory ownership mismatch | C/Python/JS don’t manage Rust’s memory  | Use smart pointers or manual `Box::leak` carefully   |
| ABI incompatibility       | Mismatched types or calling conventions | Always use `extern "C"` and `#[repr(C)]`             |
| Performance overhead      | FFI can add call overhead               | Pass simple structs or pointers, avoid large objects |
| Lifetimes                 | FFI ignores Rust lifetimes              | Convert to owned types before FFI boundaries         |

---

## 🧩 6. When to Use Which Integration

| Target           | Best Tool                   | Ideal Scenario                            |
| ---------------- | --------------------------- | ----------------------------------------- |
| C / C++          | `extern "C"`, `cbindgen`    | Embed Rust in native systems or libraries |
| Python           | `pyo3`, `maturin`           | Replace performance bottlenecks in Python |
| JavaScript / Web | `wasm-bindgen`, `wasm-pack` | Run Rust in browser or Node.js            |

---

## 🧠 Example Summary

| Integration       | Example                                |
| ----------------- | -------------------------------------- |
| Rust ↔ C          | `#[no_mangle] pub extern "C" fn add()` |
| Rust ↔ Python     | `#[pymodule] fn rustmath()`            |
| Rust ↔ JavaScript | `#[wasm_bindgen] pub fn greet()`       |

---

