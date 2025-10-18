In Rust, testing is a **first-class feature** of the language — the compiler and toolchain (`cargo test`) are designed to make writing, running, and organizing tests easy and efficient. Rust supports three major testing types: **unit tests**, **integration tests**, and **benchmarks**.

Let’s go through each with examples 👇

---

## 🧩 1. Unit Tests

Unit tests focus on **individual modules or functions**. They usually live in the **same file** as the code being tested, inside a special `#[cfg(test)]` module.

### ✅ Example:

```rust
// src/lib.rs
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*; // import functions from parent module

    #[test]
    fn test_add_positive() {
        assert_eq!(add(2, 3), 5);
    }

    #[test]
    fn test_add_negative() {
        assert_eq!(add(-2, -3), -5);
    }

    #[test]
    #[should_panic]
    fn test_panic_case() {
        panic!("This test should panic");
    }
}
```

**Run tests:**

```bash
cargo test
```

**What happens:**

* Rust compiles the code in *test mode* (includes `#[cfg(test)]`).
* It runs all functions marked with `#[test]`.
* `assert!`, `assert_eq!`, and `assert_ne!` are the main macros used for validation.

---

## 🧠 2. Integration Tests

Integration tests check the **public API of your crate**.
They live in a **separate directory**: `tests/` at the project root.
Each file inside `tests/` is compiled as a separate crate.

### ✅ Example:

```rust
// src/lib.rs
pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}
```

```rust
// tests/integration_test.rs
use my_crate_name::multiply;

#[test]
fn test_multiply() {
    assert_eq!(multiply(3, 4), 12);
}
```

**Run integration tests:**

```bash
cargo test
```

**Note:**

* You **don’t** need `#[cfg(test)]` for integration tests.
* These tests use the **public API only** — a great way to ensure modularity.

---

## ⏱️ 3. Benchmarks (Performance Testing)

Benchmark tests measure **execution speed**.
They require the **nightly compiler** and the **`test` crate**.

### ✅ Example:

Add this to your `Cargo.toml`:

```toml
[dev-dependencies]
test = "0.1.0"
```

Then in your code:

```rust
#![feature(test)]
extern crate test;

pub fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_fibonacci(b: &mut Bencher) {
        b.iter(|| fibonacci(20));
    }
}
```

**Run benchmarks:**

```bash
cargo +nightly bench
```

---

## ⚡ Best Practices

| ✅ Good Practice                 | 💡 Description                                                   |
| ------------------------------- | ---------------------------------------------------------------- |
| Use `assert_eq!` & `assert_ne!` | For simple equality comparisons                                  |
| Isolate tests                   | Keep each test independent — no shared mutable state             |
| Use `#[should_panic]`           | To check expected panics                                         |
| Use `cargo test -- --nocapture` | To print debug output during tests                               |
| Organize tests logically        | Unit tests in source files, integration tests in `tests/` folder |
| Use mocks/fakes when needed     | Especially for I/O or network-heavy functions                    |
| Add CI/CD testing               | Run `cargo test` in GitHub Actions or pipelines                  |

---

## 🧰 Bonus: Testing Async Code

If you’re using async functions (e.g., with `tokio`), wrap tests with the `#[tokio::test]` macro:

```rust
#[tokio::test]
async fn test_async_api() {
    let result = async_function().await;
    assert_eq!(result, 42);
}
```

---

## 🏁 Summary

| Test Type        | Location                          | Purpose                   | Command                |
| ---------------- | --------------------------------- | ------------------------- | ---------------------- |
| Unit Test        | Same file (inside `#[cfg(test)]`) | Test individual functions | `cargo test`           |
| Integration Test | `tests/` directory                | Test public API           | `cargo test`           |
| Benchmark        | Same file, needs nightly          | Measure performance       | `cargo +nightly bench` |

---
