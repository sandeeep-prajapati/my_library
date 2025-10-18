In Rust 🦀, **macros** are *metaprogramming tools* that let you **generate code at compile time**. They’re more powerful than regular functions because they operate on the **syntax** of the code itself, allowing you to eliminate boilerplate, create domain-specific languages (DSLs), or even generate complex implementations automatically.

---

## 🧩 Types of Macros in Rust

There are **two main types** of macros:

| Type                                  | Example                                                   | Purpose                                                             |
| ------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------- |
| **Declarative macros** (macro_rules!) | `println!`, `vec!`                                        | Pattern-based; match input syntax and expand it                     |
| **Procedural macros**                 | `#[derive(...)]`, `#[attribute]`, `#[function_like(...)]` | Code that manipulates the *Abstract Syntax Tree (AST)* of Rust code |

---

## 🧠 1. Declarative Macros (`macro_rules!`)

Declarative macros use **pattern matching** to transform code — think of them as a way to define new “syntactic sugar”.

### ✅ Example: Simple `say_hello!` Macro

```rust
// Declare a simple macro
macro_rules! say_hello {
    () => {
        println!("Hello, Rustacean!");
    };
}

fn main() {
    say_hello!(); // Expands to: println!("Hello, Rustacean!");
}
```

### 🔁 Example: Repetitions and Patterns

You can define macros that take arguments and repeat over them:

```rust
macro_rules! create_vector {
    ($($x:expr),*) => {
        {
            let mut v = Vec::new();
            $(v.push($x);)*
            v
        }
    };
}

fn main() {
    let numbers = create_vector![10, 20, 30];
    println!("{:?}", numbers); // [10, 20, 30]
}
```

📘 *How it works:*

* `$(...)*` means “repeat this pattern zero or more times”.
* `$x:expr` captures each argument as an expression.

---

## ⚙️ 2. Procedural Macros

Procedural macros are more **powerful** and **complex** — they receive Rust code as input (in the form of a *TokenStream*), manipulate it, and return new code.

They require a **separate crate** of type `proc-macro`.

---

### 🧩 Types of Procedural Macros

| Type                     | Used For                   | Example                     |
| ------------------------ | -------------------------- | --------------------------- |
| **Derive macros**        | Auto-implementing traits   | `#[derive(Debug, Clone)]`   |
| **Attribute macros**     | Custom attributes on items | `#[route(GET, "/home")]`    |
| **Function-like macros** | Function-call syntax       | `sql!(SELECT * FROM users)` |

---

### ✅ Example: Derive Macro for `HelloMacro`

**Crate 1: Define the procedural macro (`hello_macro_derive`)**

```rust
// Cargo.toml
// [lib]
// proc-macro = true

use proc_macro::TokenStream;
use quote::quote;
use syn;

#[proc_macro_derive(HelloMacro)]
pub fn hello_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_hello_macro(&ast)
}

fn impl_hello_macro(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let gen = quote! {
        impl HelloMacro for #name {
            fn hello_macro() {
                println!("Hello, Macro! My name is {}!", stringify!(#name));
            }
        }
    };
    gen.into()
}
```

**Crate 2: Use the macro**

```rust
use hello_macro_derive::HelloMacro;

trait HelloMacro {
    fn hello_macro();
}

#[derive(HelloMacro)]
struct Pancakes;

fn main() {
    Pancakes::hello_macro();
}
```

🧩 Output:

```
Hello, Macro! My name is Pancakes!
```

---

### 🧰 Example: Function-like Procedural Macro

```rust
// In the proc-macro crate
use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

#[proc_macro]
pub fn make_answer(_item: TokenStream) -> TokenStream {
    let expanded = quote! {
        fn answer() -> i32 {
            42
        }
    };
    expanded.into()
}
```

**Usage:**

```rust
use my_macros::make_answer;

make_answer!();

fn main() {
    println!("The answer is {}", answer());
}
```

---

## 🧮 Summary

| Macro Type                 | Syntax               | Example                                | Purpose                 |
| -------------------------- | -------------------- | -------------------------------------- | ----------------------- |
| Declarative                | `macro_rules!`       | `println!`, `vec!`                     | Pattern-based expansion |
| Procedural (derive)        | `#[derive(MyTrait)]` | Auto trait implementation              |                         |
| Procedural (attribute)     | `#[my_attribute]`    | Modify items using attributes          |                         |
| Procedural (function-like) | `my_macro!(...)`     | Code-generation in function-call style |                         |

---

## 🧠 Pro Tips

* Use **`quote!`** to generate Rust syntax as code inside macros.
* Use **`syn`** to parse input tokens into Rust syntax trees.
* Use **procedural macros** when logic exceeds what pattern-based macros can express.
* Keep macros in **separate crates** to maintain modularity and reduce compile-time complexity.

---

