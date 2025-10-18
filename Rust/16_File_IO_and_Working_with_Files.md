
---

## ⚙️ 1. Modules for File I/O in Rust

Rust primarily uses two modules for file handling:

| Module    | Purpose                                                     |
| --------- | ----------------------------------------------------------- |
| `std::fs` | File system operations (open, create, read, write, delete)  |
| `std::io` | Reading/writing streams, buffering, and input/output traits |

---

## 🧱 2. Opening and Reading Files

### Example 1: Reading the Entire File

```rust
use std::fs;

fn main() -> std::io::Result<()> {
    let contents = fs::read_to_string("hello.txt")?;
    println!("File contents:\n{}", contents);
    Ok(())
}
```

✅ **Key points:**

* `fs::read_to_string()` reads the whole file into a `String`.
* The `?` operator propagates any `Result::Err` automatically.
* Works great for **small text files**.

---

### Example 2: Buffered Reading (Efficient for Large Files)

```rust
use std::fs::File;
use std::io::{self, BufRead, BufReader};

fn main() -> io::Result<()> {
    let file = File::open("data.txt")?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        println!("{}", line?);
    }

    Ok(())
}
```

🧠 **Why use `BufReader`?**

* It reads **in chunks**, not byte-by-byte.
* Great for **line-by-line processing** and **large files**.

---

## ✍️ 3. Writing to Files

### Example 1: Overwriting a File

```rust
use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()> {
    let mut file = File::create("output.txt")?;
    file.write_all(b"Hello, Rust file I/O!")?;
    Ok(())
}
```

✅ `File::create()`:

* Creates a new file or **truncates (overwrites)** if it already exists.

---

### Example 2: Appending Data to a File

```rust
use std::fs::OpenOptions;
use std::io::Write;

fn main() -> std::io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open("log.txt")?;

    writeln!(file, "New log entry")?;
    Ok(())
}
```

🧠 **Why `OpenOptions`?**

* Lets you **configure** how to open the file (`read`, `write`, `append`, `truncate`, etc.)
* Safer than blindly overwriting existing data.

---

## 📦 4. Checking File Existence and Metadata

```rust
use std::fs;

fn main() -> std::io::Result<()> {
    let path = "config.json";

    if fs::metadata(path).is_ok() {
        println!("File exists!");
    } else {
        println!("File not found!");
    }

    Ok(())
}
```

You can also access metadata such as **file size**, **permissions**, and **modification time**:

```rust
let metadata = fs::metadata("config.json")?;
println!("Size: {} bytes", metadata.len());
```

---

## 🧮 5. Deleting, Copying, and Renaming Files

```rust
use std::fs;

fn main() -> std::io::Result<()> {
    fs::copy("input.txt", "backup.txt")?;
    fs::rename("old.txt", "new.txt")?;
    fs::remove_file("temp.txt")?;
    Ok(())
}
```

---

## ⚡ 6. Async File I/O (Using Tokio)

For **non-blocking file operations**, especially in servers, use `tokio::fs`.

```rust
use tokio::fs;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let contents = fs::read_to_string("hello.txt").await?;
    println!("{}", contents);
    Ok(())
}
```

🧠 This is **asynchronous** — your app can handle other tasks while waiting for file I/O.

---

## ⚠️ 7. Common Pitfalls in Rust File I/O

| Pitfall                           | Description                          | Fix / Best Practice                                |
| --------------------------------- | ------------------------------------ | -------------------------------------------------- |
| ❌ Unwrapped `Result`              | Crashes on error                     | Always use `?` or handle errors gracefully         |
| ❌ Reading large files into memory | Causes OOM for big files             | Use `BufReader` to stream                          |
| ❌ Ignoring UTF-8 issues           | `read_to_string` expects valid UTF-8 | Use `fs::read` for raw bytes if needed             |
| ❌ Concurrent file access          | Multiple writes corrupt data         | Use `std::fs::OpenOptions` or `tokio::sync::Mutex` |
| ❌ Hardcoded paths                 | Breaks portability                   | Use `std::path::Path` for safe path handling       |

---

## 🧠 8. Best Practices

✅ **Use buffered I/O** (`BufReader` / `BufWriter`) for performance.
✅ **Always propagate or handle errors** with `?` or pattern matching.
✅ **Prefer `OpenOptions`** for controlled file access.
✅ **Use `Path` and `PathBuf`** for cross-platform path handling.
✅ **Avoid blocking I/O** in async contexts — use `tokio::fs` or `async_std::fs`.
✅ **Use logs and error messages** to track I/O failures (permissions, missing files, etc.).

---

## 💡 Bonus: JSON Read/Write Example (Serde + File I/O)

```rust
use serde::{Serialize, Deserialize};
use std::fs;

#[derive(Serialize, Deserialize)]
struct Config {
    username: String,
    age: u8,
}

fn main() -> std::io::Result<()> {
    // Write JSON
    let config = Config { username: "Sandeep".to_string(), age: 22 };
    fs::write("config.json", serde_json::to_string_pretty(&config).unwrap())?;

    // Read JSON
    let data = fs::read_to_string("config.json")?;
    let loaded: Config = serde_json::from_str(&data).unwrap();

    println!("Hello, {} ({} years old)", loaded.username, loaded.age);
    Ok(())
}
```

✅ This is how you can safely persist structured data.

---

## 🧩 Summary Table

| Operation         | Function                          | Example                        |
| ----------------- | --------------------------------- | ------------------------------ |
| Read file (small) | `fs::read_to_string`              | `fs::read_to_string("a.txt")?` |
| Read large file   | `BufReader::new(File::open(...))` | Line-by-line                   |
| Write file        | `File::create` + `write_all`      | Overwrite                      |
| Append file       | `OpenOptions::new().append(true)` | Add data                       |
| Delete file       | `fs::remove_file`                 | Delete file                    |
| Copy file         | `fs::copy`                        | Backup                         |
| Async I/O         | `tokio::fs`                       | Non-blocking operations        |

---
