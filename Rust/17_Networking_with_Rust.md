Let’s go step-by-step through how to use **`reqwest`**, **`tokio`**, and **`hyper`** — the three core crates for Rust networking.

---

## 🚀 Overview

| Crate         | Purpose                                           | Layer             |
| ------------- | ------------------------------------------------- | ----------------- |
| **`reqwest`** | High-level HTTP client (like Python’s `requests`) | Application layer |
| **`hyper`**   | Low-level HTTP client/server library              | Transport layer   |
| **`tokio`**   | Asynchronous runtime (tasks, sockets, timers)     | Execution layer   |

So:
➡️ `tokio` runs your async tasks
➡️ `hyper` handles low-level HTTP requests/responses
➡️ `reqwest` gives you a convenient API built *on top of* `hyper`.

---

## 🧩 1. Using `reqwest` — High-Level HTTP Client

**Ideal for:** REST APIs, microservices, web scraping, etc.
Built on top of **`hyper`** and **`tokio`**, it supports async I/O, JSON, redirects, and more.

### ✨ Example: Simple GET Request

```rust
use reqwest;
use tokio;

#[tokio::main]
async fn main() -> Result<(), reqwest::Error> {
    let response = reqwest::get("https://api.github.com").await?;
    let body = response.text().await?;

    println!("Response: {}", body);
    Ok(())
}
```

✅ Automatically uses `tokio`’s async runtime.
✅ The `await` syntax ensures non-blocking I/O.

---

### ✨ Example: Sending a JSON POST Request

```rust
use reqwest::Client;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), reqwest::Error> {
    let client = Client::new();

    let res = client.post("https://httpbin.org/post")
        .json(&json!({
            "username": "sandeep",
            "role": "developer"
        }))
        .send()
        .await?;

    println!("Status: {}", res.status());
    println!("Body: {}", res.text().await?);
    Ok(())
}
```

💡 **Key features:**

* Automatically serializes your JSON body.
* Handles async I/O efficiently.
* You can also add headers, auth, or query params easily.

---

### ✨ Example: Custom Headers & Timeout

```rust
use reqwest::{Client, header};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), reqwest::Error> {
    let mut headers = header::HeaderMap::new();
    headers.insert("User-Agent", "RustReqwest/1.0".parse().unwrap());

    let client = Client::builder()
        .default_headers(headers)
        .timeout(Duration::from_secs(5))
        .build()?;

    let res = client.get("https://example.com").send().await?;
    println!("Status: {}", res.status());
    Ok(())
}
```

---

## ⚙️ 2. Using `tokio` — Async Runtime & TCP Networking

`tokio` is the foundation of async Rust.
It powers many frameworks — including `reqwest`, `hyper`, `axum`, and `warp`.

### ✨ Example: Building a Simple TCP Server

```rust
use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    println!("Server listening on port 8080...");

    loop {
        let (mut socket, addr) = listener.accept().await?;
        println!("New connection from {}", addr);

        tokio::spawn(async move {
            let mut buffer = [0; 1024];
            if let Ok(n) = socket.read(&mut buffer).await {
                if n > 0 {
                    socket.write_all(&buffer[0..n]).await.unwrap(); // Echo back
                }
            }
        });
    }
}
```

✅ Handles thousands of concurrent connections on a few threads.
✅ Each client runs in a separate async task — not a separate OS thread.

---

### ✨ Example: TCP Client

```rust
use tokio::net::TcpStream;
use tokio::io::{AsyncWriteExt, AsyncReadExt};

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let mut stream = TcpStream::connect("127.0.0.1:8080").await?;
    stream.write_all(b"Hello from client!").await?;

    let mut buf = [0; 1024];
    let n = stream.read(&mut buf).await?;
    println!("Received: {}", String::from_utf8_lossy(&buf[..n]));
    Ok(())
}
```

---

## 🌐 3. Using `hyper` — Low-Level HTTP Framework

**`hyper`** is a fast, low-level HTTP library built directly on async Rust.
It’s used *under the hood* by `reqwest`, `axum`, `warp`, etc.

### ✨ Example: Building an HTTP Server

```rust
use hyper::{Body, Request, Response, Server};
use hyper::service::{make_service_fn, service_fn};

async fn handle(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    println!("Request: {:?}", req);
    Ok(Response::new(Body::from("Hello from Hyper server!")))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = ([127, 0, 0, 1], 3000).into();
    let make_svc = make_service_fn(|_conn| async { Ok::<_, hyper::Error>(service_fn(handle)) });

    let server = Server::bind(&addr).serve(make_svc);

    println!("Server running on http://{}", addr);
    server.await?;
    Ok(())
}
```

✅ Fast, async, and fully customizable HTTP server.
✅ Each connection is handled by `tokio` tasks.
✅ You can extend it with middlewares, routes, etc.

---

### ✨ Example: Hyper Client

```rust
use hyper::{Client, Uri};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = Client::new();
    let uri: Uri = "http://httpbin.org/get".parse()?;
    let res = client.get(uri).await?;
    println!("Response: {}", res.status());
    Ok(())
}
```

---

## 🧠 4. Combining Everything — A Mini REST API Server

Here’s how all three play together in a real-world setup.

```rust
use serde::{Deserialize, Serialize};
use hyper::{Body, Request, Response, Server, Method};
use hyper::service::{make_service_fn, service_fn};

#[derive(Serialize, Deserialize)]
struct Message {
    text: String,
}

async fn handle(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/") => {
            Ok(Response::new(Body::from("Welcome to Rust API!")))
        }
        (&Method::POST, "/echo") => {
            let whole_body = hyper::body::to_bytes(req.into_body()).await?;
            let msg: Message = serde_json::from_slice(&whole_body).unwrap();
            let reply = serde_json::to_string(&msg).unwrap();
            Ok(Response::new(Body::from(reply)))
        }
        _ => Ok(Response::builder().status(404).body(Body::from("Not Found")).unwrap()),
    }
}

#[tokio::main]
async fn main() {
    let addr = ([127, 0, 0, 1], 8080).into();
    let make_svc = make_service_fn(|_conn| async { Ok::<_, hyper::Error>(service_fn(handle)) });

    println!("API server running at http://localhost:8080");
    Server::bind(&addr).serve(make_svc).await.unwrap();
}
```

✅ Supports GET and POST
✅ Uses async JSON parsing
✅ Runs on top of `tokio` for concurrency

---

## 🧩 5. Common Pitfalls

| Pitfall                       | Cause                                   | Fix                                    |
| ----------------------------- | --------------------------------------- | -------------------------------------- |
| ❌ Blocking I/O inside async   | Using `std::fs` or `std::net`           | Use `tokio::fs` / `tokio::net`         |
| ❌ Forgetting `.await`         | Futures not executed                    | Always `.await` async calls            |
| ❌ Mixing sync and async code  | Deadlocks or performance issues         | Keep async boundaries clear            |
| ❌ Missing runtime             | Async function without `#[tokio::main]` | Use `#[tokio::main]` or `tokio::spawn` |
| ❌ Too many clones of `Client` | Overhead                                | Reuse a single `reqwest::Client`       |

---

## 🧠 Summary

| Crate            | Role                          | Common Use                     |
| ---------------- | ----------------------------- | ------------------------------ |
| **Tokio**        | Async runtime, tasks, sockets | Foundation for concurrency     |
| **Reqwest**      | High-level HTTP client        | REST calls, APIs, web scraping |
| **Hyper**        | Low-level HTTP library        | Custom HTTP servers/clients    |
| **Serde + JSON** | Data serialization            | API request/response           |

---

✅ **Best Practice Flow:**

1. Use **Tokio** to run async tasks.
2. Use **Reqwest** for RESTful API clients.
3. Use **Hyper** (or frameworks like Axum/Warp) for building servers.
4. Combine with **Serde** for JSON handling.

---
