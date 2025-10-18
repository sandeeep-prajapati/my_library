Absolutely, Sandeep! 🦀 `actix-web` is one of Rust’s **most popular web frameworks**, known for **high performance**, **type safety**, and **async support** via `tokio`. Let’s go step by step to see how to build high-performance web servers.

---

## 1️⃣ Add Dependencies

In `Cargo.toml`:

```toml
[dependencies]
actix-web = "4"
tokio = { version = "1", features = ["full"] }  # async runtime
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

* `actix-web` → Web framework
* `tokio` → Async runtime
* `serde` → Serialization/deserialization
* `serde_json` → JSON support

---

## 2️⃣ Basic HTTP Server

```rust
use actix_web::{get, App, HttpServer, HttpResponse, Responder};

#[get("/")]
async fn hello() -> impl Responder {
    HttpResponse::Ok().body("Hello, Actix-web!")
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().service(hello))
        .bind("127.0.0.1:8080")?
        .run()
        .await
}
```

✅ Run the server:

```bash
cargo run
```

* Access: `http://127.0.0.1:8080/`
* Async functions (`async fn`) allow **non-blocking I/O**, making the server scalable.

---

## 3️⃣ Handling JSON Requests and Responses

```rust
use actix_web::{post, web, App, HttpServer, HttpResponse, Responder};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    name: String,
}

#[derive(Serialize)]
struct Output {
    message: String,
}

#[post("/greet")]
async fn greet(info: web::Json<Input>) -> impl Responder {
    let response = Output {
        message: format!("Hello, {}!", info.name),
    };
    HttpResponse::Ok().json(response)
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().service(greet))
        .bind("127.0.0.1:8080")?
        .run()
        .await
}
```

* `web::Json<T>` automatically **deserializes JSON requests**.
* `HttpResponse::Ok().json(...)` serializes the Rust struct into JSON.

---

## 4️⃣ Query Parameters and Path Variables

```rust
use actix_web::{get, web, App, HttpServer, Responder};

#[get("/user/{id}")]
async fn user(path: web::Path<u32>, query: web::Query<std::collections::HashMap<String, String>>) -> impl Responder {
    let user_id = path.into_inner();
    let name = query.get("name").unwrap_or(&"Guest".to_string());
    format!("User ID: {}, Name: {}", user_id, name)
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().service(user))
        .bind("127.0.0.1:8080")?
        .run()
        .await
}
```

* `web::Path<T>` extracts variables from the URL path.
* `web::Query<T>` extracts query parameters.

---

## 5️⃣ Middlewares

Middlewares allow logging, authentication, or request modification:

```rust
use actix_web::{App, HttpServer, middleware, HttpResponse, get};

#[get("/")]
async fn index() -> HttpResponse {
    HttpResponse::Ok().body("Hello with middleware!")
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .wrap(middleware::Logger::default()) // logs requests
            .service(index)
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

* Logging middleware automatically prints **request info**.
* Other middlewares: authentication, compression, CORS, rate limiting.

---

## 6️⃣ Shared State Across Handlers

```rust
use actix_web::{web, App, HttpServer, Responder, HttpResponse};
use std::sync::Mutex;

struct AppState {
    counter: Mutex<i32>,
}

async fn increment_counter(data: web::Data<AppState>) -> impl Responder {
    let mut counter = data.counter.lock().unwrap();
    *counter += 1;
    HttpResponse::Ok().body(format!("Counter: {}", counter))
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let state = web::Data::new(AppState {
        counter: Mutex::new(0),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .route("/increment", web::get().to(increment_counter))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

* `web::Data<T>` → Thread-safe shared state
* Mutex ensures **safe concurrent access**.

---

## 7️⃣ Async Database Integration Example

```rust
use actix_web::{web, App, HttpServer, Responder, HttpResponse};
use sqlx::postgres::PgPoolOptions;

async fn get_users(pool: web::Data<sqlx::PgPool>) -> impl Responder {
    let rows = sqlx::query!("SELECT id, name FROM users")
        .fetch_all(pool.get_ref())
        .await
        .unwrap();
    HttpResponse::Ok().json(rows)
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect("postgres://user:password@localhost/dbname")
        .await
        .unwrap();

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(pool.clone()))
            .route("/users", web::get().to(get_users))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

* `sqlx` is **async database driver**.
* `actix-web` handles async tasks efficiently using **tokio runtime**.

---

## 8️⃣ Performance Tips for Actix-Web

1. **Use async handlers**: Avoid blocking calls in async handlers. Offload heavy CPU tasks using `actix_web::web::block`.
2. **Shared state**: Use `Arc<Mutex>` or `DashMap` for concurrent access.
3. **Compression**: Enable middleware for `gzip`/`brotli`.
4. **Logging**: Use `tracing` for structured, async-friendly logs.
5. **Benchmark**: Use `wrk` or `hey` to test throughput.
6. **Release mode**: Always build in release for performance:

   ```bash
   cargo run --release
   ```

---

## ✅ Summary Table

| Feature           | Actix-Web Concept                                             |
| ----------------- | ------------------------------------------------------------- |
| HTTP routing      | `App::route`, `#[get]`, `#[post]`                             |
| JSON handling     | `web::Json<T>`                                                |
| Query/path params | `web::Query<T>`, `web::Path<T>`                               |
| Middleware        | `App::wrap()`                                                 |
| Shared state      | `web::Data<T>`                                                |
| Async DB          | Use `sqlx`, `tokio-postgres` or `mongodb` with async handlers |
| Logging           | `middleware::Logger`, `tracing`                               |
| Parallelism       | Offload CPU tasks with `web::block`                           |

---
