
---

# ✅ **Go’s Memory Model (Simple + In-depth)**

The **Go Memory Model** defines **how goroutines read and write shared variables**, and **what guarantees** the language provides to avoid data races.

It answers one key question:

> **“If two goroutines run concurrently, when is a write in one goroutine visible to another?”**

Go guarantees visibility of writes **only when** there is a proper synchronization event.

---

# 🔥 **Key Points of Go’s Memory Model**

### **1. No Data Race Guarantee**

Go says:

> **If your program has a data race, behaviour is undefined.**

A **data race** happens when:

* Two goroutines access the same variable,
* At least one is a write,
* No synchronization (mutex, channel, atomic) is used.

With a data race, Go does **not** guarantee correct, ordered, or predictable results.

---

### **2. Synchronization Ensures Visibility**

Memory becomes consistent only after a **synchronization event** like:

#### ✔ Channel send/receive

* Data written before sending on a channel
  → **is guaranteed visible** to the receiver.

#### ✔ Mutex Lock/Unlock

* `Unlock()` happens-before a corresponding `Lock()`
  → guaranteeing visibility.

#### ✔ WaitGroup Done/Wait

* `Done()` happens-before `Wait()`

#### ✔ Atomic operations

* Provide both atomicity and happens-before guarantees.

---

### **3. Happens-Before Relationship**

Go uses the concept **happens-before** to define ordering.

Example:

```go
x = 10
done <- true
```

In the other goroutine:

```go
<-done
print(x)
```

Because of the channel, the value `10` is guaranteed.

Without the channel → **undefined behaviour**.

---

### **4. Go Minimizes Locks by Design**

The model encourages:

* Channels
* Goroutines
* Immutable data
* Message passing over shared memory

This is why Go avoids complicated memory rules found in Java/C++.

---

# ⚔️ Differences vs Python and Java

---

# 🐍 **Go vs Python Memory Model**

Python is **NOT** truly concurrent due to the **GIL (Global Interpreter Lock)**.

### ✔ Python (CPython):

* Only one thread executes Python bytecode at a time.
* No true parallel execution of threads on multiple cores.
* Memory consistency issues are rare because GIL serializes operations.

### ❌ Go:

* **No GIL**
* True parallelism across threads and CPU cores.
* Developer must avoid data races manually.
* Concurrency bugs more likely if not careful.

**Bottom line:**

> In Python, GIL avoids data races but kills parallelism.
> In Go, there is real parallelism, so YOU must guarantee correct memory sync.

---

# ☕ **Go vs Java Memory Model**

Java has a **complex and strict** memory model:

### ✔ Java memory model guarantees:

* Volatile variables prevent caching.
* Atomic operations provided (`AtomicInteger` etc.)
* Locks guarantee ordering.
* Happens-before is detailed and strict.
* Reordering rules are defined in the JLS (Java Language Specification).

### ✔ Go is simpler:

* No `volatile` keyword.
* Fewer guarantees.
* Only sync primitives: channels, mutexes, atomics.
* Programmers are expected to avoid shared mutable state.

### ✔ Garbage Collector Differences:

| Feature         | Go                     | Java                        |
| --------------- | ---------------------- | --------------------------- |
| GC              | Concurrent, low pause  | Very advanced, generational |
| Memory Safety   | Fully safe             | Fully safe                  |
| Object model    | Simple, no inheritance | Complex OOP layout          |
| Escape analysis | Aggressive             | Aggressive                  |

### Summary:

> Java's memory model is more strict and layered.
> Go keeps it simple → “If you use synchronization, you're safe; if not, all bets are off.”

---

# 🎯 Final Summary (Interview-Ready)

### **Go Memory Model in One Line:**

> Go guarantees visibility of memory writes only through proper synchronization primitives like channels, mutexes, atomics, and WaitGroups. Without these, behaviour is undefined.

### **Main Differences:**

| Aspect       | Go                      | Python                   | Java                          |
| ------------ | ----------------------- | ------------------------ | ----------------------------- |
| Concurrency  | True parallelism        | GIL prevents parallelism | True parallelism              |
| Data Races   | Possible → undefined    | Rare due to GIL          | Avoided via strict rules      |
| Memory Model | Simple, sync-based      | GIL-serialized           | Complex JMM                   |
| Sync Tools   | Channels, mutex, atomic | Locks, queues            | Locks, volatile, atomics, CAS |
| GC           | Fast, concurrent        | Reference counting + GC  | Generational GC               |

---

---

# ✅ **What Are Structs in Go?**

`struct` is a **custom data type** that lets you group related fields.

```go
type User struct {
    Name  string
    Email string
    Age   int
}
```

Structs are the foundation of Go’s data modeling.

---

# ✅ **What Are Interfaces in Go?**

An **interface** defines a set of method signatures.
Any type that implements those methods **implicitly** satisfies the interface.

```go
type Notifier interface {
    Notify(msg string) error
}
```

No “implements” keyword → fully **implicit**.
This makes Go interfaces extremely flexible.

---

# 🚀 Now let's learn with **5 real-world examples**.

---

# -----------------------------------------

# **🌍 Example 1: User Authentication System**

### **Struct → User**

### **Interface → AuthService**

### **Use Case:** Login, signup, etc.

```go
type User struct {
    ID       int
    Username string
    Password string
}

// Interface for authentication actions
type AuthService interface {
    Login(username, password string) bool
    Signup(u User) error
}

type LocalAuth struct{}

func (LocalAuth) Login(username, password string) bool {
    return username == "admin" && password == "1234"
}

func (LocalAuth) Signup(u User) error {
    println("User signed up:", u.Username)
    return nil
}
```

### Why is this real-world?

* `AuthService` allows swapping e.g. **LocalAuth**, **FirebaseAuth**, **OAuth service**.

---

# -----------------------------------------

# **📦 Example 2: Payment Gateway Integration**

### **Structs → Razorpay, Paytm**

### **Interface → PaymentGateway**

```go
type PaymentGateway interface {
    Pay(amount float64) string
}

type Razorpay struct{}

func (Razorpay) Pay(amount float64) string {
    return "Paid ₹" + fmt.Sprint(amount) + " using Razorpay"
}

type Paytm struct{}

func (Paytm) Pay(amount float64) string {
    return "Paid ₹" + fmt.Sprint(amount) + " using Paytm"
}
```

### Use Case

You can switch payment provider **without changing business logic**:

```go
func Checkout(pg PaymentGateway) {
    println(pg.Pay(999.99))
}

Checkout(Razorpay{}) // Works
Checkout(Paytm{})    // Also works
```

---

# -----------------------------------------

# **📧 Example 3: Email Sender**

### **Interface → Mailer**

### **Structs → GmailService, SMTPService**

```go
type Mailer interface {
    Send(to, subject, body string) error
}

type GmailService struct{}

func (GmailService) Send(to, subject, body string) error {
    println("Sent via Gmail to:", to)
    return nil
}

type SMTPService struct{}

func (SMTPService) Send(to, subject, body string) error {
    println("Sent via SMTP to:", to)
    return nil
}
```

### Real-world benefit:

Swap Gmail ↔ SMTP ↔ SendGrid without changing your API server.

---

# -----------------------------------------

# **🚗 Example 4: Vehicle Speed Calculation**

### **Structs → Car, Bike**

### **Interface → Vehicle**

```go
type Vehicle interface {
    Speed() int
}

type Car struct {
    HorsePower int
}

func (c Car) Speed() int {
    return c.HorsePower * 2
}

type Bike struct {
    CC int
}

func (b Bike) Speed() int {
    return b.CC * 1
}
```

### Why real-world?

This pattern is used in:

* Game engines
* Fleet management systems
* Simulations

```go
func PrintSpeed(v Vehicle) {
    println("Speed:", v.Speed())
}

PrintSpeed(Car{HorsePower: 200})
PrintSpeed(Bike{CC: 150})
```

---

# -----------------------------------------

# **🌐 Example 5: Web Framework – Router Handlers**

### **Interface → Handler**

Exactly like **net/http**.

```go
type Handler interface {
    Serve()
}

type HomeHandler struct{}
func (HomeHandler) Serve() { println("Welcome to Home Page") }

type AboutHandler struct{}
func (AboutHandler) Serve() { println("About Page") }
```

Router function:

```go
func Route(h Handler) {
    h.Serve()
}

Route(HomeHandler{})
Route(AboutHandler{})
```

### Real-world use:

This is how **http.Handler** works in Go’s standard library.

---

# 🎯 **Quick Summary**

## 📌 **Structs**

* Group related fields.
* Represent data models.
* Can have methods.

## 📌 **Interfaces**

* Describe *behavior*, not data.
* Implicit implementation (no keywords).
* Enable dependency injection.
* Make code flexible, replaceable, testable.

---

Here is a **complete practice exercise set** for **arrays, slices, maps, and pointers in Go**, along with **answers**.
These are **interview-style**, practical, and gradually increasing in difficulty.

---

# ✅ **EXERCISE SET — Arrays, Slices, Maps & Pointers**

You get **12 exercises** (3 for each topic), plus full solutions.

---

# 🔵 **PART 1: Arrays (3 exercises)**

### **Q1. Create an array of 5 integers and find the sum of all elements.**

---

### **Q2. Given an array of 10 integers, count how many are even.**

---

### **Q3. Reverse an array in-place (no extra array).**

---

# 🟣 **PART 2: Slices (3 exercises)**

### **Q4. Create a slice of integers, append values, and remove the element at index 2.**

---

### **Q5. Given a slice of strings, return a new slice containing only strings longer than 3 characters.**

---

### **Q6. Write a function that takes a slice and doubles each element (modify original slice).**

---

# 🟡 **PART 3: Maps (3 exercises)**

### **Q7. Create a map of student names → marks. Add 5 students, delete 1, and print topper’s name.**

---

### **Q8. Count the frequency of each character in a string using a map.**

---

### **Q9. You have a map[int]string. Write a function that swaps keys and values to map[string]int.**

(Values are guaranteed unique.)

---

# 🔴 **PART 4: Pointers (3 exercises)**

### **Q10. Write a function that takes an integer pointer and increments the value by 10.**

---

### **Q11. Swap two integers using pointers.**

---

### **Q12. Create a struct User{Name, Age}. Write a function that updates the age using a pointer to struct.**

---

---

# 🟢 **ANSWERS (Full Code Solutions)**

Each answer is short, clear, and runnable.

---

# ✅ **A1. Sum of array**

```go
func main() {
    arr := [5]int{10, 20, 30, 40, 50}
    sum := 0

    for _, v := range arr {
        sum += v
    }

    println("Sum =", sum)
}
```

---

# ✅ **A2. Count even numbers**

```go
func main() {
    arr := [10]int{1,2,3,4,5,6,7,8,9,10}
    count := 0

    for _, v := range arr {
        if v%2 == 0 {
            count++
        }
    }

    println("Even count =", count)
}
```

---

# ✅ **A3. Reverse an array in-place**

```go
func main() {
    arr := [5]int{1,2,3,4,5}

    for i, j := 0, len(arr)-1; i < j; i, j = i+1, j-1 {
        arr[i], arr[j] = arr[j], arr[i]
    }

    fmt.Println(arr)
}
```

---

# 🟣 **Slices**

---

# ✅ **A4. Remove index 2**

```go
func main() {
    s := []int{10, 20, 30, 40, 50}

    index := 2
    s = append(s[:index], s[index+1:]...)

    fmt.Println(s) // [10 20 40 50]
}
```

---

# ✅ **A5. Filter long strings**

```go
func filterLong(strs []string) []string {
    result := []string{}

    for _, s := range strs {
        if len(s) > 3 {
            result = append(result, s)
        }
    }
    return result
}

func main() {
    fmt.Println(filterLong([]string{"go", "java", "js", "rust", "ai"}))
}
```

---

# ✅ **A6. Double slice values (modify original)**

```go
func doubleSlice(nums []int) {
    for i := range nums {
        nums[i] *= 2
    }
}

func main() {
    arr := []int{1,2,3}
    doubleSlice(arr)
    fmt.Println(arr) // [2 4 6]
}
```

---

# 🟡 **Maps**

---

# ✅ **A7. Student marks & topper**

```go
func main() {
    students := map[string]int{
        "Aman": 90,
        "Riya": 85,
        "Sahil": 92,
        "Tina": 88,
        "John": 75,
    }

    delete(students, "John")

    topper := ""
    topMarks := -1

    for name, marks := range students {
        if marks > topMarks {
            topMarks = marks
            topper = name
        }
    }

    println("Topper is:", topper)
}
```

---

# ✅ **A8. Character frequency**

```go
func main() {
    text := "hello world"
    freq := make(map[rune]int)

    for _, ch := range text {
        freq[ch]++
    }

    fmt.Println(freq)
}
```

---

# ✅ **A9. Swap keys & values**

```go
func swapMap(m map[int]string) map[string]int {
    result := make(map[string]int)

    for k, v := range m {
        result[v] = k
    }
    return result
}

func main() {
    m := map[int]string{1:"apple", 2:"banana"}
    fmt.Println(swapMap(m))
}
```

---

# 🔴 **Pointers**

---

# ✅ **A10. Increment by pointer**

```go
func increment(x *int) {
    *x += 10
}

func main() {
    a := 5
    increment(&a)
    fmt.Println(a) // 15
}
```

---

# ✅ **A11. Swap using pointers**

```go
func swap(a, b *int) {
    *a, *b = *b, *a
}

func main() {
    x, y := 10, 20
    swap(&x, &y)
    fmt.Println(x, y) // 20 10
}
```

---

# ✅ **A12. Update struct using pointer**

```go
type User struct {
    Name string
    Age  int
}

func updateAge(u *User, newAge int) {
    u.Age = newAge
}

func main() {
    u := User{"Sandeep", 21}
    updateAge(&u, 25)
    fmt.Println(u) // {Sandeep 25}
}
```

---
Here is a **clear, interview-quality explanation** of the **difference between methods and functions in Go**, with clean and simple code examples.

---

# ✅ **Difference Between Methods and Functions in Go**

In Go, **every method is a function**, but not every function is a method.

The difference is:

# 🔵 **1. Functions**

A **function** belongs to *no type*.
It is independent and can be called directly.

```go
func add(a int, b int) int {
    return a + b
}
```

You call it like:

```go
result := add(10, 20)
```

---

# 🔴 **2. Methods**

A **method** is a function that **belongs to a type** (struct or alias).
It has a **receiver**, which is like a lightweight version of “this” or “self”.

Syntax:

```go
func (receiver Type) methodName() {
    ...
}
```

Example:

```go
type User struct {
    Name string
    Age  int
}

func (u User) Greet() {
    fmt.Println("Hello,", u.Name)
}
```

You call it using a value:

```go
user := User{"Sandeep", 21}
user.Greet()
```

---

# 🔥 **Core Difference Table**

| Feature            | Function         | Method                      |
| ------------------ | ---------------- | --------------------------- |
| Receiver?          | ❌ No             | ✔ Yes                       |
| Belongs to a type? | ❌ No             | ✔ Yes                       |
| Syntax             | `func name(...)` | `func (t Type) name(...)`   |
| Called on          | Directly         | Type value or pointer       |
| Use case           | Utility logic    | Behavior of structs/objects |

---

# 🔵 **Detailed Real Example (Simple)**

## **Function example**

```go
func area(width, height int) int {
    return width * height
}
```

## **Method example**

```go
type Rectangle struct {
    Width  int
    Height int
}

func (r Rectangle) Area() int {
    return r.Width * r.Height
}
```

### Function call:

```go
println(area(10, 20))
```

### Method call:

```go
rect := Rectangle{10, 20}
println(rect.Area())
```

---

# 🔴 **Value Receiver vs Pointer Receiver (Important!)**

Go allows both:

### ✔ **Value receiver**

Method gets a copy of the struct.

```go
func (u User) Greet() {
    fmt.Println("Hi,", u.Name)
}
```

### ✔ **Pointer receiver**

Method can **modify** the original struct.

```go
func (u *User) UpdateAge(newAge int) {
    u.Age = newAge
}
```

---

# 🟢 **Final Example: Methods vs Functions Together**

```go
type BankAccount struct {
    Balance float64
}

// Method: modifies the struct
func (a *BankAccount) Deposit(amount float64) {
    a.Balance += amount
}

// Function: utility not tied to a type
func convertToUSD(inr float64) float64 {
    return inr / 83.0
}

func main() {
    acc := BankAccount{Balance: 1000}
    acc.Deposit(500)

    fmt.Println("Balance:", acc.Balance)
    fmt.Println("In USD:", convertToUSD(acc.Balance))
}
```

---

# 🎯 **Summary (Interview Ready)**

### ✔ A **function** is independent logic.

### ✔ A **method** is a function with a **receiver**, meaning it is tied to a type.

### ✔ Methods define **behavior** of structs.

### ✔ Functions define **general utility logic**.

---
Here are **5 advanced, interview-ready, interface-based design challenges in Go**, each followed by **optimal solutions**.
These challenges teach **polymorphism, dependency injection, interface composition, mocking, and clean architecture**.

---

# ⭐ **Challenge 1: Logging System with Multiple Backends**

**Goal:**
Design a logging system that supports **ConsoleLogger**, **FileLogger**, and **NoOpLogger** (used for testing).

### **Requirements**

* Common interface: `Logger` with method `Log(msg string)`
* Application must accept any logger
* Demonstrate switching between loggers

---

## ✅ **Solution**

```go
package main

import (
    "fmt"
    "os"
)

type Logger interface {
    Log(msg string)
}

type ConsoleLogger struct{}

func (ConsoleLogger) Log(msg string) {
    fmt.Println("[Console]", msg)
}

type FileLogger struct {
    file *os.File
}

func NewFileLogger(filepath string) *FileLogger {
    f, _ := os.Create(filepath)
    return &FileLogger{file: f}
}

func (l *FileLogger) Log(msg string) {
    l.file.WriteString(msg + "\n")
}

type NoOpLogger struct{}

func (NoOpLogger) Log(msg string) {} // does nothing

func Process(logger Logger) {
    logger.Log("Task started")
}

func main() {
    Process(ConsoleLogger{})
    Process(NewFileLogger("log.txt"))
    Process(NoOpLogger{})
}
```

### **Concepts learned**

* Interface polymorphism
* Swappable dependencies
* NoOp pattern for testing

---

# ⭐ **Challenge 2: Payment Gateway Strategy**

**Goal:**
Implement a **strategy pattern** using interfaces:

* `PayPal`
* `Stripe`
* `CashOnDelivery`

### **Requirements**

* `PaymentGateway` interface with `Pay(amount float64) string`
* A checkout function using the interface

---

## ✅ **Solution**

```go
package main

import "fmt"

type PaymentGateway interface {
    Pay(amount float64) string
}

type PayPal struct{}
func (PayPal) Pay(amount float64) string { return "Paid via PayPal" }

type Stripe struct{}
func (Stripe) Pay(amount float64) string { return "Paid via Stripe" }

type COD struct{}
func (COD) Pay(amount float64) string { return "Cash On Delivery selected" }

func Checkout(pg PaymentGateway) {
    fmt.Println(pg.Pay(500))
}

func main() {
    Checkout(PayPal{})
    Checkout(Stripe{})
    Checkout(COD{})
}
```

### **Concepts learned**

* Strategy pattern
* Decoupling payment logic from business logic

---

# ⭐ **Challenge 3: Custom Sorter (Like sort.Interface)**

**Goal:**
Create your own sorter that works on **any collection** that satisfies a custom interface.

### **Requirements**

* Interface `Sortable` with:

  * `Len() int`
  * `Swap(i, j int)`
  * `Less(i, j int) bool`
* Implement a Bubble Sort using the interface
* Demonstrate sorting custom types

---

## ✅ **Solution**

```go
package main

import "fmt"

type Sortable interface {
    Len() int
    Swap(i, j int)
    Less(i, j int) bool
}

func BubbleSort(s Sortable) {
    n := s.Len()
    for i := 0; i < n; i++ {
        for j := 0; j < n-i-1; j++ {
            if s.Less(j+1, j) {
                s.Swap(j, j+1)
            }
        }
    }
}

// Custom Type
type Numbers []int

func (n Numbers) Len() int           { return len(n) }
func (n Numbers) Swap(i, j int)      { n[i], n[j] = n[j], n[i] }
func (n Numbers) Less(i, j int) bool { return n[i] < n[j] }

func main() {
    nums := Numbers{5, 1, 4, 2, 3}
    BubbleSort(nums)
    fmt.Println(nums)
}
```

### **Concepts learned**

* Interface-based generic algorithms
* Behavior abstraction
* How `sort.Interface` works internally

---

# ⭐ **Challenge 4: Storage Interface (Local, Memory, Cloud Mock)**

**Goal:**
Design a flexible storage system with 3 implementations:

* In-memory storage
* Local file storage
* Mock storage for unit tests

### **Requirements**

* Interface: `Storage { Save(key, value string) }`
* Implement 3 versions
* Use dependency injection

---

## ✅ **Solution**

```go
package main

import (
    "fmt"
    "os"
)

type Storage interface {
    Save(key, value string)
}

// Memory Storage
type MemoryStore map[string]string

func (m MemoryStore) Save(key, value string) {
    m[key] = value
}

// File Storage
type FileStore struct {
    file *os.File
}

func NewFileStore(p string) *FileStore {
    f, _ := os.Create(p)
    return &FileStore{file: f}
}

func (fs *FileStore) Save(key, value string) {
    fs.file.WriteString(key + "=" + value + "\n")
}

// Mock Storage for Testing
type MockStore struct{}

func (MockStore) Save(key, value string) {
    fmt.Println("Mock save:", key, value)
}

func writeUserData(s Storage) {
    s.Save("user", "sandeep")
}

func main() {
    writeUserData(MemoryStore{})
    writeUserData(NewFileStore("data.txt"))
    writeUserData(MockStore{})
}
```

### **Concepts learned**

* Dependency injection
* Real-world storage abstraction
* Test mocking patterns

---

# ⭐ **Challenge 5: Observer Pattern (Event System)**

**Goal:**
Implement an event system where multiple observers can listen for updates.

### **Requirements**

* `Listener` interface with method `Update(data string)`
* `EventBus` struct that maintains a list of listeners
* Notify all listeners when an event occurs

---

## ✅ **Solution**

```go
package main

import "fmt"

type Listener interface {
    Update(data string)
}

type EventBus struct {
    listeners []Listener
}

func (bus *EventBus) Register(l Listener) {
    bus.listeners = append(bus.listeners, l)
}

func (bus *EventBus) Broadcast(msg string) {
    for _, l := range bus.listeners {
        l.Update(msg)
    }
}

type EmailListener struct{}
func (EmailListener) Update(data string) {
    fmt.Println("Email:", data)
}

type SMSListener struct{}
func (SMSListener) Update(data string) {
    fmt.Println("SMS:", data)
}

func main() {
    bus := EventBus{}
    bus.Register(EmailListener{})
    bus.Register(SMSListener{})

    bus.Broadcast("New User Registered!")
}
```

### **Concepts learned**

* Observer pattern
* Event-driven design
* Interface-based extensibility

---

# 🎯 **Final Summary**

You learned **5 advanced interface design patterns**:

### ✔ Logging abstraction

### ✔ Strategy pattern

### ✔ Generic sorting

### ✔ Dependency injection for storages

### ✔ Observer/event-bus pattern

All of these are used in **real-world Go applications**, including backend systems, microservices, and cloud-native apps.

---

---

# ✅ **Using Function Closures in Go (with real-world examples)**

Closures in Go are functions **that capture variables from their surrounding scope**, even after that scope has ended.

In simple terms:

> **A closure is a function + the environment it remembers.**

---

# 🧩 **1. Basic Closure Example**

```go
package main

import "fmt"

func counter() func() int {
    count := 0  // captured variable

    return func() int {
        count++
        return count
    }
}

func main() {
    c := counter()
    fmt.Println(c()) // 1
    fmt.Println(c()) // 2
    fmt.Println(c()) // 3
}
```

✔ `count` lives even after `counter()` returns
✔ Each call updates the same captured variable

---

# 🧩 **2. Closure with Parameters**

```go
func multiplier(x int) func(int) int {
    return func(y int) int {
        return x * y
    }
}

func main() {
    double := multiplier(2)
    triple := multiplier(3)

    fmt.Println(double(10)) // 20
    fmt.Println(triple(10)) // 30
}
```

---

# 🔥 **Real-World Use Cases for Closures**

Below are **the 6 places closures are actually useful** in real-world Go projects.

---

# 🧠 **Use Case 1 — Creating Configurable Functions (API, ML, Tools)**

Closures are great to **create functions with preloaded settings**.

### Example: logging with prefix

```go
func makeLogger(prefix string) func(string) {
    return func(msg string) {
        fmt.Println(prefix, msg)
    }
}

func main() {
    info := makeLogger("[INFO]")
    errLog := makeLogger("[ERROR]")

    info("server started")
    errLog("db not reachable")
}
```

---

# 🛒 **Use Case 2 — Rate Limiting (Throttle API calls)**

Closures let you store state between calls.

```go
func rateLimiter(limit int) func() bool {
    count := 0
    return func() bool {
        if count < limit {
            count++
            return true
        }
        return false
    }
}
```

---

# 🎮 **Use Case 3 — Event Handlers in GUI / WebAssembly**

(Closures are widely used in JS; Go's WASM uses same pattern.)

```go
func clickHandler() func() {
    clicks := 0
    return func() {
        clicks++
        fmt.Println("Clicked:", clicks)
    }
}
```

---

# 🧪 **Use Case 4 — Mocking State in Unit Tests**

Store test-specific state inside a closure.

```go
func fakeDB() func(string) string {
    store := map[string]string{
        "101": "User A",
    }
    return func(id string) string {
        return store[id]
    }
}
```

---

# ⚙ **Use Case 5 — Functional Pipelines / Middleware**

Closures are perfect for chaining middleware (HTTP servers, gRPC, etc.)

```go
func middleware(next func(int) int) func(int) int {
    return func(x int) int {
        fmt.Println("before")
        out := next(x)
        fmt.Println("after")
        return out
    }
}
```

---

# 📊 **Use Case 6 — Simple in-memory Caching**

```go
func memoize(f func(int) int) func(int) int {
    cache := make(map[int]int)
    return func(n int) int {
        if v, ok := cache[n]; ok {
            return v
        }
        result := f(n)
        cache[n] = result
        return result
    }
}
```

---

# 🏆 **When You Should Use Closures in Real Projects**

| When                                         | Why closures help                              |
| -------------------------------------------- | ---------------------------------------------- |
| You need a function that **remembers state** | counters, caching, rate limiting               |
| You want **configurable functions**          | preloaded loggers, preconfigured services      |
| You build **middleware**                     | HTTP/gRPC chains                               |
| You want **cleaner test mocks**              | stateful fake DBs, mock counters               |
| You want to **hide data**                    | closures provide encapsulation without classes |

---

# 🎯 Summary (super short)

* **Closures = Function + Remembered Environment**
* They help create **stateful functions** without classes.
* Used in: caching, rate limiting, middleware, mocks, logging, pipelines.

---
Here is the **perfect beginner-friendly explanation of goroutines and channels**, followed by a **pro-level advanced summary** so you understand both depth **and** simplicity.

---

# 🟢 **Beginner-Friendly Explanation**

## ✅ **What is a Goroutine?**

A **goroutine** is simply a *lightweight thread* managed by Go.

Think of it like:

> “Hey Go, run this function in the background.”

You start a goroutine by putting **go** before a function:

```go
go doSomething()
```

### Key Points (Beginner Level)

* Very cheap: you can run **thousands** of goroutines.
* They run **concurrently** with your main program.
* Main program won’t wait for them unless you tell it to.

Example:

```go
package main

import (
    "fmt"
    "time"
)

func printMsg() {
    fmt.Println("Hello from goroutine!")
}

func main() {
    go printMsg() // runs in background

    time.Sleep(time.Second) // give goroutine time to run
}
```

---

## 🟦 **What is a Channel?**

A channel is like a **pipe** used for communication between goroutines.

You can **send** data into a channel and **receive** data from it.

```go
ch := make(chan int)  // channel that sends/receives int values
```

### Send a value:

```go
ch <- 10
```

### Receive a value:

```go
val := <-ch
```

Channels ensure **safe communication** between goroutines without using locks.

---

## 🧠 **Beginner Example Using Goroutine + Channel**

```go
package main

import "fmt"

func sendData(ch chan int) {
    ch <- 42 // send data to channel
}

func main() {
    ch := make(chan int)

    go sendData(ch) // send in background

    value := <-ch   // receive data
    fmt.Println(value)
}
```

✔ No race conditions
✔ Functions can safely send/receive data
✔ Easy concurrency

---

# 🟣 **Intermediate Understanding: Why Channels Matter**

* They help goroutines **synchronize** safely.
* They replace shared memory with **message passing**.
* They act like queues or signals.

---

# 🔥 **Advanced Summary (for mastery)**

Below is the advanced-level explanation used in real-world systems, interviews, and scalable Go design.

---

# 🧵 **Goroutines — Advanced Overview**

### 🔸 What they REALLY are:

Goroutines are scheduled by Go’s **runtime scheduler**, not by the OS.

* They run on top of a pool of OS threads (M:N scheduler).
* Stack grows dynamically from 2KB to MBs.
* Extremely cheap to create.

### 🔸 Their performance strengths:

* Millions can run on a single machine.
* Context switching is extremely fast (no kernel involvement).
* They cooperate with the runtime (blocking ops → runtime parks them).

### 🔸 Lifecycle:

* Created → runnable → executing → parked → dead.
* Block on:

  * I/O
  * channel send/receive
  * mutex locking
  * network
  * `time.Sleep`

---

# 🔗 **Channels — Advanced Overview**

Channels are **synchronization primitives + memory-safe communication pipes**.

### 🔹 Properties:

* FIFO queue behavior
* Blocking send/receive unless channel is buffered
* Prevents race conditions by design

### 🔹 Types:

1. **Unbuffered channels**

   * Send blocks until receive happens
   * Used for strict synchronization

2. **Buffered channels**

   ```go
   ch := make(chan int, 5)
   ```

   * Send does not block until buffer is full
   * Great for pipelines and worker pools

3. **Directional channels**

   ```go
   func worker(in <-chan int, out chan<- int)
   ```

   * Compiler-enforced direction safety

---

# 🛠 **Advanced Use Cases in Real Go Systems**

### 🟡 1. Worker Pools

Efficient background processing.

### 🟡 2. Pipelines

Streaming data stage-by-stage.

### 🟡 3. Actor-like systems

Goroutine per entity model.

### 🟡 4. Fan-in / Fan-out concurrency

Merge or split workloads.

### 🟡 5. Cancellation with `context.Context`

Industry-standard concurrency control.

---

# 🧨 **Advanced Example: Worker Pool**

```go
func worker(id int, jobs <-chan int, results chan<- int) {
    for job := range jobs {
        results <- job * 2
    }
}

func main() {
    jobs := make(chan int, 5)
    results := make(chan int, 5)

    for i := 1; i <= 3; i++ {
        go worker(i, jobs, results)
    }

    for i := 1; i <= 5; i++ {
        jobs <- i
    }
    close(jobs)

    for i := 1; i <= 5; i++ {
        fmt.Println(<-results)
    }
}
```

---

# 🏆 **Final Comparison (Beginner → Advanced)**

| Topic               | Beginner                      | Advanced                                           |
| ------------------- | ----------------------------- | -------------------------------------------------- |
| Goroutine           | Background function           | User-space thread with cooperative scheduling      |
| Cost                | Cheap                         | Stack grows, fast context switching                |
| Channel             | Simple pipe for communication | FIFO synchronized mechanism with blocking patterns |
| Unbuffered          | Send waits for receive        | Guarantees strict synchronization                  |
| Buffered            | Temporary queue               | Decouples producers & consumers                    |
| Communication style | Send/receive                  | Actor model, pipelines, fan-out/in                 |

---
Here are **10 high-quality Go concurrency problems** covering **producer–consumer, worker pools, pipeline, fan-in, fan-out, mutex vs channels, rate limiting, fan-out workers, bounded queues, and cancellation** — **each with clean, working solutions**.

All examples are realistic and match patterns used in production Go systems.

---

# ✅ **1. Producer–Consumer (Unbuffered Channel)**

**Problem:**
Create a producer that generates numbers 1–5 and a consumer that prints them. Use an unbuffered channel so sending and receiving must synchronize.

### ✅ **Solution**

```go
package main

import "fmt"

func main() {
    ch := make(chan int)

    go func() { // Producer
        for i := 1; i <= 5; i++ {
            ch <- i
        }
        close(ch)
    }()

    for v := range ch { // Consumer
        fmt.Println("received:", v)
    }
}
```

---

# ✅ **2. Producer–Consumer with Buffered Channel**

**Problem:**
Modify the above so producer can run ahead using a buffer size 2.

### ✅ **Solution**

```go
ch := make(chan int, 2)
```

Everything else stays the same — now producer is not blocked until buffer is full.

---

# ✅ **3. Worker Pool (Fixed 3 Workers)**

**Problem:**
Given jobs 1–10, create 3 workers, each multiplies a number ×2 and returns results.

### ✅ **Solution**

```go
package main

import (
    "fmt"
)

func worker(id int, jobs <-chan int, results chan<- int) {
    for j := range jobs {
        results <- j * 2
    }
}

func main() {
    jobs := make(chan int, 10)
    results := make(chan int, 10)

    for w := 1; w <= 3; w++ {
        go worker(w, jobs, results)
    }

    for j := 1; j <= 10; j++ {
        jobs <- j
    }
    close(jobs)

    for i := 1; i <= 10; i++ {
        fmt.Println(<-results)
    }
}
```

---

# ✅ **4. Fan-Out (Start N Goroutines to Process the Same Input Channel)**

**Problem:**
Start 5 goroutines reading from **one input channel**, each squares numbers.

### ✅ **Solution**

```go
package main

import "fmt"

func squareWorker(id int, in <-chan int, out chan<- int) {
    for n := range in {
        out <- n * n
    }
}

func main() {
    in := make(chan int)
    out := make(chan int)

    for i := 1; i <= 5; i++ {
        go squareWorker(i, in, out)
    }

    go func() {
        for i := 1; i <= 10; i++ {
            in <- i
        }
        close(in)
    }()

    for i := 1; i <= 10; i++ {
        fmt.Println(<-out)
    }
}
```

---

# ✅ **5. Fan-In (Merge Multiple Channels)**

**Problem:**
Merge 2 producer goroutines into a single output channel.

### ✅ **Solution**

```go
package main

import "fmt"

func fanIn(ch1, ch2 <-chan int) <-chan int {
    out := make(chan int)

    go func() {
        defer close(out)
        for ch1 != nil || ch2 != nil {
            select {
            case v, ok := <-ch1:
                if ok {
                    out <- v
                } else {
                    ch1 = nil
                }
            case v, ok := <-ch2:
                if ok {
                    out <- v
                } else {
                    ch2 = nil
                }
            }
        }
    }()
    return out
}

func main() {
    ch1 := make(chan int)
    ch2 := make(chan int)

    go func() {
        for i := 1; i <= 5; i++ {
            ch1 <- i
        }
        close(ch1)
    }()
    go func() {
        for i := 10; i <= 15; i++ {
            ch2 <- i
        }
        close(ch2)
    }()

    for v := range fanIn(ch1, ch2) {
        fmt.Println(v)
    }
}
```

---

# ✅ **6. Pipeline (Stage 1 → Stage 2 → Stage 3)**

**Problem:**
Create a 3-stage pipeline:

1. generate numbers
2. square
3. double
   Print final results.

### ✅ **Solution**

```go
func generator(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        for _, n := range nums {
            out <- n
        }
        close(out)
    }()
    return out
}

func square(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            out <- n * n
        }
        close(out)
    }()
    return out
}

func double(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            out <- n * 2
        }
        close(out)
    }()
    return out
}

func main() {
    c := generator(1, 2, 3, 4)
    out := double(square(c))

    for v := range out {
        fmt.Println(v)
    }
}
```

---

# ✅ **7. Rate Limiter with Ticker (Allow 1 Task per Second)**

**Problem:**
Process 5 tasks but enforce **1 task per second**.

### ✅ **Solution**

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    tasks := []int{1, 2, 3, 4, 5}
    limiter := time.Tick(1 * time.Second)

    for _, t := range tasks {
        <-limiter
        fmt.Println("processing:", t)
    }
}
```

---

# ✅ **8. Mutex vs Channel (Safe Counter)**

**Problem:**
Create a counter safely incremented by 100 goroutines.

### Solution using **channels** (recommended Go style)

```go
package main

import "fmt"

func main() {
    increments := make(chan int)
    done := make(chan bool)

    go func() {
        count := 0
        for inc := range increments {
            count += inc
        }
        fmt.Println("Final:", count)
        done <- true
    }()

    for i := 0; i < 100; i++ {
        go func() { increments <- 1 }()
    }

    time.Sleep(100 * time.Millisecond) // wait for sends
    close(increments)
    <-done
}
```

---

# ✅ **9. Bounded Queue (Drop if Full)**

**Problem:**
Create a buffered channel of size 2. Any extra message should be dropped, not blocked.

### ✅ **Solution**

```go
queue := make(chan int, 2)

for i := 1; i <= 5; i++ {
    select {
    case queue <- i:
        fmt.Println("added:", i)
    default:
        fmt.Println("dropped:", i)
    }
}
```

---

# ✅ **10. Cancellation with Context (Stop Goroutine Early)**

**Problem:**
Run a goroutine that prints numbers continuously. Stop it after 3 seconds using `context.CancelFunc()`.

### ✅ **Solution**

```go
package main

import (
    "context"
    "fmt"
    "time"
)

func worker(ctx context.Context) {
    i := 0
    for {
        select {
        case <-ctx.Done():
            fmt.Println("stopped")
            return
        default:
            fmt.Println(i)
            i++
            time.Sleep(500 * time.Millisecond)
        }
    }
}

func main() {
    ctx, cancel := context.WithCancel(context.Background())
    go worker(ctx)

    time.Sleep(3 * time.Second)
    cancel()

    time.Sleep(time.Second)
}
```

---
Below is the **clearest explanation of buffered vs unbuffered channels in Go**, with **ASCII visual diagrams**, **real-world meaning**, and **working code examples**.

---

# 🟦 **1. Unbuffered Channels (No Storage)**

### 🔹 Definition

An **unbuffered channel** has **no capacity**.
A send must wait until another goroutine receives.

> **Sender and receiver must meet at the same time.**

### ➤ Create:

```go
ch := make(chan int) // capacity = 0
```

---

# 🔍 **Visual Diagram — Unbuffered Channel**

```
Goroutine A (Sender)        Goroutine B (Receiver)
----------------------      ----------------------
      ch <- 1   ─────────►   waits to receive
      (blocked)              until data arrives
```

There’s **no box**, no storage:

```
[ empty, no buffer ]
```

If receiver is not ready → sender **blocks**.
If sender is not ready → receiver **blocks**.

---

# 🧩 **Example: Unbuffered Channel**

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    ch := make(chan int)

    go func() {
        fmt.Println("Sending...")
        ch <- 10 // blocked until someone receives
        fmt.Println("Sent!")
    }()

    time.Sleep(2 * time.Second)
    fmt.Println("Receiving...")
    v := <-ch
    fmt.Println("Received:", v)
}
```

### Output (timing matters):

```
Sending...
(2 second pause)
Receiving...
Sent!
Received: 10
```

---

# 🟩 **2. Buffered Channels (Has Storage)**

### 🔹 Definition

A **buffered channel** has **capacity > 0**.
Sender can send **without waiting** (until buffer is full).

> **Sender and receiver work at different speeds.**

### ➤ Create:

```go
ch := make(chan int, 3) // capacity = 3
```

---

# 🔍 **Visual Diagram — Buffered Channel**

```
Buffer Capacity: 3
---------------------------------
|   10   |   20   |   __empty__ |
---------------------------------
 Send ok    Send ok   Send ok

```

### ✔ Send non-blocking (until full)

### ✔ Receivers pull from queue (FIFO)

---

# 🧩 **Example: Buffered Channel**

```go
package main

import "fmt"

func main() {
    ch := make(chan int, 2)

    fmt.Println("Sending 1")
    ch <- 1 // ok

    fmt.Println("Sending 2")
    ch <- 2 // ok

    fmt.Println("Sending 3 (will block)")
    go func() {
        ch <- 3 // blocked until someone receives
        fmt.Println("Sent 3")
    }()

    fmt.Println("Receiving:", <-ch)
    fmt.Println("Receiving:", <-ch)
}
```

### Output:

```
Sending 1
Sending 2
Sending 3 (will block)
Receiving: 1
Receiving: 2
Sent 3
```

---

# 🟠 **Side-by-Side Visual Comparison**

### **Unbuffered**

```
Sender ──(blocked)──► [  ] ◄──(blocked)── Receiver
```

* No storage
* Perfect synchronization
* Sender waits for receiver

---

### **Buffered**

```
Sender → [10][20][__][__] ← Receiver
           ↑ storage ↑
```

* Has storage
* Asynchronous communication
* Sender only blocks when buffer is full
* Receiver blocks only when buffer is empty

---

# 🛠 **When to Use Which?**

| Case                                       | Use Unbuffered | Use Buffered |
| ------------------------------------------ | -------------- | ------------ |
| Need strict sync (“meet at the same time”) | ✔              | ✘            |
| Producer is faster/slower than consumer    | ✘              | ✔            |
| Building pipelines / worker pools          | ✘              | ✔            |
| Simple ping-pong signaling                 | ✔              | ✘            |
| Data flows in batches                      | ✘              | ✔            |

---

# 🎯 Real-World Example: Buffered Worker Pool

```
Producer → [jobs buffer] → Workers (goroutines)
```

```go
jobs := make(chan int, 5)

go func() {
    for i := 1; i <= 10; i++ {
        jobs <- i
    }
    close(jobs)
}()
```

---

# 🎯 Real-World Example: Unbuffered Synchronization

```
caller waits → [ ] → goroutine finishes a task and signals
```

```go
done := make(chan bool)

go func() {
    // long task
    done <- true // blocks until main receives
}()

<-done // both sync at this point
```

---

# 🏁 Summary (Super Simple)

| Topic    | Unbuffered   | Buffered                  |
| -------- | ------------ | ------------------------- |
| Capacity | 0            | >0                        |
| Blocking | Sender waits | Sender waits only if full |
| Sync     | Strong sync  | Loose sync                |
| Use Case | Signals      | Pipelines, queues         |

---
Here is a **clean, production-style mini-project** in Go that uses **goroutines + worker pool + buffered channels** to scrape **20 URLs concurrently**.

This example:

* Uses **goroutines** for concurrency
* Uses a **worker pool** (fast + controlled parallelism)
* Uses **channels** for job distribution + results
* Uses `net/http` with timeouts
* Handles errors gracefully

---

# 🚀 **Mini Project: Concurrent Web Scraper (20 URLs)**

**Files:**

* `main.go`

---

# ✅ **main.go**

```go
package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"time"
)

// Worker function
func worker(id int, jobs <-chan string, results chan<- string) {
	client := http.Client{
		Timeout: 5 * time.Second,
	}

	for url := range jobs {
		resp, err := client.Get(url)
		if err != nil {
			results <- fmt.Sprintf("[Worker %d] Error fetching %s: %v", id, url, err)
			continue
		}

		body, _ := ioutil.ReadAll(resp.Body)
		resp.Body.Close()

		results <- fmt.Sprintf("[Worker %d] %s -> %d bytes", id, url, len(body))
	}
}

func main() {
	// 20 test URLs
	urls := []string{
		"https://example.com", "https://httpbin.org/get", "https://golang.org",
		"https://api.github.com", "https://jsonplaceholder.typicode.com/todos/1",
		"https://reqres.in/api/users", "https://dummyjson.com/products",
		"https://httpbin.org/uuid", "https://httpbin.org/ip", "https://httpbin.org/headers",
		"https://example.org", "https://httpstat.us/200", "https://httpstat.us/503",
		"https://httpstat.us/404", "https://httpstat.us/500", "https://icanhazip.com",
		"https://worldtimeapi.org/api/timezone/Etc/UTC",
		"https://jsonplaceholder.typicode.com/posts",
		"https://jsonplaceholder.typicode.com/users",
		"https://jsonplaceholder.typicode.com/comments",
	}

	jobs := make(chan string, len(urls))
	results := make(chan string, len(urls))

	workerCount := 5 // number of goroutines

	// Start workers
	for w := 1; w <= workerCount; w++ {
		go worker(w, jobs, results)
	}

	// Send URLs to jobs channel
	for _, url := range urls {
		jobs <- url
	}
	close(jobs)

	// Collect results
	for i := 0; i < len(urls); i++ {
		fmt.Println(<-results)
	}
}
```

---

# 🧠 **How It Works**

### **1️⃣ Worker Pool**

You start `5` workers:

```go
for w := 1; w <= workerCount; w++ {
    go worker(w, jobs, results)
}
```

Each worker:

* Waits for a URL from `jobs`
* Fetches the website
* Sends the result to `results`

---

### **2️⃣ Job Queue**

Buffered channel holds all 20 URLs:

```go
jobs := make(chan string, len(urls))
```

---

### **3️⃣ Results Queue**

Collect scraped page size or errors:

```go
results := make(chan string, len(urls))
```

---

### **4️⃣ Safe Concurrency**

Workers pull jobs **asynchronously**, so fast workers take more tasks.

This is exactly how production scrapers, crawlers, and pipelines work.

---

# 📌 Example Output (varies)

```
[Worker 2] https://example.com -> 1256 bytes
[Worker 1] https://httpbin.org/get -> 346 bytes
[Worker 4] https://golang.org -> 8900 bytes
[Worker 3] https://api.github.com -> 2345 bytes
[Worker 5] https://dummyjson.com/products -> 12000 bytes
...
```

---
Here is a clean, complete explanation of **Go’s error-handling philosophy** along with **custom error type examples**.

---

# ✅ **Go’s Error Handling Philosophy**

Go’s error handling is **explicit**, **simple**, and **predictable**.
Unlike languages such as Java, Python, or C++, Go **does not use exceptions** for normal error flow.

### 🔥 Core Principles

### **1. Errors are values**

In Go, an error is just a value that implements:

```go
type error interface {
    Error() string
}
```

Errors are returned like any other value.

---

### **2. No try/catch — check errors explicitly**

Instead of exception blocks, Go encourages you to check errors every time:

```go
data, err := readFile("config.json")
if err != nil {
    return err
}
```

This makes error handling:

* predictable
* visible
* easier to reason about

---

### **3. Fail early, return fast**

Functions usually return early when something goes wrong:

```go
if err != nil {
    return err
}
```

This avoids deep nesting.

---

### **4. Context matters — wrap errors**

Go encourages adding context:

```go
return fmt.Errorf("failed to open config: %w", err)
```

`%w` enables error unwrapping (`errors.Is`, `errors.As`).

---

### **5. Sentinel errors exist, but avoid too many**

You can define a global error:

```go
var ErrUserNotFound = errors.New("user not found")
```

But Go devs recommend using **wrapping** instead to prevent brittle code.

---

### **6. Prefer custom error types for rich info**

When you need more than a message—like an error code—define a struct.

---

# ✅ **Custom Error Types in Go (Examples)**

Below are different styles you MUST know.

---

## **1️⃣ Simple Custom Error Type**

### Example: Validation Error

```go
package main

import (
    "fmt"
)

type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation failed on %s: %s", e.Field, e.Message)
}

func validateAge(age int) error {
    if age < 18 {
        return &ValidationError{
            Field:   "age",
            Message: "must be at least 18",
        }
    }
    return nil
}
```

Usage:

```go
err := validateAge(10)
fmt.Println(err)
```

---

## **2️⃣ Rich Error With Code**

Useful for APIs.

```go
type APIError struct {
    Code    int
    Message string
}

func (e *APIError) Error() string {
    return fmt.Sprintf("API error %d: %s", e.Code, e.Message)
}

func GetUser(id int) error {
    if id == 0 {
        return &APIError{Code: 404, Message: "user not found"}
    }
    return nil
}
```

---

## **3️⃣ Wrap Errors With Context (Recommended)**

```go
func readConfig(path string) error {
    data, err := os.ReadFile(path)
    if err != nil {
        return fmt.Errorf("readConfig: %w", err)
    }
    _ = data
    return nil
}
```

Unwrap with:

```go
if errors.Is(err, os.ErrNotExist) {
    fmt.Println("File does not exist!")
}
```

---

## **4️⃣ Custom Error With Time Information**

```go
type TimeoutError struct {
    Operation string
    Duration  time.Duration
}

func (e *TimeoutError) Error() string {
    return fmt.Sprintf("%s timed out after %s", e.Operation, e.Duration)
}
```

---

## **5️⃣ Domain-Specific Error System Example**

Imagine a payment system:

```go
type PaymentError struct {
    Code       string
    Amount     float64
    Additional string
}

func (e *PaymentError) Error() string {
    return fmt.Sprintf("Payment error [%s]: amount=%.2f %s",
        e.Code, e.Amount, e.Additional)
}

func Charge(amount float64) error {
    if amount <= 0 {
        return &PaymentError{
            Code:       "INVALID_AMOUNT",
            Amount:     amount,
            Additional: "must be greater than zero",
        }
    }
    return nil
}
```

---

# 🧠 **When to Use Custom Errors in Go**

Use custom error types when you need to:

✔ attach metadata (codes, fields, timestamps)
✔ identify a specific error type via `errors.As`
✔ differentiate errors programmatically
✔ build APIs or large-scale applications

---

# ⭐ Final Summary

| Feature          | Go’s Philosophy                          |
| ---------------- | ---------------------------------------- |
| Errors mechanism | Errors are **values**, not exceptions    |
| Handling         | Always **explicit** with `if err != nil` |
| Wrapping         | Use `fmt.Errorf("... %w", err)`          |
| Sentinel errors  | Ok in small cases, avoid many            |
| Custom errors    | Use struct-based errors for metadata     |

---
Below is a **clear, complete, production-ready guide** showing how to implement **structured logging in Go** using **Zap** (Uber) and **Logrus** (Sirupsen).

You’ll learn:

✅ Setup
✅ Basic usage
✅ Fields (structured logs)
✅ Log levels
✅ JSON logging
✅ Middleware usage (HTTP)
✅ Best practices

---

# 🚀 **1. Structured Logging With Zap**

Zap is one of the **fastest** structured loggers in Go.

---

## ✅ **Installation**

```bash
go get go.uber.org/zap
```

---

# ⭐ **Example 1: Using Zap (Sugared Logger)**

Good for developer-friendly logs.

```go
package main

import (
    "go.uber.org/zap"
)

func main() {
    logger, _ := zap.NewProduction()
    defer logger.Sync()

    sugar := logger.Sugar()

    sugar.Infow("User logged in",
        "user_id", 42,
        "ip", "192.168.1.1",
    )
    sugar.Warnw("Low balance",
        "account", "ACC123",
        "balance", 50.25,
        "currency", "INR",
    )
}
```

### Output (JSON structured):

```json
{
  "level":"info",
  "msg":"User logged in",
  "user_id":42,
  "ip":"192.168.1.1",
  "time":"2025-01-01T12:30:00Z"
}
```

---

# ⭐ **Example 2: Using Zap (Strict Structured Logger)**

Use this when performance matters.

```go
logger, _ := zap.NewProduction()

logger.Info("Payment processed",
    zap.String("user", "sandeep"),
    zap.Float64("amount", 799.50),
    zap.Bool("success", true),
)
```

---

# ⭐ **Example 3: Custom Zap Configuration**

```go
config := zap.Config{
    Encoding:         "json",
    Level:            zap.NewAtomicLevelAt(zap.InfoLevel),
    OutputPaths:      []string{"stdout", "logs/app.log"},
    ErrorOutputPaths: []string{"stderr"},
    EncoderConfig: zap.NewProductionEncoderConfig(),
}

logger, _ := config.Build()
logger.Info("custom config log", zap.String("env", "production"))
```

---

# 🌐 **Example 4: Zap with Gin/Fiber HTTP Middleware**

### Zap Middleware Sample:

```go
func LoggingMiddleware(logger *zap.Logger) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {

            logger.Info("Incoming request",
                zap.String("method", r.Method),
                zap.String("path", r.URL.Path),
                zap.String("ip", r.RemoteAddr),
            )

            next.ServeHTTP(w, r)
        })
    }
}
```

---

# ⭐ **Zap Best Practices**

✔ Use `zap.NewProduction()` in production
✔ Use structured fields (`zap.String()`, `zap.Bool()`)
✔ Always call `defer logger.Sync()`
✔ Avoid `fmt.Println` inside production services

---

---

# 🔥 2. Structured Logging With Logrus

Logrus is widely used and simple, but slower than Zap.

---

## ✅ Installation

```bash
go get github.com/sirupsen/logrus
```

---

## ⭐ Example 1: Basic Structured Log

```go
package main

import log "github.com/sirupsen/logrus"

func main() {
    log.SetFormatter(&log.JSONFormatter{})

    log.WithFields(log.Fields{
        "user": "sandeep",
        "role": "admin",
        "ip":   "127.0.0.1",
    }).Info("User logged in")
}
```

---

## ⭐ Example 2: Log Levels

```go
log.SetLevel(log.WarnLevel)

log.Info("This will NOT appear")
log.Warn("This will appear")
log.Error("This is an error")
```

---

## ⭐ Example 3: Custom Logger Instance

```go
logger := log.New()

logger.SetFormatter(&log.JSONFormatter{})
logger.SetOutput(os.Stdout)
logger.SetLevel(log.DebugLevel)

logger.WithFields(log.Fields{
    "order_id": 999,
    "amount":   230.50,
}).Info("Order processed")
```

---

## ⭐ Example 4: Logrus Hook (advanced)

Add hooks like file output, Sentry, Kafka, etc.

```go
type AuthHook struct{}

func (h *AuthHook) Levels() []log.Level {
    return log.AllLevels
}

func (h *AuthHook) Fire(entry *log.Entry) error {
    entry.Data["service"] = "auth-service"
    return nil
}

func main() {
    log.AddHook(&AuthHook{})
    log.Warn("Unauthorized access")
}
```

---

# ⭐ **Logrus HTTP Middleware Example**

```go
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {

        log.WithFields(log.Fields{
            "method": r.Method,
            "path":   r.URL.Path,
            "ip":     r.RemoteAddr,
        }).Info("Incoming Request")

        next.ServeHTTP(w, r)
    })
}
```

---

# ⚔️ **Zap vs Logrus — Which Should You Use?**

| Feature           | Zap         | Logrus       |
| ----------------- | ----------- | ------------ |
| Speed             | ⚡ Very Fast | ❌ Slower     |
| Flexibility       | High        | Medium       |
| Beginner Friendly | Medium      | Easy         |
| Production Use    | ⭐ Best      | Good         |
| Extensibility     | Medium      | High (Hooks) |

✔ For serious backend projects → **Use Zap**
✔ For quick prototyping or CLIs → **Use Logrus**

---
Here is a **clean, complete, practical cheat sheet** with examples of **reading/writing files, JSON, and CSV in Go**.
All examples are beginner-friendly but production-ready.

---

# 📁 **1. Reading & Writing Files in Go**

---

## ✅ **1.1 Write to a File**

```go
package main

import (
    "os"
)

func main() {
    content := []byte("Hello, Go file handling!")

    err := os.WriteFile("output.txt", content, 0644)
    if err != nil {
        panic(err)
    }
}
```

---

## ✅ **1.2 Read a File Completely**

```go
data, err := os.ReadFile("output.txt")
if err != nil {
    panic(err)
}

fmt.Println(string(data))
```

---

## ✅ **1.3 Append to a File**

```go
f, err := os.OpenFile("log.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
if err != nil {
    panic(err)
}
defer f.Close()

f.WriteString("New log entry\n")
```

---

## ✅ **1.4 Read File Line-by-Line**

```go
file, err := os.Open("data.txt")
if err != nil {
    panic(err)
}
defer file.Close()

scanner := bufio.NewScanner(file)

for scanner.Scan() {
    fmt.Println(scanner.Text())
}
```

---

# 🔧 **2. JSON Handling in Go**

Go uses the `encoding/json` package.

---

## 📌 **2.1 Define a Struct for JSON**

```go
type User struct {
    Name  string `json:"name"`
    Email string `json:"email"`
    Age   int    `json:"age"`
}
```

---

## ✅ **2.2 Convert Struct → JSON**

```go
user := User{Name: "Sandeep", Email: "san@example.com", Age: 22}

jsonData, err := json.MarshalIndent(user, "", "  ")
if err != nil {
    panic(err)
}

os.WriteFile("user.json", jsonData, 0644)
```

---

## ✅ **2.3 Read JSON File → Struct**

```go
data, err := os.ReadFile("user.json")
if err != nil {
    panic(err)
}

var user User
err = json.Unmarshal(data, &user)
if err != nil {
    panic(err)
}

fmt.Println(user)
```

---

## ✅ **2.4 Decode JSON From Reader (API style)**

```go
file, _ := os.Open("user.json")
defer file.Close()

var u User
json.NewDecoder(file).Decode(&u)
fmt.Println(u)
```

---

## ❗ **2.5 Encode Struct → JSON to stdout or writer**

```go
json.NewEncoder(os.Stdout).Encode(user)
```

---

## 🎯 **2.6 Parse JSON to `map[string]interface{}` (dynamic JSON)**

```go
var obj map[string]interface{}

err := json.Unmarshal(data, &obj)
if err != nil {
    panic(err)
}

fmt.Println(obj["name"])
```

---

# 📊 **3. CSV Handling in Go**

Uses `encoding/csv`.

---

## 🎯 **3.1 Write CSV File**

```go
file, err := os.Create("people.csv")
if err != nil {
    panic(err)
}
defer file.Close()

writer := csv.NewWriter(file)
defer writer.Flush()

writer.Write([]string{"name", "email", "age"})
writer.Write([]string{"Sandeep", "san@example.com", "22"})
writer.Write([]string{"Raj", "raj@example.com", "24"})
```

---

## 🎯 **3.2 Read Entire CSV File**

```go
file, err := os.Open("people.csv")
if err != nil {
    panic(err)
}
defer file.Close()

reader := csv.NewReader(file)

records, err := reader.ReadAll()
if err != nil {
    panic(err)
}

for _, record := range records {
    fmt.Println(record)
}
```

---

## 🎯 **3.3 Read CSV Line-by-Line**

```go
file, _ := os.Open("people.csv")
defer file.Close()

reader := csv.NewReader(file)

for {
    row, err := reader.Read()
    if err == io.EOF {
        break
    }
    if err != nil {
        panic(err)
    }

    fmt.Println("Name:", row[0], "Email:", row[1], "Age:", row[2])
}
```

---

# 💡 Bonus: Convert CSV → Struct Slice

```go
type Person struct {
    Name  string
    Email string
    Age   int
}

func ReadCSVToStruct() []Person {
    file, _ := os.Open("people.csv")
    defer file.Close()

    reader := csv.NewReader(file)
    rows, _ := reader.ReadAll()

    persons := []Person{}

    for _, row := range rows[1:] { // skip header
        age, _ := strconv.Atoi(row[2])

        persons = append(persons, Person{
            Name:  row[0],
            Email: row[1],
            Age:   age,
        })
    }
    return persons
}
```

---

# 🏁 Summary Table

| Task               | Code                             |
| ------------------ | -------------------------------- |
| File write/read    | `os.WriteFile`, `os.ReadFile`    |
| Append             | `os.OpenFile`                    |
| JSON encode/decode | `json.Marshal`, `json.Unmarshal` |
| CSV read/write     | `csv.NewWriter`, `csv.NewReader` |

---
Below is a **clean, complete, practical guide** to help you *fully understand* Go’s **`io.Reader`** and **`io.Writer`** interfaces — including **5 REAL use cases**, code samples, diagrams, and intuitive explanations.

---

# 🚀 **Go’s `io.Reader` & `io.Writer` — The Heart of Go I/O**

Go’s entire input/output system is built on two tiny interfaces:

---

## 🌟 **1. `io.Reader` Interface**

```go
type Reader interface {
    Read(p []byte) (n int, err error)
}
```

* Reads **data INTO** a byte slice (`p`)
* Returns number of bytes read (`n`)
* Returns `io.EOF` when data ends

Any type that has `Read([]byte)` is a Reader:
✔ File
✔ Network connection
✔ HTTP response body
✔ String reader
✔ Buffer
✔ Custom readers

---

## 🌟 **2. `io.Writer` Interface**

```go
type Writer interface {
    Write(p []byte) (n int, err error)
}
```

* Writes **data FROM** a byte slice
* Returns number of bytes written

Any type that supports `Write()` is a Writer:
✔ File
✔ HTTP request body
✔ Network connection
✔ In-memory buffer
✔ Loggers
✔ gzip writer

---

# 🧠 **Why are Reader/Writer so Powerful?**

Because they enable:

✔ streaming (don’t load full files)
✔ composability (you can chain readers/writers)
✔ working with ANY input/output source
✔ memory-efficient file/network processing

---

# 🧩 **5 Real-World Use Cases (with Working Code)**

---

# ✅ **Use Case 1 — Copying Files Efficiently**

Go provides a built-in function that works with ANY Reader/Writer:

```go
func Copy(dst Writer, src Reader) (written int64, err error)
```

### 📌 Example: Copy file → file

```go
package main

import (
    "io"
    "os"
)

func main() {
    src, _ := os.Open("input.txt")
    defer src.Close()

    dst, _ := os.Create("output.txt")
    defer dst.Close()

    _, err := io.Copy(dst, src)
    if err != nil {
        panic(err)
    }
}
```

✔ Doesn’t load entire file
✔ Works for any input type (file, network, gzip, strings, etc.)

---

# ✅ **Use Case 2 — Reading an HTTP Response (Reader)**

`resp.Body` from `http.Get` is an `io.Reader`.

```go
resp, _ := http.Get("https://api.github.com")
defer resp.Body.Close()

data, _ := io.ReadAll(resp.Body)
fmt.Println(string(data))
```

✔ Resp.Body is a Reader
✔ You can stream large responses safely

---

# ✅ **Use Case 3 — Writing an HTTP Request Body (Writer)**

`http.Request` bodies use Writers behind the scenes.

### Example: Upload JSON to a server

```go
jsonData := strings.NewReader(`{"name": "sandeep"}`)

resp, err := http.Post(
    "https://example.com/api",
    "application/json",
    jsonData, // <-- io.Reader
)
```

Any Reader works (file, bytes, strings)!

---

# ✅ **Use Case 4 — Compressing Data (Writer wrapping Writer)**

Go lets Writers wrap other Writers (powerful!).

### gzip.Writer wraps another Writer.

```go
package main

import (
    "compress/gzip"
    "os"
)

func main() {
    file, _ := os.Create("data.gz")
    defer file.Close()

    gzipWriter := gzip.NewWriter(file)
    defer gzipWriter.Close()

    gzipWriter.Write([]byte("Hello from Golang"))
}
```

✔ gzip.Writer is a Writer
✔ It writes compressed data to the underlying Writer

---

# ✅ **Use Case 5 — Using In-Memory Buffers (Reader + Writer)**

`bytes.Buffer` implements BOTH interfaces.

### Example: Buffer as an in-memory file

```go
package main

import (
    "bytes"
    "fmt"
    "io"
)

func main() {
    var buf bytes.Buffer

    buf.Write([]byte("Hello "))     // Writer
    buf.WriteString("Sandeep!")    // Writer

    data, _ := io.ReadAll(&buf)    // Reader
    fmt.Println(string(data))
}
```

✔ Memory-efficient
✔ Avoids temp files
✔ Great for logs, caching, temporary data

---

# 🧑‍🏫 **Bonus: Building a Custom Reader**

You can create a custom Reader by implementing `Read()`.

### Example: A reader that generates “Hello” repeatedly

```go
type HelloReader struct{}

func (HelloReader) Read(p []byte) (int, error) {
    copy(p, []byte("Hello"))
    return 5, nil
}
```

Use it:

```go
hr := HelloReader{}
buf := make([]byte, 5)
hr.Read(buf)
fmt.Println(string(buf)) // Hello
```

Yes — this is how many Go tools work internally.

---

# 🧠 Visual Diagram — How Go’s Reader/Writer Flow Works

```
[Source: file | string | network | gzip] --> io.Reader --> your program
your program --> io.Writer --> [Destination: file | network | buffer]
```

Or chained:

```
File --> gzip.Reader --> io.Copy --> http.ResponseWriter
```

---

# ⭐ Summary Table

| Concept     | Purpose                      | Examples                         |
| ----------- | ---------------------------- | -------------------------------- |
| `io.Reader` | Reads data into byte slices  | file, http body, buffer, strings |
| `io.Writer` | Writes data from byte slices | file, http body, loggers         |
| `io.Copy`   | Copies Reader → Writer       | file copy, stream copy           |
| Chaining    | Wrap Writer/Reader           | gzip → file, bufio → file        |
| Buffer      | Implements both              | temp logs, memory store          |

---
Below is a **beginner-friendly, complete, production-style guide** on building a **REST API in Go using only `net/http` — absolutely NO frameworks**.

This gives you a solid foundation for mastering Go backend development.

---

# 🚀 **1. Project Structure**

```
go-rest-api/
│── main.go
│── handlers/
│     └── user.go
│── models/
│     └── user.go
│── storage/
│     └── memory.go
```

Simple, clean, scalable.

---

# 🚀 **2. Create a Basic HTTP Server**

`main.go`

```go
package main

import (
    "log"
    "net/http"
)

func main() {
    mux := http.NewServeMux()

    // Register routes
    mux.HandleFunc("/users", usersHandler)
    mux.HandleFunc("/users/", userByIDHandler)

    log.Println("Server running on :8080")
    http.ListenAndServe(":8080", mux)
}
```

✔ `http.NewServeMux()` = router
✔ `ListenAndServe` starts the server

---

# 🧱 **3. Create a User Model**

`models/user.go`

```go
package models

type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}
```

---

# 🗂 **4. Simple In-Memory Storage (No database)**

`storage/memory.go`

```go
package storage

import "go-rest-api/models"

var Users = []models.User{
    {ID: 1, Name: "Sandeep", Email: "san@example.com"},
    {ID: 2, Name: "Raj", Email: "raj@example.com"},
}
```

---

# 🚥 **5. Implement Handlers (Core Logic)**

## 📌 **5.1 GET /users — list all users**

`handlers/user.go`

```go
package main

import (
    "encoding/json"
    "net/http"
    "go-rest-api/storage"
)

func usersHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method == "GET" {
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(storage.Users)
        return
    }

    http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
}
```

---

## 📌 **5.2 GET /users/{id} — fetch one user**

```go
func userByIDHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != "GET" {
        http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
        return
    }

    // Extract ID from URL
    idStr := r.URL.Path[len("/users/"):]
    id, _ := strconv.Atoi(idStr)

    // Search user
    for _, user := range storage.Users {
        if user.ID == id {
            w.Header().Set("Content-Type", "application/json")
            json.NewEncoder(w).Encode(user)
            return
        }
    }

    http.NotFound(w, r)
}
```

---

# ✍️ **6. POST /users — create a new user**

```go
func usersHandler(w http.ResponseWriter, r *http.Request) {
    switch r.Method {
    case "GET":
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(storage.Users)

    case "POST":
        var user models.User
        json.NewDecoder(r.Body).Decode(&user)

        // Set ID
        user.ID = len(storage.Users) + 1
        storage.Users = append(storage.Users, user)

        w.Header().Set("Content-Type", "application/json")
        w.WriteHeader(http.StatusCreated)
        json.NewEncoder(w).Encode(user)

    default:
        http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
    }
}
```

---

# 🛠 **7. PUT /users/{id} — update user**

```go
func userByIDHandler(w http.ResponseWriter, r *http.Request) {
    idStr := r.URL.Path[len("/users/"):]
    id, _ := strconv.Atoi(idStr)

    switch r.Method {
    case "GET":
        for _, user := range storage.Users {
            if user.ID == id {
                json.NewEncoder(w).Encode(user)
                return
            }
        }
        http.NotFound(w, r)

    case "PUT":
        var updated models.User
        json.NewDecoder(r.Body).Decode(&updated)

        for i, user := range storage.Users {
            if user.ID == id {
                updated.ID = id
                storage.Users[i] = updated
                json.NewEncoder(w).Encode(updated)
                return
            }
        }
        http.NotFound(w, r)

    default:
        http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
    }
}
```

---

# ❌ **8. DELETE /users/{id} — delete user**

```go
case "DELETE":
    for i, user := range storage.Users {
        if user.ID == id {
            storage.Users = append(storage.Users[:i], storage.Users[i+1:]...)
            w.WriteHeader(http.StatusNoContent)
            return
        }
    }
    http.NotFound(w, r)
```

---

# 🧪 **9. Test Your API With curl**

### Get all users:

```
curl http://localhost:8080/users
```

### Create user:

```
curl -X POST -d '{"name":"Amit","email":"amit@gmail.com"}' \
     -H "Content-Type: application/json" \
     http://localhost:8080/users
```

---

# ⚡ Bonus — Middleware (Logging Example)

```go
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        log.Printf("%s %s", r.Method, r.URL.Path)
        next.ServeHTTP(w, r)
    })
}
```

Usage:

```go
mux := http.NewServeMux()
mux.HandleFunc("/users", usersHandler)
mux.HandleFunc("/users/", userByIDHandler)

http.ListenAndServe(":8080", LoggingMiddleware(mux))
```

---

# 💎 Summary — What You Learned

You now know how to build a REST API in Go using only:

✔ `net/http`
✔ `ServeMux`
✔ Handlers
✔ JSON encoding/decoding
✔ URL path parsing
✔ CRUD operations
✔ Middleware
✔ In-memory storage

This is the **foundation** of all Go backend frameworks (Gin, Echo, Fiber).

---
Below is a clean, production-style **CRUD API using Gin + PostgreSQL**, with migrations, folder structure, and full code.

---

# ✅ **Gin + PostgreSQL CRUD API (Users Example)**

### **Features**

* Create, Read, Update, Delete users
* PostgreSQL connection
* Environment variables
* Repository + Service + Handler clean structure
* Error handling
* JSON input validation

---

# 📁 **Folder Structure**

```
go-gin-crud/
│── cmd/
│   └── main.go
│── config/
│   └── config.go
│── db/
│   └── migrate.go
│── handlers/
│   └── user_handler.go
│── models/
│   └── user.go
│── repository/
│   └── user_repo.go
│── service/
│   └── user_service.go
│── go.mod
│── .env
```

---

# 📌 **1. go.mod**

```go
module go-gin-crud

go 1.22

require (
    github.com/gin-gonic/gin v1.10.0
    github.com/joho/godotenv v1.5.1
    github.com/jackc/pgx/v5 v5.5.0
)
```

---

# 📌 **2. .env**

```
DB_HOST=localhost
DB_USER=postgres
DB_PASS=123456
DB_NAME=gocrud
DB_PORT=5432
```

---

# 📌 **3. config/config.go → Load ENV + Connect PostgreSQL**

```go
package config

import (
	"context"
	"fmt"
	"os"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv"
)

var DB *pgxpool.Pool

func LoadEnv() {
	godotenv.Load()
}

func ConnectDB() {
	url := fmt.Sprintf(
		"postgres://%s:%s@%s:%s/%s",
		os.Getenv("DB_USER"),
		os.Getenv("DB_PASS"),
		os.Getenv("DB_HOST"),
		os.Getenv("DB_PORT"),
		os.Getenv("DB_NAME"),
	)

	pool, err := pgxpool.New(context.Background(), url)
	if err != nil {
		panic(err)
	}

	DB = pool
	fmt.Println("Connected to PostgreSQL!")
}
```

---

# 📌 **4. models/user.go**

```go
package models

type User struct {
	ID    int    `json:"id"`
	Name  string `json:"name"`
	Email string `json:"email"`
}
```

---

# 📌 **5. db/migrate.go**

```go
package db

import (
	"context"
	"fmt"
	"go-gin-crud/config"
)

func RunMigrations() {
	q := `
	CREATE TABLE IF NOT EXISTS users (
		id SERIAL PRIMARY KEY,
		name VARCHAR(100),
		email VARCHAR(150)
	)
	`
	_, err := config.DB.Exec(context.Background(), q)
	if err != nil {
		fmt.Println("Migration error:", err)
		return
	}

	fmt.Println("Migration completed!")
}
```

---

# 📌 **6. repository/user_repo.go**

```go
package repository

import (
	"context"
	"go-gin-crud/config"
	"go-gin-crud/models"
)

type UserRepository struct{}

func (r *UserRepository) Create(u models.User) error {
	_, err := config.DB.Exec(context.Background(),
		"INSERT INTO users(name, email) VALUES ($1, $2)",
		u.Name, u.Email)
	return err
}

func (r *UserRepository) GetAll() ([]models.User, error) {
	rows, err := config.DB.Query(context.Background(), "SELECT id, name, email FROM users")
	if err != nil {
		return nil, err
	}

	var users []models.User
	for rows.Next() {
		var u models.User
		rows.Scan(&u.ID, &u.Name, &u.Email)
		users = append(users, u)
	}
	return users, nil
}

func (r *UserRepository) GetByID(id int) (models.User, error) {
	var u models.User
	err := config.DB.QueryRow(context.Background(),
		"SELECT id, name, email FROM users WHERE id=$1", id).
		Scan(&u.ID, &u.Name, &u.Email)

	return u, err
}

func (r *UserRepository) Update(id int, u models.User) error {
	_, err := config.DB.Exec(context.Background(),
		"UPDATE users SET name=$1, email=$2 WHERE id=$3",
		u.Name, u.Email, id)
	return err
}

func (r *UserRepository) Delete(id int) error {
	_, err := config.DB.Exec(context.Background(),
		"DELETE FROM users WHERE id=$1", id)
	return err
}
```

---

# 📌 **7. service/user_service.go**

```go
package service

import (
	"go-gin-crud/models"
	"go-gin-crud/repository"
)

type UserService struct {
	Repo repository.UserRepository
}

func (s *UserService) Create(u models.User) error {
	return s.Repo.Create(u)
}

func (s *UserService) GetAll() ([]models.User, error) {
	return s.Repo.GetAll()
}

func (s *UserService) GetByID(id int) (models.User, error) {
	return s.Repo.GetByID(id)
}

func (s *UserService) Update(id int, u models.User) error {
	return s.Repo.Update(id, u)
}

func (s *UserService) Delete(id int) error {
	return s.Repo.Delete(id)
}
```

---

# 📌 **8. handlers/user_handler.go**

```go
package handlers

import (
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"

	"go-gin-crud/models"
	"go-gin-crud/service"
)

type UserHandler struct {
	Service service.UserService
}

func (h *UserHandler) Create(c *gin.Context) {
	var u models.User
	if err := c.ShouldBindJSON(&u); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid JSON"})
		return
	}

	if err := h.Service.Create(u); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "DB error"})
		return
	}

	c.JSON(http.StatusCreated, gin.H{"message": "user created"})
}

func (h *UserHandler) GetAll(c *gin.Context) {
	users, err := h.Service.GetAll()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "DB error"})
		return
	}
	c.JSON(http.StatusOK, users)
}

func (h *UserHandler) GetByID(c *gin.Context) {
	id, _ := strconv.Atoi(c.Param("id"))
	u, err := h.Service.GetByID(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
		return
	}
	c.JSON(http.StatusOK, u)
}

func (h *UserHandler) Update(c *gin.Context) {
	id, _ := strconv.Atoi(c.Param("id"))
	var u models.User

	if err := c.ShouldBindJSON(&u); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid JSON"})
		return
	}

	if err := h.Service.Update(id, u); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "DB error"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "updated"})
}

func (h *UserHandler) Delete(c *gin.Context) {
	id, _ := strconv.Atoi(c.Param("id"))
	if err := h.Service.Delete(id); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "DB error"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "deleted"})
}
```

---

# 📌 **9. cmd/main.go**

```go
package main

import (
	"go-gin-crud/config"
	"go-gin-crud/db"
	"go-gin-crud/handlers"
	"go-gin-crud/repository"
	"go-gin-crud/service"

	"github.com/gin-gonic/gin"
)

func main() {
	config.LoadEnv()
	config.ConnectDB()
	db.RunMigrations()

	repo := repository.UserRepository{}
	serv := service.UserService{Repo: repo}
	handler := handlers.UserHandler{Service: serv}

	r := gin.Default()

	r.POST("/users", handler.Create)
	r.GET("/users", handler.GetAll)
	r.GET("/users/:id", handler.GetByID)
	r.PUT("/users/:id", handler.Update)
	r.DELETE("/users/:id", handler.Delete)

	r.Run(":8080")
}
```

---

# 🚀 **Run the API**

```
go mod tidy
go run cmd/main.go
```

---

# 📌 **API Test Examples (cURL)**

### **Create**

```
curl -X POST localhost:8080/users \
 -H "Content-Type: application/json" \
 -d '{"name":"Sandeep","email":"sandeep@gmail.com"}'
```

### **Get All**

```
curl localhost:8080/users
```

### **Update**

```
curl -X PUT localhost:8080/users/1 \
 -H "Content-Type: application/json" \
 -d '{"name":"Updated","email":"new@gmail.com"}'
```

---
# WebSocket server in Go (complete example)

Below is a production-ish, easy-to-follow single-file WebSocket server in Go using the popular `gorilla/websocket` package. It implements a **hub** (broadcast to all clients), handles concurrent clients safely, supports ping/pong heartbeats, and shows how to send real-time messages from the server (periodic and on-demand).

## Files you’ll create

1. `go.mod` (small)
2. `server.go` (the Go server)
3. `client.html` (simple browser client to test)

---

# 1) `go.mod`

Run:

```bash
go mod init example.com/wsdemo
go get github.com/gorilla/websocket
```

(Or create `go.mod` with `module example.com/wsdemo` and `require github.com/gorilla/websocket v1.5.0` — `go get` will fetch the right version.)

---

# 2) `server.go`

```go
package main

import (
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
)

// ---- Configuration ----
const (
	// Time allowed to write a message to the peer.
	writeWait = 10 * time.Second

	// Time allowed to read the next pong message from the peer.
	pongWait = 60 * time.Second

	// Send pings to peer with this period. Must be less than pongWait.
	pingPeriod = (pongWait * 9) / 10

	// Maximum message size allowed from peer.
	maxMessageSize = 512
)

// ---- Upgrader ----
var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	// In production, check the origin and allow only trusted origins.
	CheckOrigin: func(r *http.Request) bool { return true },
}

// ---- Hub (broadcast) ----
type Hub struct {
	clients    map[*Client]bool
	broadcast  chan []byte
	register   chan *Client
	unregister chan *Client
}

func newHub() *Hub {
	return &Hub{
		clients:    make(map[*Client]bool),
		broadcast:  make(chan []byte),
		register:   make(chan *Client),
		unregister: make(chan *Client),
	}
}

func (h *Hub) run() {
	for {
		select {
		case c := <-h.register:
			h.clients[c] = true
		case c := <-h.unregister:
			if _, ok := h.clients[c]; ok {
				delete(h.clients, c)
				close(c.send)
			}
		case msg := <-h.broadcast:
			for c := range h.clients {
				// non-blocking send; drop if buffer full
				select {
				case c.send <- msg:
				default:
					close(c.send)
					delete(h.clients, c)
				}
			}
		}
	}
}

// ---- Client ----
type Client struct {
	hub  *Hub
	conn *websocket.Conn
	send chan []byte
}

func (c *Client) readPump() {
	defer func() {
		c.hub.unregister <- c
		c.conn.Close()
	}()

	c.conn.SetReadLimit(maxMessageSize)
	_ = c.conn.SetReadDeadline(time.Now().Add(pongWait))
	c.conn.SetPongHandler(func(string) error {
		_ = c.conn.SetReadDeadline(time.Now().Add(pongWait))
		return nil
	})

	for {
		_, message, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("unexpected close: %v", err)
			}
			break
		}
		// Echo received message back to all clients (broadcast)
		incoming := fmt.Sprintf("recv: %s", message)
		c.hub.broadcast <- []byte(incoming)
	}
}

func (c *Client) writePump() {
	ticker := time.NewTicker(pingPeriod)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case msg, ok := <-c.send:
			_ = c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if !ok {
				// Hub closed the channel.
				_ = c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			w, err := c.conn.NextWriter(websocket.TextMessage)
			if err != nil {
				return
			}
			_, _ = w.Write(msg)

			// If there are queued messages, send them in the same WebSocket message to reduce writes.
			n := len(c.send)
			for i := 0; i < n; i++ {
				_, _ = w.Write([]byte{'\n'})
				_, _ = w.Write(<-c.send)
			}

			if err := w.Close(); err != nil {
				return
			}

		case <-ticker.C:
			_ = c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// ---- HTTP handler to upgrade connection ----
func serveWs(hub *Hub, w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("upgrade:", err)
		return
	}
	client := &Client{hub: hub, conn: conn, send: make(chan []byte, 256)}
	client.hub.register <- client

	// Start pumps
	go client.writePump()
	go client.readPump()
}

// ---- Example: server-side periodic broadcaster ----
func startPeriodicBroadcast(hub *Hub) {
	ticker := time.NewTicker(5 * time.Second)
	go func() {
		counter := 0
		for range ticker.C {
			counter++
			msg := fmt.Sprintf("server-time: %s (tick %d)", time.Now().Format(time.RFC3339), counter)
			hub.broadcast <- []byte(msg)
		}
	}()
}

func main() {
	hub := newHub()
	go hub.run()
	startPeriodicBroadcast(hub)

	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		serveWs(hub, w, r)
	})

	// Simple health endpoint
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("WebSocket server running. Connect at /ws"))
	})

	addr := ":8080"
	log.Printf("Server started at %s", addr)
	err := http.ListenAndServe(addr, nil)
	if err != nil {
		log.Fatal("ListenAndServe:", err)
	}
}
```

---

# 3) `client.html` (test in a browser)

Save this file and open it in a browser (or serve it with a simple static server). Change the WebSocket URL if the server runs remotely or on a different port.

```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>WebSocket Test Client</title>
</head>
<body>
  <h3>WebSocket Test</h3>
  <div>
    <label>WebSocket URL: <input id="url" value="ws://localhost:8080/ws" size="30"></label>
    <button id="connect">Connect</button>
    <button id="disconnect" disabled>Disconnect</button>
  </div>

  <div style="margin-top:1em;">
    <input id="msg" size="40" placeholder="Type message">
    <button id="send" disabled>Send</button>
  </div>

  <pre id="log" style="border:1px solid #ccc; padding:10px; height:300px; overflow:auto;"></pre>

  <script>
    let ws;
    const log = (t) => {
      const p = document.getElementById('log');
      p.textContent += t + '\n';
      p.scrollTop = p.scrollHeight;
    };

    document.getElementById('connect').onclick = () => {
      const url = document.getElementById('url').value;
      ws = new WebSocket(url);

      ws.onopen = () => {
        log('CONNECTED to ' + url);
        document.getElementById('send').disabled = false;
        document.getElementById('disconnect').disabled = false;
        document.getElementById('connect').disabled = true;
      };

      ws.onmessage = (evt) => log('RECV: ' + evt.data);
      ws.onclose = () => {
        log('DISCONNECTED');
        document.getElementById('send').disabled = true;
        document.getElementById('disconnect').disabled = true;
        document.getElementById('connect').disabled = false;
      };
      ws.onerror = (e) => log('ERROR: ' + e);
    };

    document.getElementById('send').onclick = () => {
      const m = document.getElementById('msg').value;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(m);
        log('SENT: ' + m);
        document.getElementById('msg').value = '';
      }
    };

    document.getElementById('disconnect').onclick = () => {
      if (ws) ws.close();
    };
  </script>
</body>
</html>
```

---

# How to run (quick)

1. `go mod init example.com/wsdemo` (if not already)
2. `go get github.com/gorilla/websocket`
3. `go run server.go`
4. Open `client.html` in your browser and click **Connect** (or use `wscat` / other client):

   * `wscat -c ws://localhost:8080/ws` (if you have `wscat`)

You should see a periodic server-time broadcast every 5 seconds. If you type anything in the client and send, the server broadcasts `recv: <your message>` to all connected clients.

---

# Notes & best practices

* **Production concerns**: check `Origin` in `Upgrader.CheckOrigin`, use TLS (wss://), apply authentication (e.g., cookies/JWT), and monitor resource usage.
* **Scaling**: a single-process hub works for moderate loads. For many connections or multiple machines, use message brokers (Redis Pub/Sub, NATS) or sticky sessions with a load balancer.
* **Backpressure**: client `send` channel is buffered. We drop clients if their buffers are full. You can make dropping behavior configurable (or use per-client write goroutine with blocking writes and removal).
* **Ping/Pong**: implemented to detect dead clients. Browser cleaners usually reply automatically to pings, but some libraries need explicit pong handling.

---
Nice — let’s build a clean, well-explained TCP server & client in Go. I’ll give you two practical examples you can copy/run:

1. A simple newline-delimited **echo server** + client (easy to test from terminal).
2. A more robust **concurrent server** with graceful shutdown and a **length-prefixed** message framing protocol (safer for binary data / predictable reads), plus a client that demonstrates both sending and receiving.

I’ll explain each piece as we go. Code is ready-to-run.

# Simple newline-delimited Echo Server (server.go)

This is the minimal, idiomatic TCP server: accepts connections, handles each connection concurrently, reads lines, echoes them back.

```go
// server_simple.go
package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"net"
)

func handleConn(conn net.Conn) {
	defer conn.Close()
	addr := conn.RemoteAddr().String()
	log.Println("connected:", addr)

	reader := bufio.NewReader(conn)
	writer := bufio.NewWriter(conn)

	for {
		// Read until newline
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				log.Println("client disconnected:", addr)
			} else {
				log.Println("read error:", err)
			}
			return
		}

		log.Printf("recv from %s: %q", addr, line)

		// Echo back
		_, err = writer.WriteString("echo: " + line)
		if err != nil {
			log.Println("write error:", err)
			return
		}
		writer.Flush()
	}
}

func main() {
	listenAddr := ":9000"
	ln, err := net.Listen("tcp", listenAddr)
	if err != nil {
		log.Fatalf("listen error: %v", err)
	}
	log.Println("echo server listening on", listenAddr)

	for {
		conn, err := ln.Accept()
		if err != nil {
			log.Println("accept error:", err)
			continue
		}
		go handleConn(conn) // handle each client concurrently
	}
}
```

## How to run

1. `go run server_simple.go`
2. In another terminal: `nc localhost 9000` (or `telnet localhost 9000`) then type lines. The server will reply prefixed with `echo: `.

## Explanation — key points

* `net.Listen("tcp", addr)` creates a TCP listener.
* `ln.Accept()` waits for an incoming connection and returns a `net.Conn`.
* Each connection gets a goroutine `go handleConn(conn)`, so server can handle many clients concurrently.
* `bufio.Reader.ReadString('\n')` reads until newline — good for human-readable text protocols.
* Always `defer conn.Close()` to free resources.

---

# Robust concurrent TCP server with graceful shutdown + length-prefixed framing

This version:

* Uses a length-prefixed protocol (4-byte big-endian length) so messages can contain arbitrary bytes (including newlines).
* Has a connection manager and a clean shutdown on SIGINT / SIGTERM.
* Broadcasts server messages to connected clients (example of sending messages from server side).
* Shows how to avoid blocking writes (per-connection send channel).

Files: `server.go` and `client.go` (below).

## Server (server.go)

```go
// server.go
package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

// writeMessage sends a length-prefixed message (4-byte uint32 BE length + payload)
func writeMessage(w io.Writer, data []byte) error {
	var lenBuf [4]byte
	binary.BigEndian.PutUint32(lenBuf[:], uint32(len(data)))
	if _, err := w.Write(lenBuf[:]); err != nil {
		return err
	}
	_, err := w.Write(data)
	return err
}

// readMessage reads a length-prefixed message
func readMessage(r io.Reader) ([]byte, error) {
	var lenBuf [4]byte
	if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(lenBuf[:])
	if n == 0 {
		return []byte{}, nil
	}
	data := make([]byte, n)
	_, err := io.ReadFull(r, data)
	return data, err
}

type client struct {
	conn net.Conn
	send chan []byte
	addr string
}

type server struct {
	ln        net.Listener
	clients   map[*client]struct{}
	mu        sync.Mutex
	wg        sync.WaitGroup
	broadcast chan []byte
}

func newServer() *server {
	return &server{
		clients:   make(map[*client]struct{}),
		broadcast: make(chan []byte, 128),
	}
}

func (s *server) addClient(c *client) {
	s.mu.Lock()
	s.clients[c] = struct{}{}
	s.mu.Unlock()
}

func (s *server) removeClient(c *client) {
	s.mu.Lock()
	delete(s.clients, c)
	s.mu.Unlock()
	close(c.send)
}

func (s *server) runBroadcaster(ctx context.Context) {
	for {
		select {
		case msg := <-s.broadcast:
			// broadcast to all clients (non-blocking)
			s.mu.Lock()
			for c := range s.clients {
				select {
				case c.send <- msg:
				default:
					// client's send buffer full: drop client to avoid blocking
					log.Println("dropping client (send buffer full):", c.addr)
					go func(cl *client) {
						cl.conn.Close()
					}(c)
				}
			}
			s.mu.Unlock()
		case <-ctx.Done():
			return
		}
	}
}

func (s *server) handleConn(ctx context.Context, conn net.Conn) {
	c := &client{
		conn: conn,
		send: make(chan []byte, 64),
		addr: conn.RemoteAddr().String(),
	}
	s.addClient(c)
	s.wg.Add(1)
	defer s.wg.Done()
	defer s.removeClient(c)
	defer conn.Close()

	log.Println("client connected:", c.addr)

	// writer goroutine
	writeDone := make(chan struct{})
	go func() {
		defer close(writeDone)
		for {
			select {
			case data, ok := <-c.send:
				if !ok {
					return
				}
				if err := writeMessage(conn, data); err != nil {
					log.Println("write error to", c.addr, err)
					return
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	// read loop
	for {
		msg, err := readMessage(conn)
		if err != nil {
			if err == io.EOF {
				log.Println("client disconnected:", c.addr)
			} else {
				log.Println("read error:", c.addr, err)
			}
			break
		}
		log.Printf("recv from %s: %s", c.addr, string(msg))
		// Example: reply to sender and broadcast to everyone
		reply := []byte(fmt.Sprintf("server-recv: %s", string(msg)))
		select {
		case c.send <- reply:
		default:
			log.Println("client send buffer full, dropping reply to", c.addr)
		}

		// broadcast message to all clients
		bcast := []byte(fmt.Sprintf("[%s] %s", c.addr, string(msg)))
		s.broadcast <- bcast
	}

	// wait for writer to finish
	conn.Close()
	<-writeDone
	log.Println("connection handler done for", c.addr)
}

func (s *server) ListenAndServe(addr string) error {
	var err error
	s.ln, err = net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	log.Println("server listening on", addr)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// broadcaster
	go s.runBroadcaster(ctx)

	// accept loop
	for {
		conn, err := s.ln.Accept()
		if err != nil {
			// If listener closed, return
			if opErr, ok := err.(*net.OpError); ok && opErr.Err.Error() == "use of closed network connection" {
				return nil
			}
			log.Println("accept error:", err)
			continue
		}
		go s.handleConn(ctx, conn)
	}
}

// Shutdown closes listener and waits for active handlers to finish with timeout.
func (s *server) Shutdown(timeout time.Duration) {
	log.Println("shutdown: closing listener")
	if s.ln != nil {
		s.ln.Close()
	}

	done := make(chan struct{})
	go func() {
		s.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		log.Println("shutdown: all handlers finished")
	case <-time.After(timeout):
		log.Println("shutdown: timeout waiting for handlers")
	}
}

func main() {
	srv := newServer()
	go func() {
		if err := srv.ListenAndServe(":9001"); err != nil {
			log.Println("server stopped:", err)
		}
	}()

	// example: server side periodic message (every 10s)
	go func() {
		t := time.NewTicker(10 * time.Second)
		for range t.C {
			srv.broadcast <- []byte("server-broadcast: heartbeat")
		}
	}()

	// graceful shutdown on SIGINT / SIGTERM
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	log.Println("got signal, shutting down...")
	srv.Shutdown(5 * time.Second)
	log.Println("server exit")
}
```

### Key design decisions & explanations

* **Length-prefixed framing**: `writeMessage` writes a 4-byte big-endian length followed by payload. `readMessage` reads exactly that many bytes. This avoids problems when payload contains newlines or binary.
* **Per-client send channel**: avoids concurrent writes to the same `net.Conn`. A single writer goroutine is responsible for writing to the network for that connection.
* **Backpressure / drop policy**: when `client.send` buffer is full we drop the client or drop the message to avoid blocking the whole server. You can change this policy depending on requirements.
* **Graceful shutdown**: `Shutdown` closes listener and waits for active handlers via `sync.WaitGroup`. We wait up to a timeout.
* **Broadcast channel**: the server has a central `broadcast` channel; any handler can push messages to it and `runBroadcaster` fan-outs to clients.
* **Context**: used to stop goroutines on server context cancellation (not fully used in this snippet but included pattern-wise).

---

## Client demonstrating the length-prefixed protocol (client.go)

```go
// client.go
package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"net"
	"os"
)

func writeMessage(w io.Writer, data []byte) error {
	var lenBuf [4]byte
	binary.BigEndian.PutUint32(lenBuf[:], uint32(len(data)))
	if _, err := w.Write(lenBuf[:]); err != nil {
		return err
	}
	_, err := w.Write(data)
	return err
}

func readMessage(r io.Reader) ([]byte, error) {
	var lenBuf [4]byte
	if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(lenBuf[:])
	if n == 0 {
		return []byte{}, nil
	}
	data := make([]byte, n)
	_, err := io.ReadFull(r, data)
	return data, err
}

func main() {
	conn, err := net.Dial("tcp", "localhost:9001")
	if err != nil {
		log.Fatal("dial error:", err)
	}
	defer conn.Close()
	fmt.Println("connected to server")

	// start a goroutine to read messages from server
	go func() {
		for {
			msg, err := readMessage(conn)
			if err != nil {
				if err == io.EOF {
					log.Println("server closed connection")
					os.Exit(0)
				}
				log.Println("read error:", err)
				os.Exit(1)
			}
			fmt.Println("RECV:", string(msg))
		}
	}()

	// read from stdin and send to server
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Print("Enter text> ")
	for scanner.Scan() {
		text := scanner.Text()
		if text == "quit" || text == "exit" {
			fmt.Println("exiting")
			return
		}
		if err := writeMessage(conn, []byte(text)); err != nil {
			log.Println("write error:", err)
			return
		}
		fmt.Print("Enter text> ")
	}

	if err := scanner.Err(); err != nil {
		log.Println("stdin error:", err)
	}
}
```

### How to run the robust example

1. `go run server.go`
2. `go run client.go` (in another terminal). Type lines and hit enter. Server will echo and broadcast messages. Type `exit` to quit the client.

---

# Common pitfalls & tips

* **Concurrent writes**: never write to the same `net.Conn` from multiple goroutines. Use a single writer goroutine or synchronize writes with a mutex.
* **Framing**: TCP is a stream — reads may return partial messages. Use a framing strategy: newline-delimited, length-prefix, or higher-level protocols (HTTP/gRPC).
* **Dead connections**: use application-level heartbeats or set deadlines (e.g., `conn.SetDeadline`) to detect dead peers.
* **TLS**: for production, wrap connections with `tls.Config` and `tls.Listen` / `tls.Dial` for secure transport.
* **Performance**: for very high connection counts consider tuning OS limits (ulimit), use `epoll`-backed network event loops (Go's runtime handles this) and evaluate resource usage.
* **Backpressure**: design what to do when clients are slow: buffer, block, or drop — each has trade-offs.
* **Timeouts**: set read & write deadlines to avoid goroutines stuck forever on dead sockets.

---

# Quick comparison of the two approaches

* **Newline-delimited**: great for simple human-facing text protocols. Easy to debug with `nc`. But fails if payload includes newlines or arbitrary binary.
* **Length-prefixed**: robust for arbitrary payloads and binary data. Slightly more code but production-ready.

---
Below is a **complete Go testing crash course** covering:

✔ Unit tests
✔ Table-driven tests
✔ Mocks / stubs / interfaces
✔ Testing HTTP handlers
✔ Testing goroutines and channels
✔ Using `testify` (optional but useful)

Everything is in *copy-paste runnable* snippets.

---

# ✅ 1. Basics of Go Testing (`testing` package)

## File structure

Go automatically runs all files ending with `_test.go`.

```
math/
 ├── math.go
 └── math_test.go
```

---

# 📌 Example 1 — Simple Unit Test

### `math.go`

```go
package mathutil

func Add(a, b int) int {
	return a + b
}
```

### `math_test.go`

```go
package mathutil

import "testing"

func TestAdd(t *testing.T) {
	got := Add(2, 3)
	want := 5

	if got != want {
		t.Errorf("Add(2,3) = %d; want %d", got, want)
	}
}
```

### Run tests

```
go test
```

---

# 📌 Example 2 — Table-Driven Tests (MOST COMMON pattern in Go)

### `math.go`

```go
func Multiply(a, b int) int {
	return a * b
}
```

### `math_test.go`

```go
func TestMultiply(t *testing.T) {
	tests := []struct {
		name     string
		a, b     int
		expected int
	}{
		{"positive", 3, 4, 12},
		{"zero", 5, 0, 0},
		{"negative", -2, 8, -16},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Multiply(tt.a, tt.b)
			if got != tt.expected {
				t.Fatalf("Multiply(%d,%d) got %d; expected %d",
					tt.a, tt.b, got, tt.expected)
			}
		})
	}
}
```

✔ Uses `t.Run()`
✔ Each test case is isolated
✔ Clean and scalable

---

# 📌 Example 3 — Testing Errors

### `divide.go`

```go
package mathutil

import "errors"

func Divide(a, b int) (int, error) {
	if b == 0 {
		return 0, errors.New("divide by zero")
	}
	return a / b, nil
}
```

### `divide_test.go`

```go
func TestDivide(t *testing.T) {
	_, err := Divide(10, 0)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}
```

---

# 🚀 2. Mocks in Go (with interfaces)

Go doesn’t have a “mock” system like Java/Python.
We **mock via interfaces + test implementations**.

---

# 📌 Example 4 — Interface + Mock for Testing

### `emailer.go`

```go
package notify

type EmailSender interface {
	SendEmail(to, msg string) error
}

type Notifier struct {
	Email EmailSender
}

func (n *Notifier) SendWelcome(to string) error {
	return n.Email.SendEmail(to, "Welcome!")
}
```

### Mock Implementation (`emailer_test.go`)

```go
package notify

import (
	"errors"
	"testing"
)

type MockEmailSender struct {
	SentTo string
	Msg    string
	Err    error
}

func (m *MockEmailSender) SendEmail(to, msg string) error {
	m.SentTo = to
	m.Msg = msg
	return m.Err
}

func TestNotifier_SendWelcome(t *testing.T) {
	mock := &MockEmailSender{}
	n := &Notifier{Email: mock}

	err := n.SendWelcome("test@example.com")
	if err != nil {
		t.Fatal(err)
	}

	if mock.SentTo != "test@example.com" {
		t.Errorf("expected email to %s, got %s", "test@example.com", mock.SentTo)
	}

	if mock.Msg != "Welcome!" {
		t.Errorf("expected message Welcome!, got %s", mock.Msg)
	}
}
```

✔ Mock stores values so we can assert
✔ No external services involved

---

# 📌 Example 5 — Mocking Errors

```go
func TestNotifier_Fails(t *testing.T) {
	mock := &MockEmailSender{Err: errors.New("smtp down")}
	n := &Notifier{Email: mock}

	err := n.SendWelcome("x@y.com")
	if err == nil {
		t.Fatal("expected error")
	}
}
```

---

# 📌 Example 6 — Testing HTTP Handlers (Super Important)

### `handler.go`

```go
package api

import (
	"encoding/json"
	"net/http"
)

type User struct {
	Name string `json:"name"`
}

func HelloHandler(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(User{Name: "Sandeep"})
}
```

### `handler_test.go`

```go
package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHelloHandler(t *testing.T) {
	req := httptest.NewRequest("GET", "/hello", nil)
	rr := httptest.NewRecorder()

	HelloHandler(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status code = %d, want %d", rr.Code, 200)
	}

	var u User
	json.NewDecoder(rr.Body).Decode(&u)

	if u.Name != "Sandeep" {
		t.Fatalf("expected \"Sandeep\", got %q", u.Name)
	}
}
```

✔ Uses `httptest.NewRecorder()` to capture response
✔ No server needed

---

# 📌 Example 7 — Testing Goroutines and Channels

### `worker.go`

```go
package worker

func SumAsync(a, b int, out chan int) {
	go func() {
		out <- (a + b)
	}()
}
```

### `worker_test.go`

```go
package worker

import "testing"

func TestSumAsync(t *testing.T) {
	ch := make(chan int)
	SumAsync(2, 3, ch)

	result := <-ch
	if result != 5 {
		t.Fatalf("expected 5, got %d", result)
	}
}
```

---

# 📌 Example 8 — Testing with Timeout (avoid goroutine leaks)

```go
func TestSumAsyncTimeout(t *testing.T) {
	ch := make(chan int)
	SumAsync(2, 3, ch)

	select {
	case v := <-ch:
		if v != 5 {
			t.Fatalf("expected 5, got %d", v)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("test timed out (goroutine stuck?)")
	}
}
```

---

# 📌 Example 9 — Using Testify (optional but common)

Install:

```
go get github.com/stretchr/testify/assert
```

### Test using `assert`

```go
import "github.com/stretchr/testify/assert"

func TestAdd(t *testing.T) {
	assert.Equal(t, 5, Add(2, 3))
}
```

### With mocks (using `testify/mock`)

```go
type EmailMock struct {
	mock.Mock
}

func (m *EmailMock) SendEmail(to, msg string) error {
	args := m.Called(to, msg)
	return args.Error(0)
}

func TestNotifier(t *testing.T) {
	mockEmail := new(EmailMock)
	mockEmail.On("SendEmail", "hello@example.com", "Welcome!").Return(nil)

	n := Notifier{Email: mockEmail}
	n.SendWelcome("hello@example.com")

	mockEmail.AssertExpectations(t)
}
```

---

# 📘 Summary

| Topic                   | Covered |
| ----------------------- | ------- |
| Unit Tests              | ✔       |
| Table Tests             | ✔       |
| Error Testing           | ✔       |
| Interface-based Mocks   | ✔       |
| HTTP Handler Testing    | ✔       |
| Goroutines & Channels   | ✔       |
| Timeouts & Race Issues  | ✔       |
| Testify (assert & mock) | ✔       |

---
Here’s a **clear, practical, beginner-friendly** guide to **benchmarking Go functions using `testing.B`**, followed by **multiple examples**, including tips and common pitfalls.

---

# ✅ **1. What is Benchmarking in Go?**

Benchmarking helps you *measure the performance* of your Go code.

Go’s testing package has built-in benchmark support.

A benchmark function:

* Must start with `Benchmark`
* Must accept a pointer `*testing.B`
* Uses a loop `for i := 0; i < b.N; i++ { ... }`
* Automatically adjusts `b.N` to get stable results

---

# ✅ **2. Basic Benchmark Example**

### **File: `math.go`**

```go
package mathx

func Add(a, b int) int {
    return a + b
}
```

### **File: `math_test.go`**

```go
package mathx

import "testing"

func BenchmarkAdd(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Add(5, 10)
    }
}
```

### Run the benchmark:

```
go test -bench=.
```

### Output (sample):

```
BenchmarkAdd-8     1000000000     0.266 ns/op
```

* **8** → number of CPU threads used
* **1000000000** → iterations run
* **0.266 ns/op** → nanoseconds per operation

---

# ✅ **3. Realistic Example: Benchmarking String Concatenation**

### File: `stringx.go`

```go
package stringx

func Concat(a, b string) string {
    return a + b
}
```

### File: `stringx_test.go`

```go
package stringx

import "testing"

func BenchmarkConcat(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Concat("hello", "world")
    }
}
```

Run:

```
go test -bench=Concat
```

---

# ✅ **4. Writing Multiple Benchmarks in the Same File**

```go
func BenchmarkAddSmall(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Add(1, 2)
    }
}

func BenchmarkAddLarge(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Add(100000000, 999999999)
    }
}
```

---

# ✅ **5. Table-Driven Benchmarks**

Same pattern as table-driven tests.

```go
func BenchmarkTable(b *testing.B) {
    cases := []struct {
        name string
        a, b int
    }{
        {"small", 1, 2},
        {"medium", 1000, 2000},
        {"large", 1000000, 2000000},
    }

    for _, tc := range cases {
        b.Run(tc.name, func(b *testing.B) {
            for i := 0; i < b.N; i++ {
                Add(tc.a, tc.b)
            }
        })
    }
}
```

Run:

```
go test -bench=BenchmarkTable -benchmem
```

---

# ✅ **6. Benchmark Memory Allocations**

Use `-benchmem`:

```
go test -bench=. -benchmem
```

Output:

```
BenchmarkConcat-8   50000000   25.0 ns/op   8 B/op   1 allocs/op
```

* **8 B/op** → bytes allocated per op
* **1 allocs/op** → allocations per op

This helps you optimize memory-heavy functions.

---

# ✅ **7. Benchmarking Functions With Setup (Use `b.ResetTimer`)**

If your function needs heavy setup, don’t include setup time in the benchmark.

### Example:

```go
func BenchmarkSliceAppend(b *testing.B) {
    b.StopTimer()
    base := make([]int, 0, 1000)
    b.StartTimer()

    for i := 0; i < b.N; i++ {
        _ = append(base, i)
    }
}
```

Or simpler:

```go
b.ResetTimer() // resets timing + allocation counters
```

---

# ✅ **8. Benchmarking Parallel Code**

Go supports benchmarking parallel workloads:

```go
func BenchmarkParallelAdd(b *testing.B) {
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            Add(10, 20)
        }
    })
}
```

Runs multiple goroutines to simulate concurrency.

---

# ✅ **9. Avoid Benchmark Pitfalls**

| Mistake                                       | Why it's wrong                        |
| --------------------------------------------- | ------------------------------------- |
| Doing I/O inside benchmark                    | IO is slow and noisy, ruins benchmark |
| Using random numbers                          | Randomness makes results inconsistent |
| Logging inside benchmark                      | Slows down dramatically               |
| Memory allocation inside loop unless required | Inflates allocations/op               |
| Using time.Sleep                              | Never do this inside a benchmark      |

---

# ✅ **10. Benchmark Example With Real Use Case: Sorting**

```go
func BenchmarkSort(b *testing.B) {
    for i := 0; i < b.N; i++ {
        data := []int{5, 3, 6, 1, 9, 8}
        sort.Ints(data)
    }
}
```

---
Here’s a **clear, simple, beginner-friendly** explanation of the **Go race detector**, with **buggy code**, the **race detector output**, and the **fixed version**.

---

# ✅ **1. What is the Go Race Detector?**

The race detector finds **data races**, which happen when:

* Two or more goroutines access the **same memory**
* At least one writes
* They run **without synchronization**

Run with:

```
go run -race main.go
```

Or for tests:

```
go test -race .
```

---

# ✅ **2. Buggy Code With a Data Race**

Here is a small Go program that *definitely* causes a race:

### **File: `main.go`**

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    count := 0 // shared variable

    for i := 0; i < 5; i++ {
        go func() {
            count++ // unsynchronized write
        }()
    }

    time.Sleep(time.Second)
    fmt.Println("Final count:", count)
}
```

### ❌ What’s wrong?

* Multiple goroutines write to `count`
* No mutex / sync mechanism
* The result will be inconsistent

---

# ✅ **3. Run the Race Detector**

```
go run -race main.go
```

### Sample Output:

```
==================
WARNING: DATA RACE
Write at 0x00c000014088 by goroutine 7:
  main.main.func1()
      /home/user/main.go:14 +0x44
Previous read at 0x00c000014088 by main goroutine:
  main.main()
      /home/user/main.go:18 +0x60
...
==================
Found 1 data race(s)
exit status 66
```

The race detector:

* Shows the exact line numbers
* Shows which goroutine did the writing
* Shows that the main goroutine read the same variable

---

# ✅ **4. Fixing the Data Race (Using Mutex)**

### **Corrected Code**

```go
package main

import (
    "fmt"
    "sync"
)

func main() {
    count := 0
    var mu sync.Mutex
    var wg sync.WaitGroup

    for i := 0; i < 5; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            mu.Lock()
            count++
            mu.Unlock()
        }()
    }

    wg.Wait()
    fmt.Println("Final count:", count)
}
```

### Run again:

```
go run -race main.go
```

Output:

```
Final count: 5
```

✔ No race
✔ Correct result
✔ Predictable behavior

---

# ✅ **5. Another Common Data Race Example (Reading & Writing)**

### **Buggy Code**

```go
package main

import "fmt"

var shared = 0

func main() {
    go func() {
        shared = 100
    }()

    fmt.Println(shared) // read without synchronization
}
```

Run:

```
go run -race main.go
```

Data race detected!

---

# 🛠 Fix with `sync/atomic`

```go
package main

import (
    "fmt"
    "sync/atomic"
)

var shared int64

func main() {
    atomic.StoreInt64(&shared, 100)
    fmt.Println(atomic.LoadInt64(&shared))
}
```

---
Below is a **clear, structured, in-depth explanation** of **Go modules, versioning, and how dependency resolution works internally** — from beginner level to internal algorithm-level details.

---

# 🟦 PART 1 — **What Are Go Modules?**

Go Modules are Go’s dependency management system (introduced in Go 1.11+).

A module is:

* A **folder** with source code
* Having a `go.mod` file
* Defined by a **module path** → usually a repo URL

Example:

```
module github.com/sandeep/myapp
go 1.22
```

Modules allow you to:

✔ Track dependencies
✔ Use semantic versions
✔ Reproducible builds
✔ Replace or vendor dependencies

---

# 🟦 PART 2 — **The go.mod File Explained**

A simple `go.mod`:

```go
module github.com/sandeep/myapp

go 1.22

require (
    github.com/gin-gonic/gin v1.10.0
    github.com/google/uuid v1.5.0
)
```

Sections include:

### **module**

* Identifies the module

### **go**

* The version of the Go language

### **require**

* Direct dependencies with exact versions

### **replace**

* Override dependencies

### **exclude**

* Prevent certain versions from being used

---

# 🟦 PART 3 — **The go.sum File**

`go.sum` contains cryptographic checksums of:

* Every dependency version
* Their transitive dependencies

Purpose:

✔ Ensure dependency integrity
✔ Prevent “supply-chain attacks”
✔ Allow reproducible builds

Go verifies checksums using the **Go checksum database** (proxy.golang.org).

---

# 🟦 PART 4 — **How Go Resolves Dependencies (Internal Mechanics)**

Here is the **internal dependency resolution algorithm**, simplified:

---

## 🔵 1. **Load your go.mod**

Go inspects:

* Direct dependencies
* Replace/exclude rules
* Go version

---

## 🔵 2. **Scan source imports**

Example:

```go
import "github.com/gin-gonic/gin"
```

If this package is not in your module:

→ Go attempts to find it in `go.mod`
→ If missing, Go resolves it

---

## 🔵 3. **Find the module containing the import**

The import path is resolved using:

* The Go Proxy → **proxy.golang.org**
* VCS tags
* The module's `go.mod`

Example:

`github.com/gin-gonic/gin` → module root at same path.

---

## 🔵 4. **Pick the version using Minimal Version Selection (MVS)**

This is the most important rule.

### ❗ Go uses MVS (Minimal Version Selection)

Meaning:

📌 **For each dependency, Go chooses the minimum version required by your module or its dependencies — but never downgrades a version already reached.**

Rules:

1. You need version `v1.5.0`
2. A dependency requires `v1.4.0`
3. Another indirect dependency wants `v1.3.0`

📌 Go picks **v1.5.0**, the highest required.

But Go will **not** upgrade beyond what's explicitly requested.

This ensures:

* Predictability
* Reproducible builds
* No surprise major upgrades

---

## 🔵 5. **Download module files**

Modules are downloaded into:

```
$GOMODCACHE (usually $GOPATH/pkg/mod)
```

Example folder:

```
~/go/pkg/mod/github.com/gin-gonic/gin@v1.10.0/
```

---

## 🔵 6. **Verify checksums (security)**

Go compares:

* Local checksum
* go.sum checksum
* Checksum database record

If mismatched → build fails.

---

## 🔵 7. **Compile using final dependency graph**

Only the resolved versions are used.

---

# 🟦 PART 5 — **Versioning Rules (Semantic Versioning in Go)**

Go strictly uses **SemVer**:

```
vMAJOR.MINOR.PATCH
```

### **MAJOR (v2, v3...)**

Breaking changes allowed.

Go has a special rule:

👉 A module can have a higher major version only by changing its module path.

Example:

```
github.com/gin-gonic/gin → v1.x.x
github.com/gin-gonic/gin/v2 → v2.x.x
```

Why?

Because Go treats major versions as different modules.

---

### **MINOR (v1.3 → v1.4)**

* New features, no breaking changes
* Safe upgrade

---

### **PATCH (v1.4.1 → v1.4.2)**

* Bug fixes
* No new features

---

# 🟦 PART 6 — **Internal Working of Go Module Proxy**

The Go toolchain uses:

✔ proxy.golang.org
✔ sum.golang.org (checksum DB)

When you run:

```
go get github.com/gin-gonic/gin@v1.10.0
```

Go performs:

1. Resolve version
2. Fetch archive from proxy
3. Store in module cache
4. Verify checksum
5. Update `go.mod` and `go.sum`

---

# 🟦 PART 7 — **How `go get` Changes Dependencies**

### Add new version:

```
go get example.com/lib@v1.2.0
```

### Downgrade version:

```
go get example.com/lib@v1.0.0
```

### Upgrade all modules:

```
go get -u ./...
```

### Tidy unused dependencies:

```
go mod tidy
```

---

# 🟦 PART 8 — **replace and exclude (Advanced)**

### replace

Used to:

* Override modules
* Use local copy
* Patch bugs

Example:

```go
replace github.com/foo/bar => ../bar
```

---

### exclude

Prevents specific versions:

```go
exclude github.com/foo/bar v1.9.0
```

Useful when a version is broken.

---

# 🟦 PART 9 — **How Go Rebuilds the Module Graph (Detailed)**

When you run:

```
go list -m all
```

Go performs:

1. Read your `go.mod`
2. Resolve direct dependencies
3. For each dependency:

   * Load its go.mod
   * Resolve its dependencies
4. Apply MVS (minimal version selection)
5. Produce final build list

This list is deterministic.

---

# 🟦 PART 10 — Example Flow

Your module depends on:

```
A v1.0.0
B v1.0.0
```

A’s go.mod requires **B v1.2.0**

Final graph:

```
A v1.0.0
B v1.2.0
```

Why?

Because **B must be ≥ highest required version**.

---
Below is a **clear, practical, and production-ready guide** on how to organize a **large Go project using Clean Architecture**.
This is the exact structure used in real-world enterprise Go services.

---

# ✅ **Clean Architecture for Large Go Projects**

Clean Architecture splits your code into independent layers:

```
[ Presentation / Interface ]  →  [ Application / Use-cases ]  →  [ Domain ]  →  [ Infrastructure ]
```

* **Domain** = business rules, entities, interfaces
* **Usecases** = pure application logic
* **Interface/Handlers** = HTTP, gRPC, CLI, WebSocket, worker, etc.
* **Infrastructure** = DB, external APIs, frameworks, drivers

The **outer layers depend on inner layers**, NEVER the reverse.

---

# 📁 **1. Folder Structure (Industry Standard)**

```
project/
│
├── cmd/
│   └── api/
│       └── main.go
│
├── internal/
│   ├── domain/
│   │   ├── user/
│   │   │   ├── entity.go
│   │   │   └── repository.go
│   │   └── common/
│   │       └── errors.go
│   │
│   ├── usecase/
│   │   └── user/
│   │       ├── create_user.go
│   │       ├── get_user.go
│   │       └── service.go
│   │
│   ├── interface/
│   │   ├── http/
│   │   │   ├── user_handler.go
│   │   │   └── router.go
│   │   └── grpc/
│   │
│   └── infrastructure/
│       ├── db/
│       │   ├── pg_user_repo.go
│       │   └── connection.go
│       ├── logger/
│       └── env/
│
├── pkg/   (optional shared utilities)
└── go.mod
```

---

# 🚀 Let's Build a Feature: **User Management**

We'll implement:

* Domain entity
* Repository interface
* Use case service
* HTTP handler
* PostgreSQL repository

---

# 🌱 2. **Domain Layer**

Pure business objects, no frameworks.

### **internal/domain/user/entity.go**

```go
package user

type User struct {
    ID    int64
    Name  string
    Email string
}
```

### **internal/domain/user/repository.go**

```go
package user

type Repository interface {
    Create(u *User) error
    GetByID(id int64) (*User, error)
}
```

👉 This interface is used by the use case.
👉 The database will implement this interface.

---

# ⚙️ 3. **Use Case Layer**

Business workflows (CreateUser, GetUser).

### **internal/usecase/user/service.go**

```go
package user

import "project/internal/domain/user"

type Service struct {
    repo user.Repository
}

func NewService(r user.Repository) *Service {
    return &Service{repo: r}
}
```

### **internal/usecase/user/create_user.go**

```go
func (s *Service) CreateUser(name, email string) error {
    u := &user.User{Name: name, Email: email}
    return s.repo.Create(u)
}
```

### **internal/usecase/user/get_user.go**

```go
func (s *Service) GetUser(id int64) (*user.User, error) {
    return s.repo.GetByID(id)
}
```

**Usecases depend only on domain interfaces.**

---

# 🌐 4. **Interface Layer (HTTP)**

### **internal/interface/http/user_handler.go**

```go
package http

import (
    "encoding/json"
    "net/http"
    "strconv"

    "project/internal/usecase/user"
)

type UserHandler struct {
    service *user.Service
}

func NewUserHandler(s *user.Service) *UserHandler {
    return &UserHandler{service: s}
}

func (h *UserHandler) CreateUser(w http.ResponseWriter, r *http.Request) {
    var req struct {
        Name  string `json:"name"`
        Email string `json:"email"`
    }
    json.NewDecoder(r.Body).Decode(&req)

    if err := h.service.CreateUser(req.Name, req.Email); err != nil {
        w.WriteHeader(http.StatusInternalServerError)
        return
    }

    w.WriteHeader(http.StatusCreated)
}

func (h *UserHandler) GetUser(w http.ResponseWriter, r *http.Request) {
    idStr := r.URL.Query().Get("id")
    id, _ := strconv.ParseInt(idStr, 10, 64)

    user, err := h.service.GetUser(id)
    if err != nil {
        w.WriteHeader(http.StatusInternalServerError)
        return
    }

    json.NewEncoder(w).Encode(user)
}
```

### **internal/interface/http/router.go**

```go
func RegisterRoutes(mux *http.ServeMux, h *UserHandler) {
    mux.HandleFunc("/users", h.CreateUser)
    mux.HandleFunc("/user", h.GetUser)
}
```

---

# 🧱 5. **Infrastructure Layer (PostgreSQL)**

### **internal/infrastructure/db/pg_user_repo.go**

```go
package db

import (
    "database/sql"
    "project/internal/domain/user"
)

type PgUserRepo struct {
    DB *sql.DB
}

func NewPgUserRepo(db *sql.DB) *PgUserRepo {
    return &PgUserRepo{DB: db}
}

func (r *PgUserRepo) Create(u *user.User) error {
    _, err := r.DB.Exec("INSERT INTO users(name, email) VALUES ($1,$2)", u.Name, u.Email)
    return err
}

func (r *PgUserRepo) GetByID(id int64) (*user.User, error) {
    row := r.DB.QueryRow("SELECT id, name, email FROM users WHERE id=$1", id)
    var u user.User
    if err := row.Scan(&u.ID, &u.Name, &u.Email); err != nil {
        return nil, err
    }
    return &u, nil
}
```

---

# 🚀 6. **Wire Everything Together in `cmd/api/main.go`**

### **cmd/api/main.go**

```go
package main

import (
    "database/sql"
    "log"
    "net/http"

    _ "github.com/lib/pq"

    httpif "project/internal/interface/http"
    "project/internal/infrastructure/db"
    usecase "project/internal/usecase/user"
)

func main() {
    conn, err := sql.Open("postgres", "postgres://user:pass@localhost/dbname?sslmode=disable")
    if err != nil { panic(err) }

    repo := db.NewPgUserRepo(conn)
    service := usecase.NewService(repo)
    handler := httpif.NewUserHandler(service)

    mux := http.NewServeMux()
    httpif.RegisterRoutes(mux, handler)

    log.Println("Server running on :8080")
    http.ListenAndServe(":8080", mux)
}
```

---

# 🎯 Why This Architecture Works

| Benefit       | Explanation                                                      |
| ------------- | ---------------------------------------------------------------- |
| **Testable**  | You can mock the repo for use-case tests.                        |
| **Decoupled** | Database, HTTP, gRPC can be swapped without touching core logic. |
| **Scalable**  | New features are added as new domain + usecases + handlers.      |
| **Clean**     | Code has clear responsibility boundaries.                        |

---
Here is a **clear, practical guide to debugging Go programs using Delve (dlv)** with **sample commands**, **examples**, and **real-world workflows**.
This will help you debug CLI apps, servers, and concurrent programs.

---

# 🧰 **What is Delve (dlv)?**

**Delve** is the official Go debugger.
It lets you:

* Set breakpoints
* Step through code
* Inspect variables
* Debug goroutines
* Debug running servers
* Debug tests
* Attach to running processes

---

# ✅ **1. Install Delve**

```bash
go install github.com/go-delve/delve/cmd/dlv@latest
```

Binary path:

```
~/go/bin/dlv
```

---

# 🔍 **2. Run a Go Program Under Debugger**

Example Go file: `main.go`

```go
package main

import "fmt"

func main() {
    x := 10
    y := 20
    z := x + y
    fmt.Println("Result:", z)
}
```

### Run using Delve:

```bash
dlv debug
```

Delve starts an interactive debug session:

```
(dlv)
```

---

# 🎯 **3. Common Delve Commands**

| Command            | Meaning                   |
| ------------------ | ------------------------- |
| `break main.main`  | Set breakpoint            |
| `break main.go:10` | Break at line             |
| `break myFunc`     | Break at function         |
| `continue` or `c`  | Run until next breakpoint |
| `step` or `s`      | Step into function        |
| `next` or `n`      | Step over                 |
| `stepout`          | Step out                  |
| `print x`          | Print variable            |
| `locals`           | Show local variables      |
| `args`             | Function args             |
| `goroutines`       | List goroutines           |
| `goroutine <id>`   | Switch goroutine          |
| `quit` or `q`      | Exit debugger             |

---

# 🎯 **4. Setting Breakpoints**

### Break at function:

```bash
(dlv) break main.main
```

### Break at a specific line:

```bash
(dlv) break main.go:7
```

### Break conditionally:

```bash
(dlv) break main.go:10 if x == 5
```

### Show breakpoints:

```bash
(dlv) breakpoints
```

---

# 👣 **5. Stepping Through Code**

### Step into function:

```bash
(dlv) step
```

### Step over:

```bash
(dlv) next
```

### Step out:

```bash
(dlv) stepout
```

---

# 🔎 **6. Inspecting Variables**

Inside the debugger:

### Print single variable:

```bash
(dlv) print x
```

### Print struct fields:

```bash
(dlv) print user
(dlv) print user.Name
```

### View all locals:

```bash
(dlv) locals
```

### Evaluate expression:

```bash
(dlv) print x + y * 2
```

---

# 🚦 **7. Debugging a Web Server**

Suppose your server starts in `cmd/api/main.go`.

```bash
dlv debug ./cmd/api
```

Set a breakpoint in handler:

```bash
(dlv) break handler.go:25
```

Continue execution:

```bash
(dlv) continue
```

Send request:

```
curl http://localhost:8080/user?id=1
```

Debugger stops at breakpoint → inspect variables.

---

# 🧵 **8. Debugging Goroutines**

### List goroutines:

```bash
(dlv) goroutines
```

### Switch to a goroutine:

```bash
(dlv) goroutine 15
```

### View goroutine stack trace:

```bash
(dlv) goroutine 15 bt
```

Extremely useful for debugging concurrency issues.

---

# 🧪 **9. Debugging Tests with Delve**

Run tests interactively:

```bash
dlv test
```

Debug a specific test file:

```bash
dlv test ./service
```

Debug a test and stop at specific breakpoint:

```bash
dlv test -- -test.run TestUserService
```

---

# 🛠️ **10. Attach Delve to a Running Process**

Find Go process:

```bash
ps aux | grep myserver
```

Suppose PID = **4523**

Attach:

```bash
sudo dlv attach 4523
```

Useful for debugging staging/production replicas.

---

# 🏹 **11. Use Delve with VS Code**

Install extension:

```
Go — by Google
```

Launch debug config (`.vscode/launch.json`):

```json
{
  "name": "Debug Go",
  "type": "go",
  "request": "launch",
  "program": "${workspaceFolder}/cmd/api"
}
```

Press **F5** → breakpoints inside editor work.

---

# 🔥 **12. Use Delve in Headless Mode (API mode)**

For remote debugging:

```bash
dlv debug --headless --listen=:2345 --api-version=2
```

Connect from VS Code or GoLand.

---

# 🧪 Full Sample Session

```
$ dlv debug
Type 'help' for list of commands.

(dlv) break main.go:8
Breakpoint 1 set at main.go:8

(dlv) continue
> main.main() ./main.go:8 (hits total:1)

(dlv) print x
10

(dlv) next
(dlv) print z
30

(dlv) quit
```

---
Below is a **practical, real-project style guide** for using **`go vet`**, **`golint`**, and **`staticcheck`** the way professional Go teams use them in CI/CD, large repositories, and daily development.

---

# ✅ **Overview**

| Tool            | Purpose                                                                                   | Type                      |
| --------------- | ----------------------------------------------------------------------------------------- | ------------------------- |
| **go vet**      | Catches suspicious code likely to be bugs (logic issues, misuse of fmt, unreachable code) | Built-in Go analyzer      |
| **golint**      | Checks style issues (naming, comments, formatting)                                        | Deprecated but still used |
| **staticcheck** | Advanced static analysis (best practices, unused code, performance, correctness)          | Modern linting suite      |

---

# 📁 **Example Project Structure**

```
myapp/
├── cmd/
│   └── api/
│       └── main.go
├── internal/
│   ├── handler/
│   ├── service/
│   └── repo/
└── go.mod
```

The commands work on individual packages or entire modules.

---

# 🔍 **1. Using `go vet` in a Real Project**

`go vet` finds **actual bugs**, not style issues.

### ✔️ Install (already built-in):

No installation needed.

### ✔️ Example: Run on whole module

```bash
go vet ./...
```

### Sample output:

```
handler/user_handler.go:42: Printf format %d has arg name of wrong type string
```

### ✔️ Run on a specific package:

```bash
go vet ./internal/service
```

### ✔️ Common bugs `go vet` catches:

* Incorrect `fmt.Printf` formatting
* Copying mutex values
* Methods with value-receiver vs pointer mismatch
* Unreachable code
* Misuse of `context.CancelFunc`
* Shadowed variables

---

# 🎨 **2. Using `golint`**

⚠️ **Note:** `golint` is deprecated, but still widely used in older codebases for style checking.

### ✔️ Install:

```bash
go install golang.org/x/lint/golint@latest
```

### ✔️ Run on entire module:

```bash
golint ./...
```

### Sample output:

```
internal/service/user_service.go:5:6: exported type UserService should have comment or be unexported
```

### ✔️ Run on a single file:

```bash
golint internal/repo/user_repo.go
```

### What `golint` checks:

* Exported functions must have comments
* Naming conventions (CamelCase)
* Package comments
* Dead code patterns
* Stuttering (`user.UserService`)

---

# 🧠 **3. Using `staticcheck`**

This is the **most powerful** linting tool in Go.

### ✔️ Install:

```bash
go install honnef.co/go/tools/cmd/staticcheck@latest
```

### ✔️ Run on your project:

```bash
staticcheck ./...
```

### Sample output:

```
internal/service/user_service.go:77:2: SA4006: this value is never used
internal/handler/user_handler.go:25: http.Request.Body must be closed (SA2002)
```

### What `staticcheck` catches:

* Dead code
* Goroutine leaks
* Inefficient allocations
* Bugs with maps/slices
* Incorrect error handling
* Missed `context` cancellations
* Missed timeouts
* Unused variables/functions
* Magic constants
* API misuse

---

# 🧩 **Example: Real Project Workflow**

Let's say your project has a bug and style issues.

### File: `internal/service/user.go`

```go
func GetUserName(id int) (string, error) {
    user, err := repo.FindByID(id)
    fmt.Printf("User found: %d", user.Name)   // BUG: wrong format verb
    return user.Name, err
}
```

### Run `go vet`:

```bash
go vet ./...
```

Output:

```
Printf format %d has arg user.Name of wrong type string
```

---

### File: `internal/service/order.go`

```go
func process(i int) (j int) {
    j = i + 2
    return i // BUG: returning wrong variable
}
```

### Run `staticcheck`:

```bash
staticcheck ./...
```

Output:

```
SA4001: returning the result of a redundant operation
```

---

### File: `internal/handler/http.go`

```go
type Api struct {}
```

### Run `golint`:

```bash
golint ./...
```

Output:

```
exported type Api should have comment or be unexported
```

---

# 🎯 **4. Combine All Tools in a Makefile**

### **Makefile**

```makefile
.PHONY: vet lint staticcheck fmt test build all

vet:
	go vet ./...

lint:
	golint ./... || true

staticcheck:
	staticcheck ./...

fmt:
	go fmt ./...

test:
	go test ./...

build:
	go build ./cmd/api

all: vet lint staticcheck test build
```

Then run:

```bash
make all
```

---

# 🚀 **5. Add These Tools to CI (GitHub Actions)**

### `.github/workflows/lint.yml`

```yaml
name: Go Linting

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Go
        uses: actions/setup-go@v4
        with:
          go-version: '1.22'

      - name: Run vet
        run: go vet ./...

      - name: Install staticcheck
        run: go install honnef.co/go/tools/cmd/staticcheck@latest

      - name: Run staticcheck
        run: staticcheck ./...

      - name: Install golint (optional)
        run: go install golang.org/x/lint/golint@latest

      - name: Run golint
        run: golint ./...
```

---
Here is a **simple, clear, and diagram-rich explanation** of **how the Go Garbage Collector (GC)** works.
The diagrams are **ASCII-style** so you can easily visualize the internals.

---

# 🧠 **Go’s Garbage Collector: High-Level View**

Go uses:

### **• A concurrent**,

### **• tri-color**,

### **• mark-and-sweep garbage collector**,

### **• with a write barrier**,

### **• designed for low pauses (~<1ms).**

It runs **alongside your program (goroutines)** and only pauses briefly.

---

# 🧩 **1. Memory Layout Diagram**

```
+-------------------------------+
|            HEAP               |
|  (objects allocated via new)  |
+-------------------------------+

+-------------------------------+
|            STACKS             |
|  (goroutine stack frames)     |
+-------------------------------+

Program roots = { global vars, stacks, CPU registers }
```

The **GC starts at the roots** and traces reachable heap objects.

---

# 🎨 **2. Tri-Color GC Model (Core Concept)**

Objects are divided into 3 colors:

```
WHITE   = Unvisited (candidate for deletion)
GRAY    = Reachable but children not scanned yet
BLACK   = Reachable and fully scanned
```

### Diagram:

```
        +-------------+       +------------+       +-----------+
        |   WHITE     | --->  |   GRAY     | --->  |   BLACK   |
        |  (dead?)    |       |  (pending) |       |  (live)   |
        +-------------+       +------------+       +-----------+
```

At the end of GC:

```
ALL WHITE OBJECTS = garbage (freeable)
```

---

# ⚙️ **3. Step-by-Step Go GC Cycle**

## **Step 1 — Mark Start (brief stop-the-world)**

Go pauses the world for ~0.1–1ms:

* It stops goroutines momentarily.
* It identifies **root references**.

📌 Diagram:

```
ROOTS:
+---------+    +----------+
| Global  |    | Stacks   |
| Vars    |    | (goroutines)
+---------+    +----------+
```

All root-reachable objects become **gray**.

---

## **Step 2 — Marking (concurrent)**

Goroutines resume.

The GC works in the background:

```
[GRAY objects] → scan → turn BLACK → mark children GRAY
```

Diagram:

```
    (Gray Queue)
         |
         v
     +--------+         +-----------+
     |  GRAY  | ----->  |  BLACK    |
     | object |         | scanned   |
     +--------+         +-----------+
          |
          | marks
          v
     +----------+
     | children |
     |   GRAY   |
     +----------+
```

---

## **Step 3 — Write Barrier (while marking)**

When your program modifies a pointer, Go intercepts it.

### Why?

To maintain tri-color invariants.

### Diagram (simplified):

```
App write: A.black → B.white

Write barrier:
    mark B as GRAY
```

---

## **Step 4 — Sweep Phase (background)**

After marking, we know:

* **BLACK = live**
* **WHITE = garbage**

Diagram:

```
HEAP BLOCK:
+---------+---------+---------+
| BLACK   | WHITE   | WHITE   |  ← white blocks freed
+---------+---------+---------+
```

Go returns memory to free lists.

---

# 🧠 **4. Visualization of an Entire GC Cycle**

```
START GC
   |
   v
[Root Scan] (STW small pause)
   |
   v
[Mark: gray → black]  (concurrent)
   |
   +-- write barrier keeps objects gray when needed
   |
   v
[Mark Done] (STW small pause)
   |
   v
[Sweep unmarked objects]
   |
   v
END GC
```

---

# 🕒 **5. GC Pacing (How Go decides when to run GC)**

Go tries to keep:

```
heap_growth / live_heap <= GOGC (default 100%)
```

Meaning:

* If live objects = 100 MB
* GC tries to run before heap > 200 MB

### Diagram:

```
live heap = 100MB
       |
       |    GC runs here
       v
heap reaches ~200MB
```

You can tune using:

```bash
GOGC=50   # more frequent GC
GOGC=200  # less frequent GC
GOGC=off  # disable GC
```

---

# ⚡ **6. Go Garbage Collector Timeline Diagram**

```
Time -->
+--------+--------------------+---------+
|  STW   |  concurrent mark   |   STW   |
| (short)|  + write barrier   | (short) |
+--------+--------------------+---------+
                     \
                      +--- concurrent sweep
```

---

# 🔬 **7. Example With Objects**

Suppose you have:

```
A → B → C
D  (unreachable)
```

Initial state:

```
A,B,C = reachable
D = unreachable

colors:
A,B,C = WHITE
D = WHITE
```

Mark roots:

```
A = GRAY
```

Scan A:

```
A = BLACK
B = GRAY
```

Scan B:

```
B = BLACK
C = GRAY
```

Scan C:

```
C = BLACK
```

Final:

```
BLACK: A, B, C
WHITE: D  (deleted)
```

Sweep removes D.

---

# 🎯 **8. Why Go’s GC Is Fast**

* No full stop-the-world cycle
* Uses write barrier
* Incremental + concurrent mark
* Small pause times
* Parallel sweep
* Efficient span + bitmap memory layout

---

# 🧩 **9. Go GC in One Diagram**

```
        +------------------------------+
        |    Concurrent Mark-Sweep GC  |
        +------------------------------+
        | 1. Small STW: root marking   |
        | 2. Concurrent mark (gray→black)
        | 3. Write barrier maintains invariants
        | 4. Small STW: mark termination
        | 5. Concurrent sweep (free white)
        +------------------------------+
```

---
Below is a **clear, deep, internal explanation** of **how slices and maps work in Go**, including **memory layout diagrams**, **growth behavior**, and **performance implications** so you can design faster Go systems.

---

# 🧩 **1. How Slices Work Internally**

A **slice** is *not* an array.
It is a **descriptor** (tiny struct) that *points* to an underlying array.

### 📦 Internal Structure of a Slice (Go runtime)

Equivalent to:

```go
type slice struct {
    ptr *element   // pointer to backing array
    len int        // number of visible elements
    cap int        // total capacity before reallocation
}
```

### 🖼 Diagram

```
slice:      +-----------------------+
            | ptr | len | cap       |
            +-----------------------+
                 |
                 v
backing array:  [0] [1] [2] [3] [4] [5] ...
```

---

# 🔁 **Slice Growth Behavior**

When you append:

```go
s = append(s, x)
```

If `len < cap`:
✔ fast, no allocation

If `len == cap`:
❗ array resize happens

### Growth rules:

* For small slices (< 1024 elements): **double capacity**
* After ~1024 elements: grows by **~25%**

### Diagram: Resize

```
Before:
slice: [ptr → A] len=4 cap=4
array A: [1 2 3 4]

append(5) → new array

After:
slice: [ptr → B] len=5 cap=8
array B: [1 2 3 4 5 _ _ _]
```

### ❗ Implication

Appending can cause:

* New allocation
* Full copy of existing elements
* Old array becomes garbage

👉 Avoid repeated reallocation inside loops.

---

# ⚡ **Performance Tips for Slices**

### 1️⃣ Pre-allocate if possible

```go
s := make([]int, 0, 1000)
```

Avoids multiple allocations.

---

### 2️⃣ Slicing shares the same array

```go
a := make([]int, 5) // [0 0 0 0 0]
b := a[:3]
c := a[2:5]
```

Diagram:

```
a: [0 0 0 0 0]
b: [0 0 0]
c:       [0 0 0]
```

They share memory.

### ❗ Implication

Changing one affects all.

---

### 3️⃣ Slicing does **not** copy — but appending *might*

If you append to a slice that still has capacity, it modifies the same array:

```
a → [1 2 3 _ _]
b := a[:3]
b = append(b, 99)   // uses same backing array
```

Both change:

```
a → [1 2 3 99 _]
```

If capacity is exceeded → append makes a copy.
This often causes subtle bugs.

---

### 4️⃣ Copying a slice copies elements

```go
copy(dst, src)
```

Good for avoiding unintended sharing.

---

### 5️⃣ Slices of structs vs slices of pointers

* `[]T` → better locality, but expensive to copy if T is big
* `[]*T` → lower memory use, but more cache misses

---

# 🧩 **2. How Maps Work Internally**

Go maps use a **hash table with buckets**, optimized for speed and concurrency safety (single goroutine access).

### 🧱 Map Internal Structure

A map internally looks like this:

```go
type hmap struct {
    count     int
    B         uint8      // log₂(number of buckets)
    buckets   *bmap      // pointer to array of buckets
    oldbuckets *bmap     // during grow
    ...
}
```

### 📦 Bucket Structure

Each **bucket** stores:

* up to **8 key/value pairs**
* a **tophash array** for fast rejection

Diagram:

```
bucket:
+------------+------------+
| tophash[8] | key/value  |
+------------+------------+
```

### Overall diagram:

```
map → buckets:
            +---------+
hash(k1) →  | bucket0 | → next (overflow)
            +---------+
hash(k2) →  | bucket1 |
            +---------+
hash(k3) →  | bucket2 |
            +---------+
```

---

# 🔑 **How a Key Lookup Works**

1. Compute hash
2. Select bucket:
   `bucketIndex = hash & ((1<<B) - 1)`
3. Scan bucket’s tophash (top 8 bits)
4. Compare keys if tophash matches
5. If not found → scan overflow bucket

---

# 🚀 **Map Growth (Resizing)**

Maps grow gradually:

* When the load factor exceeds ~6.5 (bucket fullness)
* Go doubles number of buckets: `B += 1`
* Rehashing is **incremental**

### Diagram: Incremental Rehash

```
oldbuckets → gradually copied → buckets
```

No huge pause like full rehashing.

---

# 🧨 **Performance Implications of Maps**

### 1️⃣ Map operations are not constant-time always

Because:

* collisions
* overflow buckets
* incremental rehashing

---

### 2️⃣ Too many collisions → slow

If hash function is bad or key types are complex:

```
bucket → overflow → overflow → overflow
```

lookup becomes `O(n)`.

---

### 3️⃣ Pre-allocating maps helps a lot

```go
m := make(map[string]int, 10000)
```

This creates enough buckets upfront → fewer grows.

---

### 4️⃣ Maps do not maintain order

Iteration is randomized (for security).

---

### 5️⃣ Maps are **not safe for concurrent writes**

```go
fatal error: concurrent map writes
```

Because map resize + mutation is dangerous.

Use `sync.Map` or `sync.Mutex`.

---

### 6️⃣ Storing large objects = cache inefficiency

Prefer map of pointers for large structs:

```
map[string]*User   // faster
map[string]User    // copies whole struct
```

---

# 🧩 Summary Table

| Feature           | Slice                | Map                      |
| ----------------- | -------------------- | ------------------------ |
| Backing structure | Array                | Hash table + buckets     |
| Grows?            | Yes (copy + realloc) | Yes (incremental rehash) |
| Sharing memory?   | Yes                  | No                       |
| Concurrency safe? | No                   | No (writes)              |
| Lookup complexity | O(1)                 | O(1) avg, O(n) worst     |
| Best usage        | Sequences            | Key/value fast lookup    |

---
Here is the **clearest and most practical explanation of `context.Context` in Go**, followed by **real diagrams** and **a complete cancellation example**.

---

# ✅ **What is `context.Context`?**

`context.Context` is Go’s built-in mechanism for:

### ✔ Cancelling goroutines

### ✔ Setting deadlines / timeouts

### ✔ Passing request-scoped values

Contexts are **immutable** and form a **tree**.
When a parent context is cancelled, all its children get cancelled **instantly**.

---

# 📌 **Context Tree Diagram (How it Actually Works)**

```
context.Background()
        |
        +-- ctxA (WithCancel)
        |         |
        |         +-- ctxA1 (child goroutine)
        |         +-- ctxA2 (child goroutine)
        |
        +-- ctxB (WithTimeout)
                  |
                  +-- ctxB1 (DB Query)
```

If `ctxA` is cancelled → **ctxA1 and ctxA2 receive the cancellation signal**.
`ctxB` is unaffected because it’s a different branch.

---

# 🧠 **Key Attributes of a Context**

| Method       | Meaning                                                   |
| ------------ | --------------------------------------------------------- |
| `Done()`     | Returns a channel closed on cancellation                  |
| `Err()`      | Returns reason: `context.Canceled` or `deadline exceeded` |
| `Deadline()` | Time when context will cancel automatically               |
| `Value()`    | Used for request-scoped data (but avoid over-use)         |

---

# 🔥 **Types of Contexts**

### 1. **context.Background()**

Root context → empty / never cancelled.

### 2. **context.TODO()**

Use when you *don’t yet know* what context to use.

### 3. **context.WithCancel(parent)**

Manual cancellation.

### 4. **context.WithTimeout(parent, time.Duration)**

Auto-cancel after duration.

### 5. **context.WithDeadline(parent, time.Time)**

Auto-cancel at specific time.

### 6. **context.WithValue(parent, key, value)**

Attach request-scoped values.

---

# 🚀 **Build a Cancellation Example (Simple & Clear)**

We’ll create:

* A parent context with cancellation
* A worker goroutine that listens for cancellation
* We cancel after 2 seconds

---

# ✅ **Complete Working Go Code (with comments)**

```go
package main

import (
	"context"
	"fmt"
	"time"
)

func worker(ctx context.Context) {
	for {
		select {
		case <-ctx.Done(): // cancellation signal
			fmt.Println("Worker stopped:", ctx.Err())
			return
		default:
			fmt.Println("Working...")
			time.Sleep(500 * time.Millisecond)
		}
	}
}

func main() {
	// Create cancellable context
	ctx, cancel := context.WithCancel(context.Background())

	// Start worker
	go worker(ctx)

	// Run for 2 seconds
	time.Sleep(2 * time.Second)

	// Cancel context
	fmt.Println("Cancelling context...")
	cancel()

	// Wait to show output
	time.Sleep(1 * time.Second)
}
```

---

# 🖼 **Flow Diagram of This Example**

```
main()
  |
  |-- ctx, cancel := WithCancel
  |
  |-- start worker goroutine
  |
  |-- sleep 2s
  |
  |-- cancel() --------------------------+
                                          |
worker():                                 |
  loop:                                    |
    default: "Working..." (500ms)         |
    <-ctx.Done()  <-----------------------+
         print: "Worker stopped: context canceled"
```

---

# ⛔ Why You MUST Use Context in Real Projects

Contexts are essential for:

### ✔ HTTP servers

Cancel requests when client disconnects.

### ✔ Database queries

Timeouts prevent hanging queries.

### ✔ Microservices

Propagate cancellation across service calls.

### ✔ Preventing goroutine leaks

Contexts avoid goroutines running forever.

---

# 🎯 **Bonus: Context Timeout Example**

```go
ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
defer cancel()

select {
case <-time.After(2 * time.Second):
    fmt.Println("Finished work")
case <-ctx.Done():
    fmt.Println("Timed out:", ctx.Err())
}
```

Output:

```
Timed out: context deadline exceeded
```

---

Here is a **practical, engineer-friendly guide to memory optimization techniques in Go**, with **clear explanations, diagrams, and example code** you can use in real projects.

---

# 🚀 **1. Avoid Unnecessary Allocations**

### ❌ Slow (allocates every iteration)

```go
func bad() []int {
    var out []int
    for i := 0; i < 10000; i++ {
        out = append(out, i) // repeatedly grows, causing many allocations
    }
    return out
}
```

### ✅ Fast (pre-allocate with capacity)

```go
func good() []int {
    out := make([]int, 0, 10000) // capacity known → no resizing
    for i := 0; i < 10000; i++ {
        out = append(out, i)
    }
    return out
}
```

### 💡 Why?

Growing slices reallocates memory → **expensive heap operations**.

---

# 🚀 **2. Avoid Slicing Large Arrays (Memory Leak Pattern)**

This is a *classic Go footgun*.

### ❌ Problem: slicing holds reference to the entire underlying array

```go
func leak() []byte {
    big := make([]byte, 10<<20) // 10 MB
    small := big[:10]           // holds entire 10MB in memory
    return small
}
```

Even though you return 10 bytes, **the 10MB cannot be freed**.

### ✅ Solution: copy into a new small slice

```go
func noLeak() []byte {
    big := make([]byte, 10<<20)
    small := make([]byte, 10)
    copy(small, big[:10])
    return small
}
```

---

# 🚀 **3. Use `sync.Pool` to Reuse Heavy Objects**

Good for big buffers, JSON encoders, etc.

### Example

```go
var bufPool = sync.Pool{
    New: func() interface{} {
        return make([]byte, 1024) // 1KB buffer
    },
}

func handler() {
    buf := bufPool.Get().([]byte)
    defer bufPool.Put(buf)

    // use buffer...
}
```

### When to use:

✔ High throughput
✔ Objects expensive to allocate
✔ Short-lived and reusable

### When NOT to use:

❌ Long-lived data
❌ Objects with state that must be reset

---

# 🚀 **4. Reduce Allocations with `bytes.Buffer` or `strings.Builder`**

### ❌ Inefficient string concatenation (allocates each time)

```go
s := ""
for i := 0; i < 10000; i++ {
    s += "x"
}
```

### ✅ Using `strings.Builder`

```go
var b strings.Builder
b.Grow(10000)
for i := 0; i < 10000; i++ {
    b.WriteByte('x')
}
s := b.String()
```

---

# 🚀 **5. Avoid Interface Conversions & Escape to Heap**

### Example: Variables escape to heap when returned as `interface{}`

```go
func box(i int) interface{} {
    return i // escapes to heap
}
```

Heap allocations are more expensive.

### Solution

Use concrete types wherever possible.

---

# 🚀 **6. Use Value Receivers When Methods Do Not Mutate**

### ❌ Using pointer forces heap escapes for small objects:

```go
type Point struct {
    X, Y int
}

func (p *Point) Print() {
    fmt.Println(p.X, p.Y)
}
```

### ✅ Use value receivers when possible

```go
func (p Point) Print() {
    fmt.Println(p.X, p.Y)
}
```

Value receiver → stored on stack → fewer heap allocations.

---

# 🚀 **7. Reuse Byte Slices Instead of Creating New Ones**

### ❌ Allocates new slice each time:

```go
data := make([]byte, 0)
```

### ✅ Reuse existing slices

```go
data = data[:0] // reset length but keep capacity
```

This is especially useful in tight loops.

---

# 🚀 **8. Avoid Large Struct Copies — Use Pointers**

### ❌ Copying a huge struct costs memory and CPU:

```go
type Big struct {
    Data [1 << 20]byte // 1MB
}

func process(b Big) {} // copies 1MB each call
```

### ✅ Use pointer receiver

```go
func process(b *Big) {}
```

---

# 🚀 **9. Use Maps Carefully — They Allocate More Than You Think**

### Important rules:

✔ Use `make(map[...], capacity)` if size is predictable
✔ Use `delete(m, key)` to allow reuse
✔ Avoid storing large structs inside maps → use pointers

Example:

```go
m := make(map[string]*User, 1000)
```

---

# 🚀 **10. Tune the Garbage Collector (`GOGC`)**

### What is `GOGC`?

Controls how often GC runs.

```
GOGC=100 (default): normal frequency  
GOGC=200: runs less often → lower CPU, more memory  
GOGC=50: runs more often → less memory, more CPU  
```

### Example: reduce memory usage

```bash
export GOGC=50
```

---

# 🚀 **11. Zero Allocation Techniques in Hot Paths**

### Example: scanning JSON without allocating

```go
var s string
json.Unmarshal(data, &s)  // allocates

// but...
scan := []byte(s)  // reuse memory
```

Use:

* `unsafe.String()` (Go 1.20+)
* `bytes.Clone()`
* `sync.Pool` reusable buffers

To eliminate garbage.

---

# 🚀 **12. Use Profiling Tools (Very Important)**

### Memory profile:

```bash
go test -run=none -bench=. -benchmem
```

### pprof:

```bash
go tool pprof -http=:8080 mem.out
```

Find:

* escaped variables
* large allocations
* slice growth hotspots

---

# 📊 **Memory Leak Diagram**

```
Big slice (cap=10MB)
  |
  +-- small := big[:10]
            ↑
      Still holds reference to entire 10MB
```

---
