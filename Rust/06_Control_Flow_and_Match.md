### **Topic:** How Does Rust Handle Control Flow Using `if`, `loop`, `while`, `for`, and the Powerful `match` Expression?  

---

Control flow in Rust is both expressive and safe, enabling clear and efficient decision-making and iteration. Rust provides several constructs for managing control flow: `if` for conditional logic, `loop`, `while`, and `for` for iteration, and `match` for pattern matching. Let’s explore each in detail.

---

### **1. `if` Expression**  
Rust's `if` works as both a conditional statement and an expression that can return a value.

#### **Basic Usage**  
```rust
let number = 10;

if number > 5 {
    println!("The number is greater than 5.");
} else {
    println!("The number is 5 or less.");
}
```

#### **Using `if` as an Expression**  
```rust
let number = 10;
let result = if number % 2 == 0 { "even" } else { "odd" };
println!("The number is {}.", result); // Output: "The number is even."
```

- All branches of the `if` expression must return the same type.

---

### **2. Loops**  

#### **`loop`: Infinite Loop**  
The `loop` keyword creates an infinite loop.  
```rust
let mut count = 0;

loop {
    if count >= 5 {
        break;
    }
    println!("Count: {}", count);
    count += 1;
}
```

#### **Returning Values from `loop`**  
```rust
let result = loop {
    let mut count = 0;
    count += 1;
    if count == 3 {
        break count * 2; // Return value from loop
    }
};
println!("Result: {}", result); // Output: "Result: 6"
```

---

### **3. `while` Loop**  
The `while` loop continues as long as a condition evaluates to `true`.  
```rust
let mut number = 3;

while number > 0 {
    println!("{}!", number);
    number -= 1;
}
println!("Liftoff!");
```

- Use `while` for cases where the termination condition is dynamic or unknown upfront.

---

### **4. `for` Loop**  
The `for` loop is idiomatic in Rust for iterating over collections or ranges.  

#### **Iterating Over a Range**  
```rust
for i in 1..5 {
    println!("Number: {}", i); // Outputs 1, 2, 3, 4 (end is exclusive)
}

for i in 1..=5 {
    println!("Number: {}", i); // Outputs 1, 2, 3, 4, 5 (end is inclusive)
}
```

#### **Iterating Over Collections**  
```rust
let numbers = vec![10, 20, 30];

for num in numbers.iter() {
    println!("Number: {}", num);
}
```

#### **Enumerating Items**  
```rust
for (index, value) in numbers.iter().enumerate() {
    println!("Index: {}, Value: {}", index, value);
}
```

---

### **5. `match` Expression**  
The `match` expression in Rust is a powerful pattern-matching construct. It is exhaustive, meaning all possible cases must be handled.  

#### **Basic Usage**  
```rust
let number = 2;

match number {
    1 => println!("One"),
    2 => println!("Two"),
    _ => println!("Something else"), // `_` is the catch-all pattern
}
```

#### **Matching with Ranges**  
```rust
let number = 15;

match number {
    1..=10 => println!("Between 1 and 10"),
    11..=20 => println!("Between 11 and 20"),
    _ => println!("Something else"),
}
```

#### **Matching Multiple Patterns**  
```rust
let number = 1;

match number {
    1 | 3 | 5 => println!("Odd number"),
    2 | 4 | 6 => println!("Even number"),
    _ => println!("Something else"),
}
```

#### **Using `match` as an Expression**  
```rust
let value = match 10 {
    10 => "ten",
    _ => "other",
};
println!("Value: {}", value); // Output: "Value: ten"
```

#### **Destructuring in `match`**  
`match` can destructure enums, tuples, and structs:  
```rust
let point = (2, 3);

match point {
    (0, 0) => println!("Origin"),
    (x, y) => println!("Point: ({}, {})", x, y),
}
```

---

### **6. Combining Control Flow Constructs**  

Rust's control flow constructs can be combined for powerful logic.  

#### **Example: A Basic FizzBuzz Program**  
```rust
for number in 1..=15 {
    match (number % 3 == 0, number % 5 == 0) {
        (true, true) => println!("FizzBuzz"),
        (true, false) => println!("Fizz"),
        (false, true) => println!("Buzz"),
        _ => println!("{}", number),
    }
}
```

---

### **7. Conclusion**  
Rust's control flow constructs are designed to be safe, expressive, and efficient. By leveraging `if`, `loop`, `while`, `for`, and `match`, programmers can handle complex logic while maintaining clarity and type safety. The `match` expression, in particular, stands out for its versatility and pattern-matching power.