Here’s a clean, reactive **Alpine.js counter** with increment/decrement buttons and a double-click reset:

---

### **Solution Code**
```html
<!DOCTYPE html>
<html lang="en" x-data="{ count: 0 }">
<head>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <style>
        button { margin: 0.5rem; padding: 0.5rem 1rem; }
        .count { font-size: 2rem; font-weight: bold; }
    </style>
</head>
<body>
    <div>
        <!-- Display Counter -->
        <div class="count" x-text="count"></div>

        <!-- Increment/Decrement Buttons -->
        <button @click="count++">+1</button>
        <button @click="count--">-1</button>

        <!-- Reset on Double-Click -->
        <button 
            @dblclick="count = 0" 
            @click.prevent
            x-text="'Reset (Double-Click)'"
        ></button>
    </div>
</body>
</html>
```

---

### **Key Features**
1. **`@click` Events**:  
   - `count++` and `count--` directly modify the reactive `count` variable.

2. **`@dblclick` Reset**:  
   - Double-clicking the reset button sets `count = 0`.  
   - `@click.prevent` prevents single clicks from interfering.

3. **Reactivity**:  
   - Alpine automatically updates the DOM when `count` changes—no manual refreshes needed.

---

### **Challenge Extensions**
1. **Add Limits**:  
   Prevent negative values or set a max (e.g., 10):
   ```html
   <button @click="count = Math.min(count + 1, 10)">+1 (Max 10)</button>
   <button @click="count = Math.max(count - 1, 0)">-1 (Min 0)</button>
   ```

2. **Animate Changes**:  
   Use `x-transition` for visual feedback:
   ```html
   <div 
       x-text="count" 
       x-transition:enter="transition ease-out duration-300"
       x-transition:enter-start="opacity-0 scale-90"
   ></div>
   ```

3. **Input Field Sync**:  
   Add an `<input>` to manually set the count:
   ```html
   <input 
       type="number" 
       x-model="count" 
       @input="count = parseInt($event.target.value) || 0"
   >
   ```

---

### **Why This Works**
- Alpine’s **declarative syntax** binds events and data seamlessly.  
- No jQuery or `addEventListener`—just **direct HTML bindings** with JavaScript logic.  

Try it out! Each interaction updates the counter instantly. 🚀