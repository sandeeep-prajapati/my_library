### **Handling Button Clicks (`wire:click`) and Form Submissions in Livewire**  

Livewire makes it easy to handle user interactions like button clicks and form submissions **without writing JavaScript**. Here’s how to master these features:

---

## **1. Basic Button Clicks (`wire:click`)**
### **A. Trigger a PHP Method**
```html
<button wire:click="increment">+</button>
```
```php
// In your Livewire component (e.g., Counter.php)
public $count = 0;

public function increment()
{
    $this->count++; // Updates the UI automatically
}
```
**→ Clicking the button calls `increment()` and updates `$count` in real-time.**

---

### **B. Pass Parameters**
```html
<button wire:click="add(5)">Add 5</button>
```
```php
public function add($number)
{
    $this->count += $number;
}
```

---

### **C. Confirm Actions (Like `confirm()` in JS)**
```html
<button wire:click="delete" wire:confirm="Are you sure?">Delete</button>
```
**→ Shows a browser confirmation dialog before executing `delete()`.**

---

## **2. Handling Form Submissions**
### **A. Basic Form**
```html
<form wire:submit.prevent="save"> <!-- Prevent default page reload -->
    <input wire:model="name" type="text">
    <button type="submit">Save</button>
</form>
```
```php
public $name = '';

public function save()
{
    // Validate and save data
    $this->validate(['name' => 'required|min:3']);
    User::create(['name' => $this->name]);
    
    // Reset the field
    $this->name = '';
    
    // Flash a success message
    session()->flash('message', 'User saved!');
}
```

---

### **B. Validation & Error Display**
```php
protected $rules = [
    'email' => 'required|email',
];

public function save()
{
    $this->validate(); // Triggers validation rules
    
    // Proceed if valid...
}
```
Show errors in Blade:
```html
<input wire:model="email" type="email">
@error('email') <span class="error">{{ $message }}</span> @enderror
```

---

### **C. File Uploads in Forms**
```html
<form wire:submit.prevent="upload" enctype="multipart/form-data">
    <input type="file" wire:model="photo">
    <button type="submit">Upload</button>
</form>
```
```php
public $photo;

public function upload()
{
    $this->validate([
        'photo' => 'image|max:1024', // 1MB max
    ]);
    
    $path = $this->photo->store('photos');
    // Save $path to the database...
}
```

---

## **3. Advanced Interactions**
### **A. Redirect After Submission**
```php
public function save()
{
    // Save data...
    return redirect()->to('/success-page'); // Standard Laravel redirect
}
```

### **B. Loading States**
Show a spinner during processing:
```html
<button wire:click="save" wire:loading.class="opacity-50" wire:loading.attr="disabled">
    <span wire:loading.remove>Save</span>
    <span wire:loading>Saving...</span>
</button>
```
**→ Automatically adds `disabled` and CSS classes during AJAX calls.**

---

### **C. Chaining Methods**
Run multiple actions sequentially:
```html
<button wire:click="validateThenSave">Submit</button>
```
```php
public function validateThenSave()
{
    $this->validate();
    $this->save();
}
```

---

## **4. Key Directives Cheat Sheet**
| Directive               | Purpose                                 | Example                          |
|-------------------------|-----------------------------------------|----------------------------------|
| `wire:click`            | Calls a PHP method on click             | `<button wire:click="delete">`   |
| `wire:submit.prevent`   | Submits a form without page reload      | `<form wire:submit.prevent="save">` |
| `wire:confirm`          | Adds a JS confirmation dialog           | `<button wire:click="delete" wire:confirm="Sure?">` |
| `wire:loading`          | Toggles UI during AJAX calls            | `<span wire:loading>Saving...</span>` |
| `wire:target`           | Limits `wire:loading` to specific actions | `<button wire:click="save" wire:target="save">` |

---

## **5. Real-World Example: Todo List**
```php
// app/Livewire/TodoList.php
public $task = '';
public $todos = [];

public function addTodo()
{
    $this->validate(['task' => 'required|min:3']);
    $this->todos[] = $this->task;
    $this->task = '';
}

public function deleteTodo($index)
{
    unset($this->todos[$index]);
}
```
```html
<div>
    <input wire:model="task" placeholder="New task">
    <button wire:click="addTodo">Add</button>
    
    <ul>
        @foreach($todos as $index => $todo)
            <li>
                {{ $todo }}
                <button wire:click="deleteTodo({{ $index }})">×</button>
            </li>
        @endforeach
    </ul>
</div>
```

---

## **6. Troubleshooting**
- **Method not firing?**  
  - Ensure the method is `public` in the Livewire component.  
  - Check your browser’s console for JavaScript errors.  
- **Form not submitting?**  
  - Verify `wire:submit.prevent` is used (not `wire:click` on the submit button).  
  - Ensure no validation errors block the submission.  

---