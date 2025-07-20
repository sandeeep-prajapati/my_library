# **Livewire CRUD Example (Create, Read, Update, Delete)**

Let's build a complete **Task Manager** with Livewire, covering all CRUD operations. This example includes:
- **Listing tasks** (Read)
- **Adding new tasks** (Create)
- **Editing tasks** (Update)
- **Deleting tasks** (Delete)

---

## **1. Setup the Component**
```bash
php artisan make:livewire TaskManager
```
This creates:
- `app/Livewire/TaskManager.php`
- `resources/views/livewire/task-manager.blade.php`

---

## **2. Database & Model**
### **A. Create Migration**
```bash
php artisan make:migration create_tasks_table
```
```php
// database/migrations/..._create_tasks_table.php
Schema::create('tasks', function (Blueprint $table) {
    $table->id();
    $table->string('title');
    $table->boolean('completed')->default(false);
    $table->timestamps();
});
```
Run migrations:
```bash
php artisan migrate
```

### **B. Create Task Model**
```bash
php artisan make:model Task
```
```php
// app/Models/Task.php
protected $fillable = ['title', 'completed'];
```

---

## **3. Livewire Component Logic**
```php
// app/Livewire/TaskManager.php
use App\Models\Task;

class TaskManager extends Component
{
    public $tasks, $title, $editId, $editTitle;

    // Initialize data
    public function mount()
    {
        $this->tasks = Task::latest()->get();
    }

    // Create a new task
    public function create()
    {
        Task::create(['title' => $this->title]);
        $this->title = ''; // Clear input
        $this->mount(); // Refresh task list
    }

    // Set task for editing
    public function edit($id)
    {
        $task = Task::find($id);
        $this->editId = $task->id;
        $this->editTitle = $task->title;
    }

    // Update task
    public function update()
    {
        Task::find($this->editId)->update(['title' => $this->editTitle]);
        $this->cancelEdit(); // Reset edit state
        $this->mount(); // Refresh list
    }

    // Cancel editing
    public function cancelEdit()
    {
        $this->editId = null;
        $this->editTitle = '';
    }

    // Delete task
    public function delete($id)
    {
        Task::find($id)->delete();
        $this->mount(); // Refresh list
    }

    public function render()
    {
        return view('livewire.task-manager');
    }
}
```

---

## **4. Blade View (UI)**
```html
<!-- resources/views/livewire/task-manager.blade.php -->
<div>
    <!-- Create Task -->
    <div>
        <input wire:model="title" placeholder="New task...">
        <button wire:click="create">Add</button>
    </div>

    <!-- Task List -->
    <ul>
        @foreach($tasks as $task)
            <li>
                @if($editId == $task->id)
                    <!-- Edit Mode -->
                    <input wire:model="editTitle">
                    <button wire:click="update">Save</button>
                    <button wire:click="cancelEdit">Cancel</button>
                @else
                    <!-- Read Mode -->
                    {{ $task->title }}
                    <button wire:click="edit({{ $task->id }})">Edit</button>
                    <button wire:click="delete({{ $task->id }})">Delete</button>
                @endif
            </li>
        @endforeach
    </ul>
</div>
```

---

## **5. Using the Component**
Embed in any Blade view:
```html
<livewire:task-manager />
```
Or in routes:
```php
Route::get('/tasks', function () {
    return view('tasks'); // Contains <livewire:task-manager />
});
```

---

## **6. Key Features Demonstrated**
| CRUD Operation | Livewire Implementation |
|---------------|-------------------------|
| **Create** | `wire:click="create"` + `wire:model="title"` |
| **Read** | `$tasks = Task::latest()->get()` in `mount()` |
| **Update** | Edit/Save flow with `$editId` tracking |
| **Delete** | `wire:click="delete({{ $id }})"` |

---

## **7. Enhancements (Optional)**
### **A. Add Validation**
```php
public function create()
{
    $this->validate(['title' => 'required|min:3']);
    // ... rest of create logic
}
```

### **B. Loading States**
```html
<button wire:click="create" wire:loading.attr="disabled">
    <span wire:loading.remove>Add</span>
    <span wire:loading>Adding...</span>
</button>
```

### **C. Flash Messages**
```php
public function create()
{
    // ... after successful creation
    session()->flash('message', 'Task added!');
}
```
```html
@if(session('message'))
    <div>{{ session('message') }}</div>
@endif
```

---