Of course. This is a perfect use case for Inertia. We'll build a Todo app where all CRUD operations are handled through Laravel controllers that return Inertia responses, using Inertia's form helpers on the Vue side for a seamless experience.

### **02 - Todo App with Inertia.js and Vue 3**

This guide covers creating a full CRUD Todo application without building a separate API.

---

#### **Step 1: Database & Model Setup**

1.  **Create Migration:**
    ```bash
    php artisan make:migration create_tasks_table
    ```
    Edit the migration file (`database/migrations/[...]_create_tasks_table.php`):
    ```php
    public function up(): void
    {
        Schema::create('tasks', function (Blueprint $table) {
            $table->id();
            $table->string('title');
            $table->boolean('is_completed')->default(false);
            $table->timestamps();
        });
    }
    ```

2.  **Run Migration:**
    ```bash
    php artisan migrate
    ```

3.  **Create Eloquent Model:**
    ```bash
    php artisan make:model Task
    ```
    Edit `app/Models/Task.php` to make the `title` field fillable:
    ```php
    use HasFactory;

    protected $fillable = ['title', 'is_completed'];
    ```

---

#### **Step 2: Create the Controller**

```bash
php artisan make:controller TaskController
```

Edit `app/Http/Controllers/TaskController.php`:

```php
<?php

namespace App\Http\Controllers;

use App\Models\Task;
use Illuminate\Http\Request;
use Inertia\Inertia;
// Don't forget to import the Task model

class TaskController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index()
    {
        // Get all tasks, ordered by newest first
        $tasks = Task::latest()->get();

        // Return the Vue component with the tasks as a prop
        return Inertia::render('Tasks/Index', [
            'tasks' => $tasks,
        ]);
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        // Validate the request
        $validated = $request->validate([
            'title' => 'required|string|max:255',
        ]);

        // Create the task
        Task::create($validated);

        // Redirect back to the index page
        return redirect()->route('tasks.index');
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, Task $task)
    {
        // Validate the request
        $validated = $request->validate([
            'title' => 'sometimes|required|string|max:255', // Only validate if present
            'is_completed' => 'sometimes|boolean',          // Only validate if present
        ]);

        // Update the task
        $task->update($validated);

        // Redirect back
        return redirect()->route('tasks.index');
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(Task $task)
    {
        // Delete the task
        $task->delete();

        // Redirect back
        return redirect()->route('tasks.index');
    }
}
```

---

#### **Step 3: Define Routes**

Edit `routes/web.php`:

```php
<?php

use App\Http\Controllers\TaskController;
use Illuminate\Support\Facades\Route;

Route::get('/', function () {
    return redirect()->route('tasks.index');
});

// Resourceful routes for our TaskController
Route::resource('tasks', TaskController::class)->only(['index', 'store', 'update', 'destroy']);
```

---

#### **Step 4: Build the Vue Components (The Frontend)**

**1. Main Index Page (`resources/js/Pages/Tasks/Index.vue`)**

This is the core component that lists todos and handles all interactions.

```vue
<script setup>
// Import Inertia utilities and components
import { Head, Link, router } from '@inertiajs/vue3';
// Import Inertia's powerful form helper
import { useForm } from '@inertiajs/vue3';

// Define props passed from the Controller
defineProps({
  tasks: Array,
});

// Setup form for creating a new task
// useForm gives us a reactive object with helper methods
const form = useForm({
  title: '',
});

// Function to submit the new task form
const submit = () => {
  // POST to the 'tasks.store' route
  form.post(route('tasks.store'), {
    // Reset the form on success
    onSuccess: () => form.reset('title'),
  });
};

// Function to toggle task completion
const toggleCompleted = (task) => {
  // PATCH to the 'tasks.update' route for the specific task
  // Inertia's `router` object handles these visits without full page reloads
  router.patch(route('tasks.update', task.id), {
    is_completed: !task.is_completed,
  });
};

// Function to delete a task
const deleteTask = (task) => {
  if (confirm('Are you sure you want to delete this task?')) {
    // DELETE to the 'tasks.destroy' route
    router.delete(route('tasks.destroy', task.id));
  }
};
</script>

<template>
  <Head title="Todo App" />

  <div class="max-w-2xl mx-auto p-6">
    <h1 class="text-3xl font-bold text-gray-800 mb-8">Inertia Todo App</h1>

    <!-- Form for creating a new task -->
    <form @submit.prevent="submit" class="mb-8">
      <div class="flex gap-2">
        <input
          v-model="form.title"
          type="text"
          placeholder="What needs to be done?"
          class="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          :disabled="form.processing"
        />
        <button
          type="submit"
          class="px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50"
          :disabled="form.processing || !form.title"
        >
          Add Task
        </button>
      </div>
      <!-- Display validation errors for the 'title' field -->
      <div v-if="form.errors.title" class="text-sm text-red-600 mt-2">
        {{ form.errors.title }}
      </div>
    </form>

    <!-- List of tasks -->
    <div v-if="tasks.length > 0" class="bg-white shadow rounded-lg divide-y">
      <div
        v-for="task in tasks"
        :key="task.id"
        class="p-4 flex items-center justify-between group"
      >
        <div class="flex items-center space-x-3">
          <!-- Checkbox to toggle completion -->
          <input
            type="checkbox"
            :checked="task.is_completed"
            @change="toggleCompleted(task)"
            class="h-5 w-5 text-blue-600 rounded focus:ring-blue-500"
          />
          <!-- Task title with strikethrough if completed -->
          <span
            :class="[
              'text-lg',
              task.is_completed ? 'line-through text-gray-500' : 'text-gray-800'
            ]"
          >
            {{ task.title }}
          </span>
        </div>
        <!-- Delete button -->
        <button
          @click="deleteTask(task)"
          class="text-red-500 opacity-0 group-hover:opacity-100 transition-opacity p-2 hover:bg-red-50 rounded-full"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      </div>
    </div>

    <!-- Empty state -->
    <div v-else class="text-center py-12 text-gray-500">
      <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
      </svg>
      <p>No tasks yet. Add one above!</p>
    </div>
  </div>
</template>
```

---

#### **Key Concepts Explained:**

1.  **`useForm()` Helper:** This is the star of the show. It creates a reactive form object that:
    *   Tracks field values (`v-model="form.title"`).
    *   Manages submission state (`form.processing`).
    *   Holds validation errors from Laravel (`form.errors.title`).
    *   Provides methods (`post()`, `get()`, `patch()`, `delete()`) to submit data using Inertia visits.

2.  **Inertia Visits:** Instead of using `axios` or `fetch`, we use Inertia's methods to navigate and submit data:
    *   **`form.post(...)`**: Submits the form to the specified route.
    *   **`router.patch(...)`**, **`router.delete(...)`**: Perform PATCH and DELETE requests directly. These are all "visits" that update the page component and props without a full browser refresh.

3.  **Laravel Controller Returns:** The controller doesn't return JSON. It either:
    *   `return Inertia::render(...)`: Renders a Vue component with props.
    *   `return redirect()->route(...)`: Redirects back to a page, which triggers a new `Inertia::render()` and updates the list of tasks.

4.  **Route Helper:** `route('tasks.update', task.id)` generates the correct URL based on your `web.php` routes, ensuring consistency between Laravel and Vue.

This architecture seamlessly blends Laravel's backend logic with Vue's reactivity, creating a single-page app experience without the complexity of building and maintaining a separate API.