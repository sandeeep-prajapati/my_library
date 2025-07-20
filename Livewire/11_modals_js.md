# **Launching Bootstrap Modals with Livewire Events**

Here's a complete guide to implementing Bootstrap modals in Livewire with proper event handling:

## **1. Basic Modal Implementation**

### **Component (PHP)**
```php
// app/Livewire/UserManager.php
use App\Models\User;

class UserManager extends Component
{
    public $showModal = false;
    public $selectedUser;
    public $name, $email;

    // Open modal
    public function showUserModal($userId = null)
    {
        $this->resetForm();
        $this->selectedUser = $userId;
        
        if ($userId) {
            $user = User::find($userId);
            $this->name = $user->name;
            $this->email = $user->email;
        }
        
        $this->showModal = true;
    }

    // Close modal
    public function closeModal()
    {
        $this->showModal = false;
    }

    // Save user
    public function saveUser()
    {
        $validated = $this->validate([
            'name' => 'required',
            'email' => 'required|email'
        ]);

        if ($this->selectedUser) {
            User::find($this->selectedUser)->update($validated);
            session()->flash('message', 'User updated!');
        } else {
            User::create($validated);
            session()->flash('message', 'User created!');
        }

        $this->closeModal();
    }

    private function resetForm()
    {
        $this->reset(['name', 'email', 'selectedUser']);
        $this->resetErrorBag();
    }

    public function render()
    {
        return view('livewire.user-manager', [
            'users' => User::all()
        ]);
    }
}
```

### **View (Blade)**
```html
<div>
    <!-- Button to open modal -->
    <button class="btn btn-primary" wire:click="showUserModal">
        Add New User
    </button>

    <!-- Users Table -->
    <table class="table">
        @foreach($users as $user)
            <tr>
                <td>{{ $user->name }}</td>
                <td>{{ $user->email }}</td>
                <td>
                    <button wire:click="showUserModal({{ $user->id }})" 
                            class="btn btn-sm btn-info">
                        Edit
                    </button>
                </td>
            </tr>
        @endforeach
    </table>

    <!-- Bootstrap Modal -->
    <div class="modal fade" id="userModal" tabindex="-1" 
         wire:ignore.self aria-hidden="true"
         @if($showModal) style="display: block;" @endif>
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">
                        {{ $selectedUser ? 'Edit User' : 'Create User' }}
                    </h5>
                    <button type="button" class="btn-close" 
                            wire:click="closeModal"></button>
                </div>
                <div class="modal-body">
                    @if(session('message'))
                        <div class="alert alert-success">
                            {{ session('message') }}
                        </div>
                    @endif

                    <form wire:submit.prevent="saveUser">
                        <div class="mb-3">
                            <label>Name</label>
                            <input type="text" class="form-control" wire:model="name">
                            @error('name') <span class="text-danger">{{ $message }}</span> @enderror
                        </div>
                        <div class="mb-3">
                            <label>Email</label>
                            <input type="email" class="form-control" wire:model="email">
                            @error('email') <span class="text-danger">{{ $message }}</span> @enderror
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" 
                            wire:click="closeModal">
                        Close
                    </button>
                    <button type="button" class="btn btn-primary" 
                            wire:click="saveUser">
                        Save
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Modal Backdrop -->
    @if($showModal)
        <div class="modal-backdrop fade show"></div>
    @endif
</div>

@push('scripts')
<script>
    // Initialize Bootstrap modal with Livewire
    document.addEventListener('livewire:load', function() {
        Livewire.on('closeModal', () => {
            // Optional: Add any custom close logic
        });
    });
</script>
@endpush
```

## **2. Key Features**

1. **Modal State Management**
   - `$showModal` controls visibility
   - `wire:ignore.self` prevents Livewire from managing modal DOM

2. **Dual-Purpose Modal**
   - Handles both create and edit modes
   - Auto-populates form when editing

3. **Proper Bootstrap Integration**
   - Manual backdrop implementation
   - Correct modal display styling

4. **Form Handling**
   - Validation with error display
   - Success messages
   - Reset on close

## **3. Advanced Implementation**

### **With Alpine.js (Recommended)**
```html
<div x-data="{ open: @entangle('showModal') }">
    <!-- Modal Trigger -->
    <button @click="open = true">Open Modal</button>

    <!-- Modal -->
    <div x-show="open" @keydown.escape.window="open = false" class="modal">
        <div class="modal-content">
            <button @click="open = false">×</button>
            <!-- Modal content -->
        </div>
    </div>
</div>
```

### **With Livewire Events**
```php
// In your component
protected $listeners = ['showUserModal' => 'showUserModal'];

// In other components
$this->emit('showUserModal', $userId);
```

## **4. Troubleshooting**

**Common Issues:**
1. **Modal not showing?**
   - Ensure you have both the modal div AND the backdrop
   - Check for conflicting z-index values

2. **Form submissions not working?**
   - Verify `wire:submit.prevent` is used
   - Check browser console for JavaScript errors

3. **Multiple modals conflict?**
   - Use unique IDs for each modal
   - Consider Alpine.js for better state management

## **5. Performance Optimization**

1. **Lazy Load Modals**
```html
@if($showModal)
    @include('modals.user-modal')
@endif
```

2. **Defer Heavy Content**
```html
<div wire:init="loadModalData">
    <!-- Content loaded only when modal opens -->
</div>
```

This implementation gives you a complete, production-ready modal system with:
- Create/edit functionality
- Built-in validation
- Proper Bootstrap integration
- Responsive design
- Clean state management