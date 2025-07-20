# **Handling File Uploads in Livewire (With Validation)**

Livewire makes file uploads simple while keeping everything in PHP. Here's a complete guide with validation:

## **1. Basic File Upload**

### **Component (PHP)**
```php
// app/Livewire/FileUpload.php
use Livewire\WithFileUploads;

class FileUpload extends Component
{
    use WithFileUploads; // Required trait
    
    public $file;
    
    public function save()
    {
        $this->validate([
            'file' => 'required|file|max:1024', // 1MB max
        ]);
        
        $path = $this->file->store('uploads');
        // Save $path to database or process file
    }
    
    public function render()
    {
        return view('livewire.file-upload');
    }
}
```

### **View (Blade)**
```html
<form wire:submit.prevent="save" enctype="multipart/form-data">
    <input type="file" wire:model="file">
    
    @error('file') <span class="error">{{ $message }}</span> @enderror
    
    <button type="submit">
        Upload
    </button>
    
    <!-- Show upload progress -->
    <div wire:loading wire:target="file">
        Uploading... 
        <span x-text="Math.round($wire.uploadProgress)"></span>%
    </div>
</form>
```

## **2. Key Features**

1. **`WithFileUploads` Trait**
   - Enables file upload functionality
   - Handles temporary file storage

2. **Validation Rules**
   ```php
   'file' => 'required|file|mimes:jpg,png,pdf|max:1024' // 1MB, specific types
   ```

3. **Automatic Progress Indicator**
   - Built-in upload progress tracking
   - Accessible via `$wire.uploadProgress`

## **3. Advanced Implementation**

### **Multiple Files**
```php
public $files = [];

public function save()
{
    $this->validate([
        'files.*' => 'file|max:1024',
    ]);
    
    foreach ($this->files as $file) {
        $file->store('uploads');
    }
}
```
```html
<input type="file" wire:model="files" multiple>
```

### **Image Preview**
```html
@if($file && in_array($file->extension(), ['jpg', 'jpeg', 'png']))
    <img src="{{ $file->temporaryUrl() }}" width="200">
@endif
```

### **Custom Storage**
```php
$path = $this->file->storeAs(
    'custom-folder',
    'custom-filename.jpg',
    's3' // Storage disk
);
```

## **4. Important Notes**

1. **Required Attributes**
   - `enctype="multipart/form-data"` on form
   - `wire:model` on file input

2. **Temporary Files**
   - Files stored temporarily in `livewire-tmp/`
   - Automatically cleaned after 24h

3. **Security**
   - Always validate file types and sizes
   - Process files in a secure environment

## **5. Troubleshooting**

- **"File upload not working?"**
  - Check PHP `upload_max_filesize` in php.ini
  - Verify `post_max_size` is larger than your files

- **"Temporary URL not working?"**
  - For image previews, configure temporary URL disk in `config/livewire.php`:
    ```php
    'temporary_file_upload' => [
        'disk' => 'local', // Or 's3' for cloud
    ],
    ```

## **Complete Example: Avatar Upload**

```php
// app/Livewire/AvatarUpload.php
use Livewire\WithFileUploads;

class AvatarUpload extends Component
{
    use WithFileUploads;
    
    public $avatar;
    public $user;
    
    public function mount(User $user)
    {
        $this->user = $user;
    }
    
    public function save()
    {
        $this->validate([
            'avatar' => 'required|image|max:2048', // 2MB max
        ]);
        
        $this->user->update([
            'avatar_path' => $this->avatar->store('avatars', 'public')
        ]);
    }
}
```
```html
<div>
    @if($avatar)
        <img src="{{ $avatar->temporaryUrl() }}" width="100">
    @elseif($user->avatar_path)
        <img src="{{ asset('storage/'.$user->avatar_path) }}" width="100">
    @endif
    
    <input type="file" wire:model="avatar">
    
    <button wire:click="save" wire:loading.attr="disabled">
        Save Avatar
    </button>
</div>
```

This implementation gives you a complete, secure file upload system with:
- Validation
- Progress indicators
- Image previews
- Proper error handling