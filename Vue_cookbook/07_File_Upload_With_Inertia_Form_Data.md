I'll show you how to handle file uploads with image previews in Vue and process them in Laravel using Inertia's form submission.

## 1. Vue Component with File Upload and Image Preview

```vue
<template>
  <form @submit.prevent="submit" class="space-y-6">
    <!-- Text Fields -->
    <div>
      <label for="name" class="block text-sm font-medium text-gray-700">Name</label>
      <input
        id="name"
        v-model="form.name"
        type="text"
        class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
        :class="{ 'border-red-500': form.errors.name }"
      />
      <div v-if="form.errors.name" class="text-red-600 text-sm mt-1">
        {{ form.errors.name }}
      </div>
    </div>

    <!-- File Upload with Preview -->
    <div>
      <label class="block text-sm font-medium text-gray-700">Profile Image</label>
      
      <!-- Image Preview -->
      <div v-if="imagePreview" class="mt-2">
        <img 
          :src="imagePreview" 
          alt="Image preview" 
          class="w-32 h-32 object-cover rounded-lg border"
        />
        <button
          type="button"
          @click="removeImage"
          class="mt-2 text-sm text-red-600 hover:text-red-800"
        >
          Remove Image
        </button>
      </div>

      <!-- File Input -->
      <div v-else class="mt-2 flex items-center">
        <label
          for="image"
          class="cursor-pointer bg-white py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          Choose Image
        </label>
        <input
          id="image"
          type="file"
          accept="image/*"
          @change="handleImageSelect"
          class="hidden"
        />
        <span class="ml-3 text-sm text-gray-500">JPEG, PNG, GIF up to 5MB</span>
      </div>

      <div v-if="form.errors.image" class="text-red-600 text-sm mt-1">
        {{ form.errors.image }}
      </div>
    </div>

    <!-- Multiple Files Upload -->
    <div>
      <label class="block text-sm font-medium text-gray-700">Additional Images</label>
      
      <!-- Selected Files List -->
      <div v-if="selectedFiles.length > 0" class="mt-2 space-y-2">
        <div
          v-for="(file, index) in selectedFiles"
          :key="index"
          class="flex items-center justify-between p-2 bg-gray-50 rounded-md"
        >
          <span class="text-sm text-gray-600">{{ file.name }}</span>
          <button
            type="button"
            @click="removeFile(index)"
            class="text-red-600 hover:text-red-800"
          >
            &times;
          </button>
        </div>
      </div>

      <!-- File Input for Multiple -->
      <div class="mt-2">
        <label
          for="files"
          class="cursor-pointer bg-white py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          Add Files
        </label>
        <input
          id="files"
          type="file"
          multiple
          @change="handleFilesSelect"
          class="hidden"
        />
      </div>

      <div v-if="form.errors.files" class="text-red-600 text-sm mt-1">
        {{ form.errors.files }}
      </div>
    </div>

    <!-- Submit Button -->
    <button
      type="submit"
      :disabled="form.processing"
      class="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
    >
      {{ form.processing ? 'Creating...' : 'Create Profile' }}
    </button>
  </form>
</template>

<script setup>
import { useForm } from '@inertiajs/inertia-vue3'
import { ref } from 'vue'

const imagePreview = ref(null)
const selectedFiles = ref([])

const form = useForm({
  name: '',
  image: null,
  files: [],
})

// Handle single image selection with preview
const handleImageSelect = (event) => {
  const file = event.target.files[0]
  if (!file) return

  // Validate file type and size
  if (!file.type.startsWith('image/')) {
    form.errors.image = 'Please select an image file'
    return
  }

  if (file.size > 5 * 1024 * 1024) { // 5MB
    form.errors.image = 'Image must be less than 5MB'
    return
  }

  form.image = file
  form.errors.image = null

  // Create preview
  const reader = new FileReader()
  reader.onload = (e) => {
    imagePreview.value = e.target.result
  }
  reader.readAsDataURL(file)
}

// Handle multiple file selection
const handleFilesSelect = (event) => {
  const files = Array.from(event.target.files)
  
  // Validate files
  const validFiles = files.filter(file => {
    if (file.size > 10 * 1024 * 1024) { // 10MB per file
      form.errors.files = 'Each file must be less than 10MB'
      return false
    }
    return true
  })

  selectedFiles.value = [...selectedFiles.value, ...validFiles]
  form.files = selectedFiles.value
  form.errors.files = null
}

// Remove single image
const removeImage = () => {
  form.image = null
  imagePreview.value = null
  form.errors.image = null
}

// Remove file from multiple selection
const removeFile = (index) => {
  selectedFiles.value.splice(index, 1)
  form.files = selectedFiles.value
}

// Submit form
const submit = () => {
  // Create FormData for file uploads
  const formData = new FormData()
  formData.append('name', form.name)
  
  if (form.image) {
    formData.append('image', form.image)
  }
  
  selectedFiles.value.forEach((file, index) => {
    formData.append(`files[${index}]`, file)
  })

  form.post('/profiles', {
    data: formData,
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onSuccess: () => {
      // Reset form and previews
      form.reset()
      imagePreview.value = null
      selectedFiles.value = []
    },
    onError: (errors) => {
      console.log('Form errors:', errors)
    },
  })
}
</script>
```

## 2. Laravel Controller for File Processing

```php
<?php

namespace App\Http\Controllers;

use App\Models\Profile;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Facades\Validator;

class ProfileController extends Controller
{
    public function store(Request $request)
    {
        $validator = Validator::make($request->all(), [
            'name' => 'required|string|max:255',
            'image' => 'nullable|image|mimes:jpeg,png,jpg,gif|max:5120', // 5MB
            'files.*' => 'nullable|file|mimes:jpeg,png,jpg,gif,pdf,doc,docx|max:10240', // 10MB each
        ], [
            'image.image' => 'The profile image must be a valid image.',
            'image.max' => 'The profile image must not exceed 5MB.',
            'files.*.max' => 'Each file must not exceed 10MB.',
            'files.*.mimes' => 'Allowed file types: jpeg, png, jpg, gif, pdf, doc, docx.',
        ]);

        if ($validator->fails()) {
            return redirect()->back()
                ->withErrors($validator)
                ->withInput();
        }

        $profileData = [
            'name' => $request->name,
        ];

        // Handle single image upload
        if ($request->hasFile('image')) {
            $imagePath = $this->storeImage($request->file('image'));
            $profileData['image_path'] = $imagePath;
        }

        // Create profile
        $profile = Profile::create($profileData);

        // Handle multiple file uploads
        if ($request->hasFile('files')) {
            $this->storeFiles($request->file('files'), $profile);
        }

        return redirect()->route('profiles.index')
            ->with('success', 'Profile created successfully!');
    }

    private function storeImage($image)
    {
        // Store in storage/app/public/images
        $path = $image->store('images', 'public');
        
        // Optional: Create thumbnails or resize images
        // $this->createThumbnail($path);
        
        return $path;
    }

    private function storeFiles($files, $profile)
    {
        foreach ($files as $file) {
            $path = $file->store('documents', 'public');
            
            $profile->files()->create([
                'original_name' => $file->getClientOriginalName(),
                'path' => $path,
                'mime_type' => $file->getMimeType(),
                'size' => $file->getSize(),
            ]);
        }
    }

    // Optional: Image processing example
    private function createThumbnail($imagePath)
    {
        $image = Image::make(storage_path('app/public/' . $imagePath));
        $image->resize(300, 300, function ($constraint) {
            $constraint->aspectRatio();
            $constraint->upsize();
        });
        $thumbnailPath = 'thumbnails/' . basename($imagePath);
        $image->save(storage_path('app/public/' . $thumbnailPath));
        
        return $thumbnailPath;
    }

    public function update(Request $request, Profile $profile)
    {
        $validator = Validator::make($request->all(), [
            'name' => 'required|string|max:255',
            'image' => 'nullable|image|mimes:jpeg,png,jpg,gif|max:5120',
        ]);

        if ($validator->fails()) {
            return redirect()->back()
                ->withErrors($validator)
                ->withInput();
        }

        $profile->name = $request->name;

        // Handle image update - delete old image if new one is uploaded
        if ($request->hasFile('image')) {
            // Delete old image if exists
            if ($profile->image_path) {
                Storage::disk('public')->delete($profile->image_path);
            }
            
            $imagePath = $this->storeImage($request->file('image'));
            $profile->image_path = $imagePath;
        }

        $profile->save();

        return redirect()->route('profiles.index')
            ->with('success', 'Profile updated successfully!');
    }
}
```

## 3. Enhanced Vue Composables for File Handling

```js
// composables/useFileUpload.js
import { ref } from 'vue'

export function useFileUpload() {
  const imagePreview = ref(null)
  const selectedFiles = ref([])

  const handleImageSelect = (event, maxSize = 5 * 1024 * 1024) => {
    const file = event.target.files[0]
    if (!file) return null

    // Validation
    if (!file.type.startsWith('image/')) {
      throw new Error('Please select an image file')
    }

    if (file.size > maxSize) {
      throw new Error(`Image must be less than ${maxSize / 1024 / 1024}MB`)
    }

    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => {
      imagePreview.value = e.target.result
    }
    reader.readAsDataURL(file)

    return file
  }

  const handleFilesSelect = (event, maxSize = 10 * 1024 * 1024) => {
    const files = Array.from(event.target.files)
    const validFiles = files.filter(file => file.size <= maxSize)
    
    selectedFiles.value = [...selectedFiles.value, ...validFiles]
    return validFiles
  }

  const removeImage = () => {
    imagePreview.value = null
  }

  const removeFile = (index) => {
    selectedFiles.value.splice(index, 1)
  }

  const clearAllFiles = () => {
    imagePreview.value = null
    selectedFiles.value = []
  }

  return {
    imagePreview,
    selectedFiles,
    handleImageSelect,
    handleFilesSelect,
    removeImage,
    removeFile,
    clearAllFiles,
  }
}
```

## 4. Enhanced Form Component with Progress Tracking

```vue
<template>
  <form @submit.prevent="submit" class="space-y-6">
    <!-- Progress Bar for File Uploads -->
    <div v-if="uploadProgress > 0" class="w-full bg-gray-200 rounded-full h-2">
      <div
        class="bg-blue-600 h-2 rounded-full transition-all duration-300"
        :style="{ width: `${uploadProgress}%` }"
      ></div>
    </div>

    <!-- File Upload Section -->
    <div>
      <label class="block text-sm font-medium text-gray-700">Upload Files</label>
      <div
        @dragover="onDragOver"
        @dragleave="onDragLeave"
        @drop="onDrop"
        :class="[
          'mt-2 flex justify-center px-6 pt-5 pb-6 border-2 border-dashed rounded-md',
          isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
        ]"
      >
        <div class="space-y-1 text-center">
          <svg class="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
          <div class="flex text-sm text-gray-600">
            <label class="relative cursor-pointer bg-white rounded-md font-medium text-blue-600 hover:text-blue-500">
              <span>Upload files</span>
              <input
                type="file"
                multiple
                @change="handleFilesSelect"
                class="sr-only"
              />
            </label>
            <p class="pl-1">or drag and drop</p>
          </div>
          <p class="text-xs text-gray-500">PNG, JPG, GIF, PDF up to 10MB each</p>
        </div>
      </div>
    </div>

    <!-- Selected Files List -->
    <div v-if="selectedFiles.length > 0" class="space-y-2">
      <div
        v-for="(file, index) in selectedFiles"
        :key="index"
        class="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
      >
        <div class="flex items-center space-x-3">
          <span class="text-sm font-medium text-gray-700">{{ file.name }}</span>
          <span class="text-xs text-gray-500">{{ formatFileSize(file.size) }}</span>
        </div>
        <button
          type="button"
          @click="removeFile(index)"
          class="text-red-600 hover:text-red-800 text-lg"
        >
          &times;
        </button>
      </div>
    </div>

    <button
      type="submit"
      :disabled="form.processing"
      class="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50"
    >
      Upload Files
    </button>
  </form>
</template>

<script setup>
import { useForm } from '@inertiajs/inertia-vue3'
import { ref } from 'vue'
import { useFileUpload } from '@/composables/useFileUpload'

const isDragging = ref(false)
const uploadProgress = ref(0)

const { selectedFiles, handleFilesSelect, removeFile, clearAllFiles } = useFileUpload()

const form = useForm({
  files: [],
})

const onDragOver = (event) => {
  event.preventDefault()
  isDragging.value = true
}

const onDragLeave = () => {
  isDragging.value = false
}

const onDrop = (event) => {
  event.preventDefault()
  isDragging.value = false
  
  const files = Array.from(event.dataTransfer.files)
  handleFilesSelect({ target: { files } })
}

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const submit = () => {
  if (selectedFiles.value.length === 0) {
    form.errors.files = 'Please select at least one file'
    return
  }

  form.files = selectedFiles.value

  form.post('/upload', {
    data: form.data(),
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onProgress: (event) => {
      if (event.lengthComputable) {
        uploadProgress.value = (event.loaded / event.total) * 100
      }
    },
    onSuccess: () => {
      uploadProgress.value = 0
      clearAllFiles()
    },
    onError: () => {
      uploadProgress.value = 0
    },
  })
}
</script>
```

## 5. Laravel Model and Migration

```php
// migration
public function up()
{
    Schema::create('profiles', function (Blueprint $table) {
        $table->id();
        $table->string('name');
        $table->string('image_path')->nullable();
        $table->timestamps();
    });

    Schema::create('profile_files', function (Blueprint $table) {
        $table->id();
        $table->foreignId('profile_id')->constrained()->onDelete('cascade');
        $table->string('original_name');
        $table->string('path');
        $table->string('mime_type');
        $table->unsignedBigInteger('size');
        $table->timestamps();
    });
}

// Profile model
class Profile extends Model
{
    protected $fillable = ['name', 'image_path'];

    public function files()
    {
        return $this->hasMany(ProfileFile::class);
    }
}

// ProfileFile model
class ProfileFile extends Model
{
    protected $fillable = [
        'profile_id', 'original_name', 'path', 'mime_type', 'size'
    ];

    public function profile()
    {
        return $this->belongsTo(Profile::class);
    }

    public function getUrlAttribute()
    {
        return Storage::disk('public')->url($this->path);
    }
}
```

This implementation provides:

1. **Image previews** before upload
2. **Drag and drop** functionality
3. **File validation** (type, size)
4. **Progress tracking** for uploads
5. **Multiple file uploads** support
6. **Seamless integration** with Laravel validation
7. **Proper error handling** for both client and server
8. **File management** (delete old files when updating)

The files are stored in Laravel's storage system with proper validation and error handling throughout the process.