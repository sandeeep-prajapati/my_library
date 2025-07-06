# **Fetch and Display API Data with Alpine.js + Loading Spinner**

Here's a complete solution to fetch posts from JSONPlaceholder API with a loading spinner:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <style>
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .post { 
            border: 1px solid #ddd;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        .error { color: red; }
    </style>
</head>
<body>
    <div 
        x-data="{
            posts: [],
            isLoading: false,
            error: null,
            fetchPosts() {
                this.isLoading = true;
                this.error = null;
                fetch('https://jsonplaceholder.typicode.com/posts')
                    .then(response => {
                        if (!response.ok) throw new Error('Network error');
                        return response.json();
                    })
                    .then(data => this.posts = data.slice(0, 5)) // Show first 5 posts
                    .catch(err => this.error = err.message)
                    .finally(() => this.isLoading = false);
            }
        }"
        x-init="fetchPosts()"
    >
        <h1>Recent Posts</h1>
        
        <!-- Loading Spinner -->
        <div x-show="isLoading" class="loader"></div>
        
        <!-- Error Message -->
        <p x-show="error" class="error" x-text="'Error: ' + error"></p>
        
        <!-- Posts List -->
        <template x-if="!isLoading && !error">
            <div>
                <template x-for="post in posts" :key="post.id">
                    <div class="post">
                        <h3 x-text="post.title"></h3>
                        <p x-text="post.body"></p>
                    </div>
                </template>
            </div>
        </template>
        
        <!-- Refresh Button -->
        <button 
            @click="fetchPosts()"
            :disabled="isLoading"
            class="mt-4 px-4 py-2 bg-blue-500 text-white rounded"
        >
            <span x-show="!isLoading">Refresh Posts</span>
            <span x-show="isLoading">Loading...</span>
        </button>
    </div>
</body>
</html>
```

## **Key Features Explained**

### **1. Data Fetching**
- `fetchPosts()` function:
  - Sets `isLoading` to `true` before fetching
  - Uses `fetch()` to get data from JSONPlaceholder
  - Handles success/error states
  - Automatically updates reactive `posts` array

### **2. Loading State**
- Spinner appears while `isLoading` is true:
  ```html
  <div x-show="isLoading" class="loader"></div>
  ```

### **3. Error Handling**
- Displays error message if fetch fails:
  ```html
  <p x-show="error" class="error" x-text="'Error: ' + error"></p>
  ```

### **4. Displaying Data**
- Renders posts only when not loading and no errors:
  ```html
  <template x-if="!isLoading && !error">
  ```
- Uses `x-for` to loop through posts

### **5. Refresh Capability**
- Button re-fetches data when clicked:
  ```html
  <button @click="fetchPosts()">
  ```

## **Enhancements You Can Add**

1. **Pagination**:
```javascript
fetch(`https://jsonplaceholder.typicode.com/posts?_page=${this.page}&_limit=5`)
```

2. **Search Filter**:
```html
<input x-model="searchTerm" placeholder="Search posts...">
<!-- Then filter posts -->
<template x-for="post in posts.filter(p => p.title.includes(searchTerm))">
```

3. **Skeleton Loader** (instead of spinner):
```html
<div x-show="isLoading" class="skeleton-loader">
    <div class="h-4 bg-gray-200 mb-2"></div>
    <div class="h-4 bg-gray-200 w-3/4"></div>
</div>
```

This implementation gives users visual feedback during loading, handles errors gracefully, and provides a clean way to refresh data - all with Alpine.js's reactive system.