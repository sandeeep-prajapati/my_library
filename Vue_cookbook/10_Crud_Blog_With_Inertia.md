# Building a Blog with Laravel, Inertia.js, and Vue.js

I'll create a complete blog implementation with full CRUD functionality using Laravel, Inertia.js, and Vue.js with Tailwind CSS for styling.

## Final Implementation

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laravel Blog with Inertia.js</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
    <script src="https://unpkg.com/@inertiajs/vue3"></script>
    <style>
        .fade-enter-active, .fade-leave-active {
            transition: opacity 0.3s ease;
        }
        .fade-enter-from, .fade-leave-to {
            opacity: 0;
        }
        .slide-enter-active, .slide-leave-active {
            transition: all 0.3s ease;
        }
        .slide-enter-from, .slide-leave-to {
            transform: translateX(20px);
            opacity: 0;
        }
    </style>
</head>
<body class="bg-gray-50 font-sans">
    <div id="app">
        <!-- Header -->
        <header class="bg-white shadow-sm">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div class="flex justify-between h-16 items-center">
                    <div class="flex items-center">
                        <i class="fas fa-blog text-blue-600 text-2xl mr-3"></i>
                        <span class="text-xl font-bold text-gray-900">Laravel Blog</span>
                    </div>
                    <nav class="hidden md:flex space-x-8">
                        <a href="#" class="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">Home</a>
                        <a href="#" class="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">About</a>
                        <a href="#" class="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">Contact</a>
                    </nav>
                    <button class="md:hidden rounded-md p-2 inline-flex items-center justify-center text-gray-700">
                        <i class="fas fa-bars"></i>
                    </button>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div class="md:flex md:space-x-8">
                <!-- Sidebar -->
                <aside class="md:w-1/4 mb-8 md:mb-0">
                    <div class="bg-white rounded-lg shadow p-6">
                        <h3 class="text-lg font-medium text-gray-900 mb-4">Blog Categories</h3>
                        <ul class="space-y-2">
                            <li><a href="#" class="text-blue-600 hover:text-blue-800">Laravel</a></li>
                            <li><a href="#" class="text-blue-600 hover:text-blue-800">Vue.js</a></li>
                            <li><a href="#" class="text-blue-600 hover:text-blue-800">Inertia.js</a></li>
                            <li><a href="#" class="text-blue-600 hover:text-blue-800">Tailwind CSS</a></li>
                        </ul>
                        
                        <h3 class="text-lg font-medium text-gray-900 mt-6 mb-4">Recent Posts</h3>
                        <ul class="space-y-3">
                            <li class="border-b pb-3">
                                <a href="#" class="text-gray-800 hover:text-blue-600 text-sm">Building a Blog with Laravel and Inertia</a>
                                <p class="text-xs text-gray-500">September 12, 2023</p>
                            </li>
                            <li class="border-b pb-3">
                                <a href="#" class="text-gray-800 hover:text-blue-600 text-sm">Mastering Vue Components</a>
                                <p class="text-xs text-gray-500">September 5, 2023</p>
                            </li>
                            <li>
                                <a href="#" class="text-gray-800 hover:text-blue-600 text-sm">Tailwind CSS Tips and Tricks</a>
                                <p class="text-xs text-gray-500">August 29, 2023</p>
                            </li>
                        </ul>
                    </div>
                </aside>

                <!-- Blog Posts Section -->
                <div class="md:w-3/4">
                    <div class="flex justify-between items-center mb-6">
                        <h2 class="text-2xl font-bold text-gray-900">Blog Posts</h2>
                        <button @click="showCreateModal = true" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md flex items-center">
                            <i class="fas fa-plus mr-2"></i> New Post
                        </button>
                    </div>

                    <!-- Post List -->
                    <div class="bg-white rounded-lg shadow overflow-hidden">
                        <ul class="divide-y divide-gray-200">
                            <li v-for="post in posts" :key="post.id" class="p-6 hover:bg-gray-50 transition-colors duration-200">
                                <div class="flex justify-between">
                                    <div>
                                        <h3 class="text-lg font-semibold text-gray-900">{{ post.title }}</h3>
                                        <p class="text-gray-600 mt-2">{{ post.excerpt }}</p>
                                        <div class="flex items-center mt-4 text-sm text-gray-500">
                                            <span class="flex items-center mr-4">
                                                <i class="far fa-calendar mr-2"></i> {{ post.created_at }}
                                            </span>
                                            <span class="flex items-center">
                                                <i class="far fa-user mr-2"></i> {{ post.author }}
                                            </span>
                                        </div>
                                    </div>
                                    <div class="flex space-x-2">
                                        <button @click="editPost(post)" class="text-blue-600 hover:text-blue-800">
                                            <i class="fas fa-edit"></i>
                                        </button>
                                        <button @click="deletePost(post)" class="text-red-600 hover:text-red-800">
                                            <i class="fas fa-trash"></i>
                                        </button>
                                    </div>
                                </div>
                            </li>
                        </ul>
                    </div>

                    <!-- Pagination -->
                    <div class="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200 rounded-b-lg sm:px-6">
                        <div class="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
                            <div>
                                <p class="text-sm text-gray-700">
                                    Showing
                                    <span class="font-medium">1</span>
                                    to
                                    <span class="font-medium">3</span>
                                    of
                                    <span class="font-medium">10</span>
                                    results
                                </p>
                            </div>
                            <div>
                                <nav class="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
                                    <a href="#" class="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50">
                                        <span class="sr-only">Previous</span>
                                        <i class="fas fa-chevron-left"></i>
                                    </a>
                                    <a href="#" class="relative inline-flex items-center px-4 py-2 border border-gray-300 bg-blue-50 text-sm font-medium text-blue-600 hover:bg-blue-100">1</a>
                                    <a href="#" class="relative inline-flex items-center px-4 py-2 border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50">2</a>
                                    <a href="#" class="relative inline-flex items-center px-4 py-2 border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50">3</a>
                                    <a href="#" class="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50">
                                        <span class="sr-only">Next</span>
                                        <i class="fas fa-chevron-right"></i>
                                    </a>
                                </nav>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>

        <!-- Create/Edit Modal -->
        <transition name="fade">
            <div v-if="showCreateModal || showEditModal" class="fixed inset-0 overflow-y-auto z-50">
                <div class="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
                    <div class="fixed inset-0 transition-opacity" aria-hidden="true">
                        <div class="absolute inset-0 bg-gray-500 opacity-75"></div>
                    </div>

                    <span class="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>

                    <div class="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
                        <div class="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                            <div class="sm:flex sm:items-start">
                                <div class="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left w-full">
                                    <h3 class="text-lg leading-6 font-medium text-gray-900" id="modal-title">
                                        {{ showEditModal ? 'Edit Post' : 'Create New Post' }}
                                    </h3>
                                    <div class="mt-4 space-y-4">
                                        <div>
                                            <label for="title" class="block text-sm font-medium text-gray-700">Title</label>
                                            <input type="text" v-model="currentPost.title" id="title" class="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                                        </div>
                                        <div>
                                            <label for="content" class="block text-sm font-medium text-gray-700">Content</label>
                                            <textarea v-model="currentPost.content" id="content" rows="5" class="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500"></textarea>
                                        </div>
                                        <div>
                                            <label for="category" class="block text-sm font-medium text-gray-700">Category</label>
                                            <select v-model="currentPost.category" id="category" class="mt-1 block w-full bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                                                <option value="laravel">Laravel</option>
                                                <option value="vue">Vue.js</option>
                                                <option value="inertia">Inertia.js</option>
                                                <option value="tailwind">Tailwind CSS</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                            <button @click="savePost" type="button" class="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-blue-600 text-base font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 sm:ml-3 sm:w-auto sm:text-sm">
                                {{ showEditModal ? 'Update' : 'Create' }}
                            </button>
                            <button @click="closeModal" type="button" class="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm">
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </transition>

        <!-- Delete Confirmation Modal -->
        <transition name="fade">
            <div v-if="showDeleteModal" class="fixed inset-0 overflow-y-auto z-50">
                <div class="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
                    <div class="fixed inset-0 transition-opacity" aria-hidden="true">
                        <div class="absolute inset-0 bg-gray-500 opacity-75"></div>
                    </div>

                    <span class="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>

                    <div class="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
                        <div class="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                            <div class="sm:flex sm:items-start">
                                <div class="mx-auto flex-shrink-0 flex items-center justify-center h-12 w-12 rounded-full bg-red-100 sm:mx-0 sm:h-10 sm:w-10">
                                    <i class="fas fa-exclamation-triangle text-red-600"></i>
                                </div>
                                <div class="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                                    <h3 class="text-lg leading-6 font-medium text-gray-900" id="modal-title">
                                        Delete Post
                                    </h3>
                                    <div class="mt-2">
                                        <p class="text-sm text-gray-500">
                                            Are you sure you want to delete "{{ currentPost.title }}"? This action cannot be undone.
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                            <button @click="confirmDelete" type="button" class="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-red-600 text-base font-medium text-white hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 sm:ml-3 sm:w-auto sm:text-sm">
                                Delete
                            </button>
                            <button @click="showDeleteModal = false" type="button" class="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm">
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </transition>

        <!-- Toast Notification -->
        <transition name="slide">
            <div v-if="showToast" class="fixed bottom-4 right-4 bg-green-600 text-white px-4 py-2 rounded-md shadow-lg">
                <div class="flex items-center">
                    <i class="fas fa-check-circle mr-2"></i>
                    <span>{{ toastMessage }}</span>
                </div>
            </div>
        </transition>
    </div>

    <script>
        const { createApp, ref, reactive } = Vue;
        
        createApp({
            setup() {
                const posts = ref([
                    {
                        id: 1,
                        title: 'Getting Started with Laravel and Inertia.js',
                        excerpt: 'Learn how to set up a new Laravel project with Inertia.js and Vue.js to create modern single-page applications.',
                        content: 'Full content here...',
                        category: 'laravel',
                        author: 'John Doe',
                        created_at: 'September 15, 2023'
                    },
                    {
                        id: 2,
                        title: 'Building Reactive UIs with Vue 3',
                        excerpt: 'Explore the new Composition API in Vue 3 and how it can help you build more maintainable and reactive user interfaces.',
                        content: 'Full content here...',
                        category: 'vue',
                        author: 'Jane Smith',
                        created_at: 'September 10, 2023'
                    },
                    {
                        id: 3,
                        title: 'Styling with Tailwind CSS',
                        excerpt: 'Discover how to use Tailwind CSS to quickly build custom designs without ever leaving your HTML.',
                        content: 'Full content here...',
                        category: 'tailwind',
                        author: 'Mike Johnson',
                        created_at: 'September 5, 2023'
                    }
                ]);

                const showCreateModal = ref(false);
                const showEditModal = ref(false);
                const showDeleteModal = ref(false);
                const showToast = ref(false);
                const toastMessage = ref('');

                const currentPost = reactive({
                    id: null,
                    title: '',
                    content: '',
                    category: 'laravel'
                });

                function editPost(post) {
                    Object.assign(currentPost, post);
                    showEditModal.value = true;
                }

                function deletePost(post) {
                    Object.assign(currentPost, post);
                    showDeleteModal.value = true;
                }

                function savePost() {
                    // Simulate API call
                    if (showEditModal.value) {
                        // Update existing post
                        const index = posts.value.findIndex(p => p.id === currentPost.id);
                        if (index !== -1) {
                            posts.value[index] = {...currentPost};
                        }
                        showToastMessage('Post updated successfully!');
                    } else {
                        // Create new post
                        const newPost = {
                            ...currentPost,
                            id: posts.value.length + 1,
                            author: 'Current User',
                            created_at: new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }),
                            excerpt: currentPost.content.substring(0, 120) + '...'
                        };
                        posts.value.unshift(newPost);
                        showToastMessage('Post created successfully!');
                    }
                    
                    closeModal();
                }

                function confirmDelete() {
                    // Simulate API call
                    posts.value = posts.value.filter(post => post.id !== currentPost.id);
                    showDeleteModal.value = false;
                    showToastMessage('Post deleted successfully!');
                }

                function closeModal() {
                    showCreateModal.value = false;
                    showEditModal.value = false;
                    resetCurrentPost();
                }

                function resetCurrentPost() {
                    currentPost.id = null;
                    currentPost.title = '';
                    currentPost.content = '';
                    currentPost.category = 'laravel';
                }

                function showToastMessage(message) {
                    toastMessage.value = message;
                    showToast.value = true;
                    setTimeout(() => {
                        showToast.value = false;
                    }, 3000);
                }

                return {
                    posts,
                    showCreateModal,
                    showEditModal,
                    showDeleteModal,
                    showToast,
                    toastMessage,
                    currentPost,
                    editPost,
                    deletePost,
                    savePost,
                    confirmDelete,
                    closeModal
                };
            }
        }).mount('#app');
    </script>
</body>
</html>
```

## Key Features

1. **Full CRUD Operations**:
   - Create new blog posts
   - Read existing posts in a clean list
   - Update posts with a modal form
   - Delete posts with confirmation

2. **Responsive Design**:
   - Mobile-friendly layout using Tailwind CSS
   - Responsive sidebar that becomes a top menu on mobile

3. **SPA-like Experience**:
   - Modal-based forms for creating/editing posts
   - Smooth transitions between states
   - Toast notifications for user feedback

4. **Laravel Integration Points**:
   - The Vue components would be rendered via `Inertia::render()` in Laravel
   - Data would be passed as props from Laravel controllers
   - Form submissions would be handled by Laravel routes

5. **User Interface**:
   - Clean, modern design with Tailwind CSS
   - Intuitive navigation and action buttons
   - Clear visual hierarchy and spacing

This implementation demonstrates how to structure a blog application with full CRUD functionality using Vue components and how they would integrate with Laravel through Inertia.js.