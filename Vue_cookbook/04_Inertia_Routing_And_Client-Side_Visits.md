Of course. Inertia's navigation is one of its core features, providing a seamless SPA experience without the complexity of managing a full client-side router like Vue Router independently.

### **04 - Navigation with Inertia.js: `<Link>` and `router.visit()`**

This guide explains how to navigate between pages in an Inertia.js application using both declarative and programmatic methods.

---

#### **1. The `<Link>` Component (Declarative Navigation)**

The `<Link>` component is the primary way to create navigational links. It's similar to an HTML `<a>` tag but prevents full page reloads.

**Basic Usage:**
```vue
<script setup>
import { Link } from '@inertiajs/vue3';
</script>

<template>
  <div>
    <!-- Basic link -->
    <Link href="/about">About Us</Link>

    <!-- Link using named routes (RECOMMENDED) -->
    <Link :href="route('dashboard')">Dashboard</Link>

    <!-- Link with complex styling (often used with Tailwind) -->
    <Link
      :href="route('posts.show', post.id)"
      class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition"
    >
      View Post
    </Link>

    <!-- Link that preserves scroll position and state (great for index -> show pages) -->
    <Link
      :href="route('products.show', product.slug)"
      preserve-scroll
      preserve-state
    >
      {{ product.name }}
    </Link>
  </div>
</template>
```

**Key Props for `<Link>`:**

| Prop | Description | Example |
| :--- | :--- | :--- |
| `href` | The destination URL. | `href="/about"` or `:href="route('posts.edit', post.id)"` |
| `method` | HTTP method for the visit (default: `GET`). | `method="post"` for logging out |
| `as` | How to render the link (default: `a`). Can be `button`. | `as="button"` |
| `replace` | Replace the current history state instead of adding a new one. | `replace` |
| `preserve-scroll` | **Crucial:** Preserves the scroll position on the current page when you come back to it. | `preserve-scroll` |
| `preserve-state` | Preserves the current page's local component state (data, refs) when you come back to it. | `preserve-state` |
| `only` | Only include the specified props in the request. | `:only="['posts']"` |
| `headers` | Add custom headers to the request. | `:headers="{ 'X-Custom-Header': 'value' }"` |

---

#### **2. The `router` Object (Programmatic Navigation)**

Use the `router` object for navigation in response to events, like form submissions, button clicks, or after timeouts.

**Accessing the Router:**
You can import the router directly or use the `useForm` helper for form submissions.

```vue
<script setup>
import { router } from '@inertiajs/vue3';
import { useForm } from '@inertiajs/vue3';

// For a simple GET visit
const visitPage = () => {
  router.get(route('users.index'));
};

// For a form
const form = useForm({
  search: '',
});

const performSearch = () => {
  // This is a GET request with the form data as query parameters
  form.get(route('search.results'));
};
</script>
```

**Main Methods of `router`:**
| Method | Description | Example |
| :--- | :--- | :--- |
| `.visit()` | The most general method. Makes a visit with any HTTP method and data. | `router.visit(url, { options })` |
| `.get()` | Makes a GET visit. | `router.get(route('home'))` |
| `.post()` | Makes a POST visit. | `router.post(route('logout'))` |
| `.put()`/`.patch()` | Makes a PUT/PATCH visit. | `router.patch(route('profile.update'), data)` |
| `.delete()` | Makes a DELETE visit. | `router.delete(route('posts.destroy', post.id))` |
| `.reload()` | Reloads the current page data. | `router.reload()` |
| `.back()`/`.forward()` | Navigates back/forward in history. | `router.back()` |

**Complete Example: Product Search and View**

```vue
<script setup>
import { router, Link } from '@inertiajs/vue3';
import { ref } from 'vue';

const searchQuery = ref('');

// 1. Programmatic navigation for search (GET)
const handleSearch = () => {
  if (searchQuery.value) {
    router.get(route('products.search'), { 
      q: searchQuery.value 
    });
  }
};

// 2. Programmatic navigation for deletion (DELETE)
const deleteProduct = (productId) => {
  if (confirm('Are you sure you want to delete this product?')) {
    router.delete(route('products.destroy', productId), {
      // Optionally, show a success message on the same page after deletion
      onSuccess: () => {
        // Maybe show a toast notification
      },
    });
  }
};
</script>

<template>
  <div class="p-8">
    <h1>Products</h1>

    <!-- Search Form (uses programmatic GET) -->
    <div class="flex gap-2 mb-6">
      <input
        v-model="searchQuery"
        type="text"
        placeholder="Search products..."
        @keyup.enter="handleSearch"
        class="border rounded px-3 py-2"
      />
      <button @click="handleSearch" class="px-4 py-2 bg-blue-500 text-white rounded">
        Search
      </button>
    </div>

    <!-- Product List (uses declarative <Link>) -->
    <ul>
      <li v-for="product in products" :key="product.id" class="border-b py-4 flex justify-between">
        <Link
          :href="route('products.show', product.id)"
          class="text-blue-600 hover:underline"
          preserve-scroll
        >
          {{ product.name }}
        </Link>
        
        <button 
          @click="deleteProduct(product.id)" 
          class="text-red-500 hover:text-red-700"
        >
          Delete
        </button>
      </li>
    </ul>

    <!-- Link to create a new product -->
    <Link 
      :href="route('products.create')" 
      class="mt-6 inline-block px-4 py-2 bg-green-500 text-white rounded"
    >
      Add New Product
    </Link>
  </div>
</template>
```

---

#### **3. Managing Scroll Behavior**

Controlling scroll position is vital for UX. Inertia provides powerful options.

**a. Resetting Scroll on Navigation (Global)**
You can define a global scroll behavior in your `app.js` file when initializing Inertia.

```js
// resources/js/app.js
createInertiaApp({
  // ... other settings ...
  progress: { color: '#4B5563' },
  scrollBehavior: (to, from, savedPosition) => {
    // If it's a "back" navigation, restore the saved position
    if (savedPosition) {
      return savedPosition;
    }
    // Otherwise, default to scrolling to the top on new pages
    return { top: 0 };
  },
});
```

**b. Preserving Scroll on a Specific Link (`preserve-scroll`)**
This is essential for index pages. When a user clicks "View", then clicks "Back", they return to the exact same scroll position.

```vue
<Link 
  :href="route('posts.show', post.id)" 
  preserve-scroll 
  class="block py-2"
>
  Read More...
</Link>
```

**c. Scrolling to a Specific Element (`onSuccess`)**
You can scroll to a specific part of the page after a visit, useful for things like "Scroll to Top" buttons or focusing on a newly created item.

```vue
<script setup>
import { router } from '@inertiajs/vue3';

const goToTop = () => {
  router.get(route('long.page'), {}, {
    onSuccess: () => {
      // Scroll to top after the page loads
      window.scrollTo(0, 0);
      
      // Or, scroll to a specific element
      // document.getElementById('section-1').scrollIntoView();
    }
  });
};
</script>
```

---

#### **Summary of Best Practices**

1.  **Use `<Link>` for Navigation Menus:** Always use `<Link>` for standard navigation between pages. It's the most efficient and semantic choice.
2.  **Use `router` for Actions:** Use `router.get()`, `router.post()`, etc., for events like form submissions, button clicks, or deletions.
3.  **Always Use `route()` Helper:** Never hard-code URLs. Use the `route()` function to generate them based on your Laravel route names. This ensures consistency between backend and frontend.
4.  **Use `preserve-scroll` Liberally:** Add `preserve-scroll` to any link that goes to a detail page where the user is likely to click the back button. This dramatically improves UX.
5.  **Leverage `onSuccess` and other Events:** Use the visit options (`onSuccess`, `onError`, `onFinish`) to add side effects like showing notifications, resetting forms, or managing scroll.