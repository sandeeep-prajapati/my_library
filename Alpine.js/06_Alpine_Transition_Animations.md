# Smooth Transitions with x-transition for Modals

To create smooth entrance and exit animations for a modal using Alpine.js's `x-transition`, we'll combine fade and scale effects. Here's how to implement it:

## Basic Modal Structure with Transitions

```html
<div x-data="{ open: false }">
    <!-- Trigger Button -->
    <button @click="open = true" class="bg-blue-500 text-white px-4 py-2 rounded">
        Open Modal
    </button>

    <!-- Modal Backdrop -->
    <div x-show="open"
         x-transition:enter="ease-out duration-300"
         x-transition:enter-start="opacity-0"
         x-transition:enter-end="opacity-100"
         x-transition:leave="ease-in duration-200"
         x-transition:leave-start="opacity-100"
         x-transition:leave-end="opacity-0"
         class="fixed inset-0 bg-black bg-opacity-50 z-40">
    </div>

    <!-- Modal Content -->
    <div x-show="open"
         @click.away="open = false"
         x-transition:enter="ease-out duration-300"
         x-transition:enter-start="opacity-0 scale-95"
         x-transition:enter-end="opacity-100 scale-100"
         x-transition:leave="ease-in duration-200"
         x-transition:leave-start="opacity-100 scale-100"
         x-transition:leave-end="opacity-0 scale-95"
         class="fixed inset-0 flex items-center justify-center z-50">
        
        <div class="bg-white p-6 rounded-lg shadow-xl max-w-md w-full mx-4">
            <h2 class="text-xl font-bold mb-4">Modal Title</h2>
            <p class="mb-4">This modal has smooth fade and scale animations!</p>
            <button @click="open = false" class="bg-gray-200 px-4 py-2 rounded">
                Close
            </button>
        </div>
    </div>
</div>
```

## How It Works

1. **Backdrop Animation**:
   - Fades in smoothly when appearing (`ease-out` over 300ms)
   - Fades out when disappearing (`ease-in` over 200ms)

2. **Modal Content Animation**:
   - **Entrance**: Starts at 95% scale and fully transparent, animates to full size and opacity
   - **Exit**: Shrinks back to 95% scale while fading out

## Customizing the Animation

You can adjust these aspects:

1. **Timing**:
   - Change `duration-300` to other values like `duration-200` or `duration-500`

2. **Easing**:
   - Replace `ease-out`/`ease-in` with other curves like `linear` or custom cubic-bezier

3. **Scale Amount**:
   - Adjust `scale-95` to values like `scale-90` for more dramatic effect

## Tailwind CSS Requirement

For this to work, you need Tailwind CSS with the transition plugin enabled in your `tailwind.config.js`:

```js
module.exports = {
    // ...
    plugins: [
        require('@tailwindcss/forms'),
        require('@tailwindcss/typography'),
        // other plugins
    ]
}
```

This creates a professional-looking modal with smooth, coordinated animations for both the backdrop and content.