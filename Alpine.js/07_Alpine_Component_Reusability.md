# Creating a Reusable Modal Component with Alpine.data()

Here's how to extract modal logic into a reusable component using `Alpine.data()` that can accept dynamic content:

## 1. Define the Reusable Modal Component

```javascript
document.addEventListener('alpine:init', () => {
    Alpine.data('modal', (config = {}) => ({
        open: false,
        title: config.title || 'Modal Title',
        content: config.content || 'Default modal content',
        maxWidth: config.maxWidth || 'max-w-md',
        
        init() {
            // Set up any initial configuration here
        },
        
        openModal(title = null, content = null) {
            if (title) this.title = title;
            if (content) this.content = content;
            this.open = true;
        },
        
        closeModal() {
            this.open = false;
        }
    }));
});
```

## 2. Use the Component in Your HTML

```html
<!-- Example 1: Basic Usage -->
<div x-data="modal()">
    <button @click="openModal()" class="btn-primary">
        Open Default Modal
    </button>
    
    <template x-teleport="body">
        <!-- Backdrop -->
        <div x-show="open" 
             x-transition.opacity
             class="fixed inset-0 bg-black/50 z-40"></div>
        
        <!-- Modal Content -->
        <div x-show="open"
             @click.away="closeModal()"
             x-transition
             class="fixed inset-0 flex items-center justify-center z-50 p-4">
            <div :class="['bg-white rounded-lg shadow-xl w-full', maxWidth]">
                <div class="p-6">
                    <h2 x-text="title" class="text-xl font-bold mb-4"></h2>
                    <div x-html="content" class="mb-4"></div>
                    <button @click="closeModal()" class="btn-secondary">
                        Close
                    </button>
                </div>
            </div>
        </div>
    </template>
</div>

<!-- Example 2: Customized Modal -->
<div x-data="modal({
    title: 'Custom Title',
    content: '<p>This is <strong>custom</strong> content!</p>',
    maxWidth: 'max-w-lg'
})">
    <button @click="openModal()" class="btn-primary">
        Open Custom Modal
    </button>
    
    <!-- Same template as above -->
</div>

<!-- Example 3: Dynamic Content -->
<div x-data="modal()">
    <button @click="openModal('Dynamic Title', '<p>Loaded dynamically!</p>')" 
            class="btn-primary">
        Open Dynamic Modal
    </button>
    
    <!-- Same template as above -->
</div>
```

## Key Features of This Implementation

1. **Configurable Defaults**:
   - Title, content, and max width can be set during initialization
   - Sensible defaults provided for all properties

2. **Dynamic Content Support**:
   - The `openModal()` method accepts parameters to change content on the fly
   - Uses `x-html` to render HTML content safely

3. **Responsive Design**:
   - Uses Tailwind's max-width utilities (`max-w-md`, `max-w-lg`, etc.)
   - Responsive padding and sizing

4. **Accessibility**:
   - Backdrop click closes modal (`@click.away`)
   - Proper z-index stacking

5. **Teleportation**:
   - Modal is teleported to the `<body>` element to avoid z-index issues

## Advanced Usage

For more complex scenarios, you could extend this component:

```javascript
Alpine.data('advancedModal', (config = {}) => ({
    ...Alpine.data('modal')(config), // Inherit from base modal
    showFooter: config.showFooter ?? true,
    footerContent: config.footerContent || '',
    
    // Custom methods
    confirmAction() {
        // Handle confirmation logic
        this.closeModal();
    }
}));
```

This approach gives you a highly reusable modal component that can be customized for different use cases while maintaining consistent behavior and animations across your application.