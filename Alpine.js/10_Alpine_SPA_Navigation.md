# Client-Side Routing with Alpine.js for Multi-Tab Interface

Let's create a tabbed interface that switches content without page reloads using Alpine.js. This solution will include:
- Tab navigation
- Content switching
- URL hash updates
- History support

## 1. Basic Tab Structure

```html
<div x-data="{
        activeTab: 'home',
        tabs: [
            { id: 'home', label: 'Home' },
            { id: 'products', label: 'Products' },
            { id: 'about', label: 'About' },
            { id: 'contact', label: 'Contact' }
        ],
        init() {
            // Check URL hash on load
            if (window.location.hash) {
                const hash = window.location.hash.substring(1);
                if (this.tabs.some(tab => tab.id === hash)) {
                    this.activeTab = hash;
                }
            }
            
            // Update hash when tab changes
            this.$watch('activeTab', (value) => {
                window.location.hash = value;
            });
        }
    }" 
    class="max-w-4xl mx-auto">
    
    <!-- Tab Navigation -->
    <div class="flex border-b">
        <template x-for="tab in tabs" :key="tab.id">
            <button @click="activeTab = tab.id"
                    :class="{
                        'border-blue-500 text-blue-600': activeTab === tab.id,
                        'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300': activeTab !== tab.id
                    }"
                    class="py-4 px-6 text-center border-b-2 font-medium">
                <span x-text="tab.label"></span>
            </button>
        </template>
    </div>
    
    <!-- Tab Content -->
    <div class="p-6">
        <div x-show="activeTab === 'home'">
            <h2 class="text-2xl font-bold mb-4">Home Content</h2>
            <p>Welcome to our website! This is the home tab content.</p>
        </div>
        
        <div x-show="activeTab === 'products'" class="space-y-4">
            <h2 class="text-2xl font-bold mb-4">Our Products</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="border p-4 rounded-lg">
                    <h3 class="font-bold">Product 1</h3>
                    <p class="text-gray-600">Description of product 1</p>
                </div>
                <!-- More products... -->
            </div>
        </div>
        
        <div x-show="activeTab === 'about'">
            <h2 class="text-2xl font-bold mb-4">About Us</h2>
            <p>Learn more about our company and mission.</p>
        </div>
        
        <div x-show="activeTab === 'contact'">
            <h2 class="text-2xl font-bold mb-4">Contact Information</h2>
            <p>Email us at: contact@example.com</p>
        </div>
    </div>
</div>
```

## 2. Advanced Version with Dynamic Content Loading

For larger applications, you might want to load content dynamically:

```html
<div x-data="{
        activeTab: 'home',
        tabs: [
            { id: 'home', label: 'Home' },
            { id: 'dashboard', label: 'Dashboard' },
            { id: 'settings', label: 'Settings' }
        ],
        isLoading: false,
        tabContent: {},
        async loadTabContent(tabId) {
            this.isLoading = true;
            
            // Simulate API call or content loading
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // In a real app, you might fetch from an API:
            // const response = await fetch(`/api/tabs/${tabId}`);
            // this.tabContent[tabId] = await response.json();
            
            // Mock content
            const mockContent = {
                home: '<h2 class="text-2xl font-bold mb-4">Home Content</h2><p>Dynamically loaded home content.</p>',
                dashboard: '<h2 class="text-2xl font-bold mb-4">Dashboard</h2><p>Your dashboard metrics would appear here.</p>',
                settings: '<h2 class="text-2xl font-bold mb-4">Settings</h2><form class="space-y-4">...</form>'
            };
            
            this.tabContent[tabId] = mockContent[tabId];
            this.isLoading = false;
        },
        init() {
            // Initial load
            this.loadTabContent(this.activeTab);
            
            // Handle hash changes
            window.addEventListener('hashchange', () => {
                const hash = window.location.hash.substring(1);
                if (this.tabs.some(tab => tab.id === hash)) {
                    this.activeTab = hash;
                    if (!this.tabContent[hash]) {
                        this.loadTabContent(hash);
                    }
                }
            });
        }
    }">
    
    <!-- Tab Navigation -->
    <div class="flex border-b">
        <template x-for="tab in tabs" :key="tab.id">
            <button @click="activeTab = tab.id; if (!tabContent[tab.id]) loadTabContent(tab.id)"
                    :class="{
                        'border-blue-500 text-blue-600': activeTab === tab.id,
                        'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300': activeTab !== tab.id
                    }"
                    class="py-4 px-6 text-center border-b-2 font-medium">
                <span x-text="tab.label"></span>
            </button>
        </template>
    </div>
    
    <!-- Tab Content -->
    <div class="p-6 min-h-64">
        <template x-if="isLoading">
            <div class="flex justify-center items-center py-8">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            </div>
        </template>
        
        <template x-if="!isLoading">
            <div x-html="tabContent[activeTab]"></div>
        </template>
    </div>
</div>
```

## 3. With Transition Animations

Add smooth transitions between tabs:

```html
<div x-data="{
        // ... same data as basic example ...
    }">
    
    <!-- ... same tab navigation ... -->
    
    <!-- Tab Content with Transitions -->
    <div class="p-6 relative overflow-hidden">
        <template x-for="tab in tabs" :key="tab.id">
            <div x-show="activeTab === tab.id"
                 x-transition:enter="transition ease-out duration-300"
                 x-transition:enter-start="opacity-0 translate-x-4"
                 x-transition:enter-end="opacity-100 translate-x-0"
                 x-transition:leave="transition ease-in duration-200"
                 x-transition:leave-start="opacity-100 translate-x-0"
                 x-transition:leave-end="opacity-0 -translate-x-4"
                 class="absolute inset-0 p-6">
                <template x-if="activeTab === 'home'">
                    <!-- Home content -->
                </template>
                
                <!-- Other tabs content -->
            </div>
        </template>
    </div>
</div>
```

## Key Features

1. **URL Synchronization**:
   - Updates browser hash when tabs change
   - Reads hash on page load
   - Supports back/forward navigation

2. **Responsive Design**:
   - Works on mobile and desktop
   - Clean, modern UI with Tailwind CSS

3. **Performance**:
   - Basic version shows/hides existing content
   - Advanced version loads content on demand

4. **Extensible**:
   - Easy to add more tabs
   - Can integrate with backend APIs
   - Supports transitions and loading states

5. **Accessibility**:
   - Semantic HTML structure
   - Keyboard navigable
   - ARIA attributes can be easily added

This implementation gives you a solid foundation that you can customize further by:
- Adding icons to tabs
- Implementing permission-based tab visibility
- Adding swipe gestures for mobile
- Persisting tab state in localStorage
- Integrating with a backend router for hybrid apps