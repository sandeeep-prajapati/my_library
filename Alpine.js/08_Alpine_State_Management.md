# Global Shopping Cart State with Alpine.store

Let's create a real-time shopping cart system that shares state across multiple components using `Alpine.store`. This solution will include:
- A centralized cart store
- Product display components
- A cart summary component
- Real-time updates across all instances

## 1. Initialize the Global Store

```javascript
document.addEventListener('alpine:init', () => {
    Alpine.store('cart', {
        items: [],
        
        init() {
            // Load from localStorage if available
            const savedCart = localStorage.getItem('alpine-cart');
            if (savedCart) this.items = JSON.parse(savedCart);
        },
        
        addItem(product) {
            const existingItem = this.items.find(item => item.id === product.id);
            
            if (existingItem) {
                existingItem.quantity++;
            } else {
                this.items.push({ ...product, quantity: 1 });
            }
            
            this.persistCart();
        },
        
        removeItem(id) {
            this.items = this.items.filter(item => item.id !== id);
            this.persistCart();
        },
        
        updateQuantity(id, newQuantity) {
            const item = this.items.find(item => item.id === id);
            if (item) {
                item.quantity = Math.max(1, newQuantity);
                this.persistCart();
            }
        },
        
        get totalItems() {
            return this.items.reduce((sum, item) => sum + item.quantity, 0);
        },
        
        get subtotal() {
            return this.items.reduce((sum, item) => sum + (item.price * item.quantity), 0);
        },
        
        persistCart() {
            localStorage.setItem('alpine-cart', JSON.stringify(this.items));
        },
        
        clearCart() {
            this.items = [];
            this.persistCart();
        }
    });
});
```

## 2. Product Component (Multiple Instances)

```html
<div x-data="{
        product: {
            id: 1,
            name: 'Premium Headphones',
            price: 199.99,
            image: '/images/headphones.jpg'
        }
    }" 
    class="border p-4 rounded-lg">
    <img :src="product.image" :alt="product.name" class="w-full h-40 object-cover mb-2">
    <h3 x-text="product.name" class="font-bold"></h3>
    <p x-text="`$${product.price.toFixed(2)}`" class="text-gray-600 mb-3"></p>
    
    <button @click="$store.cart.addItem(product)"
            class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded">
        Add to Cart
    </button>
</div>
```

## 3. Cart Summary Component (Always Visible)

```html
<div x-data class="fixed bottom-4 right-4 bg-white shadow-lg rounded-lg p-4 z-50">
    <div class="flex items-center justify-between mb-2">
        <h3 class="font-bold">Your Cart</h3>
        <span x-text="$store.cart.totalItems" 
              class="bg-blue-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
        </span>
    </div>
    
    <template x-if="$store.cart.items.length > 0">
        <div>
            <div class="max-h-64 overflow-y-auto mb-3">
                <template x-for="item in $store.cart.items" :key="item.id">
                    <div class="flex items-center justify-between py-2 border-b">
                        <div class="flex-1 truncate">
                            <span x-text="item.name"></span>
                            <span x-text="`×${item.quantity}`" class="text-sm text-gray-500"></span>
                        </div>
                        <div class="flex items-center">
                            <span x-text="`$${(item.price * item.quantity).toFixed(2)}`" 
                                  class="mr-3"></span>
                            <button @click="$store.cart.removeItem(item.id)"
                                    class="text-red-500 hover:text-red-700">
                                &times;
                            </button>
                        </div>
                    </div>
                </template>
            </div>
            
            <div class="font-bold mb-3">
                Subtotal: $<span x-text="$store.cart.subtotal.toFixed(2)"></span>
            </div>
            
            <button @click="$store.cart.clearCart()"
                    class="text-sm text-red-500 hover:text-red-700 mr-3">
                Clear Cart
            </button>
            <button class="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded">
                Checkout
            </button>
        </div>
    </template>
    
    <template x-if="$store.cart.items.length === 0">
        <p class="text-gray-500">Your cart is empty</p>
    </template>
</div>
```

## 4. Quantity Editor Component (Reusable)

```html
<div x-data="{ id: null, quantity: 1 }" 
     x-init="id = $el.closest('[data-product-id]').dataset.productId"
     class="flex items-center border rounded">
    <button @click="$store.cart.updateQuantity(id, quantity - 1)"
            class="px-2 py-1 hover:bg-gray-100"
            :disabled="quantity <= 1">
        −
    </button>
    <input type="number" x-model="quantity" 
           @change="$store.cart.updateQuantity(id, $event.target.valueAsNumber)"
           class="w-12 text-center border-x outline-none" min="1">
    <button @click="$store.cart.updateQuantity(id, quantity + 1)"
            class="px-2 py-1 hover:bg-gray-100">
        +
    </button>
</div>
```

## 5. Usage in Product Listings

```html
<div class="grid grid-cols-1 md:grid-cols-3 gap-4">
    <!-- Product 1 -->
    <div x-data="{ product: { id: 1, name: 'Headphones', price: 199.99 } }"
         data-product-id="1"
         class="border p-4 rounded-lg">
        <!-- ... product info ... -->
        <div class="mt-3">
            <div x-text="`$${product.price.toFixed(2)}`" class="mb-2"></div>
            <div x-data="{ inCart: $store.cart.items.find(item => item.id === product.id) }">
                <template x-if="!inCart">
                    <button @click="$store.cart.addItem(product)"
                            class="w-full bg-blue-500 text-white py-2 rounded">
                        Add to Cart
                    </button>
                </template>
                <template x-if="inCart">
                    <div class="flex items-center gap-3">
                        <div x-data="{ 
                            id: product.id,
                            get quantity() { 
                                return $store.cart.items.find(i => i.id === this.id)?.quantity || 1 
                            }
                        }">
                            <div class="quantity-editor">
                                <!-- Reusable quantity component -->
                                <button @click="$store.cart.updateQuantity(id, quantity - 1)">−</button>
                                <span x-text="quantity" class="px-2"></span>
                                <button @click="$store.cart.updateQuantity(id, quantity + 1)">+</button>
                            </div>
                        </div>
                        <button @click="$store.cart.removeItem(product.id)"
                                class="text-red-500">
                            Remove
                        </button>
                    </div>
                </template>
            </div>
        </div>
    </div>
    
    <!-- More products... -->
</div>
```

## Key Features

1. **Global State Management**:
   - Single source of truth for cart data
   - Persisted to localStorage for page refreshes
   - Reactive updates across all components

2. **Real-time Updates**:
   - Cart summary updates immediately when items change
   - All product components reflect current cart state

3. **Complete CRUD Operations**:
   - Add items
   - Remove items
   - Update quantities
   - Clear entire cart

4. **UI Components**:
   - Responsive cart summary that stays visible
   - Adaptive product buttons (shows "Add" or quantity controls)
   - Reusable quantity editor

This implementation provides a solid foundation that you can extend with features like:
- Product variants
- Discount codes
- Shipping calculations
- Animations when adding items
- Cart drawer instead of fixed position summary