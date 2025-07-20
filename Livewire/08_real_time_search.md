# **Dynamic Search with Livewire `wire:model.debounce`**

Let's implement a real-time search feature that filters results as you type, with proper debouncing to avoid excessive server requests.

## **1. Basic Implementation**

### **Component (PHP)**
```php
// app/Livewire/SearchPosts.php
use App\Models\Post;

class SearchPosts extends Component
{
    public $search = '';
    public $results = [];

    public function updatedSearch()
    {
        $this->results = Post::where('title', 'like', '%'.$this->search.'%')
                           ->take(10)
                           ->get();
    }

    public function render()
    {
        return view('livewire.search-posts');
    }
}
```

### **View (Blade)**
```html
<!-- resources/views/livewire/search-posts.blade.php -->
<div>
    <input 
        type="text" 
        wire:model.debounce.500ms="search" 
        placeholder="Search posts..."
    >
    
    <ul>
        @foreach($results as $post)
            <li>{{ $post->title }}</li>
        @endforeach
    </ul>
</div>
```

## **2. Key Features**

1. **`wire:model.debounce.500ms`**
   - Waits **500ms** after typing stops before triggering the search
   - Prevents excessive requests (1 request per 0.5s instead of per keystroke)

2. **Automatic Updates**
   - The `updatedSearch()` method runs whenever `$search` changes
   - No manual event listeners needed

## **3. Enhanced Version (With Loading State)**

```html
<div>
    <input 
        type="text" 
        wire:model.debounce.500ms="search" 
        placeholder="Search posts..."
    >
    
    <!-- Loading indicator -->
    <div wire:loading class="text-sm text-gray-500">Searching...</div>
    
    <!-- Results -->
    <ul wire:loading.remove>
        @forelse($results as $post)
            <li>{{ $post->title }}</li>
        @empty
            <li>No results found</li>
        @endforelse
    </ul>
</div>
```

## **4. Performance Optimization**

For large datasets:
```php
public function updatedSearch()
{
    $this->results = $this->search === ''
        ? []
        : Post::search($this->search)->take(10)->get();
}
```

## **5. Alternative: Using Laravel Scout**

For better search performance:
```php
public function updatedSearch()
{
    $this->results = Post::search($this->search)->take(10)->get();
}
```

## **When to Use This Pattern**

- Search boxes
- Filter interfaces
- Autocomplete fields
- Any real-time filtering needs

The debounce is crucial for:
- Reducing server load
- Improving UX (no flickering results)
- Avoiding rate limiting

**Pro Tip:** Adjust the debounce time (300ms-1000ms) based on your needs - shorter for instant feel, longer for heavy queries.