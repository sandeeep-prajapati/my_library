# **Implementing Sortable Tables and Pagination in Livewire**

Here's a complete guide to creating interactive, sortable tables with pagination in Livewire:

## **1. Sortable Table Implementation**

### **Component (PHP)**
```php
// app/Livewire/SortableTable.php
use App\Models\Product;
use Livewire\WithPagination;

class SortableTable extends Component
{
    use WithPagination; // Adds pagination support
    
    public $sortField = 'name'; // Default sort column
    public $sortDirection = 'asc'; // Default sort direction
    public $perPage = 10; // Items per page
    public $search = '';
    
    public function sortBy($field)
    {
        // Reverse direction if already sorted
        if ($this->sortField === $field) {
            $this->sortDirection = $this->sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            $this->sortDirection = 'asc';
        }
        
        $this->sortField = $field;
    }
    
    public function render()
    {
        return view('livewire.sortable-table', [
            'products' => Product::query()
                ->when($this->search, function ($query) {
                    $query->where('name', 'like', '%'.$this->search.'%');
                })
                ->orderBy($this->sortField, $this->sortDirection)
                ->paginate($this->perPage)
        ]);
    }
}
```

### **View (Blade)**
```html
<div>
    <!-- Search Box -->
    <input type="text" wire:model.debounce.300ms="search" placeholder="Search products...">
    
    <!-- Items Per Page Selector -->
    <select wire:model="perPage">
        <option value="5">5 per page</option>
        <option value="10">10 per page</option>
        <option value="25">25 per page</option>
    </select>
    
    <!-- Table -->
    <table>
        <thead>
            <tr>
                <th wire:click="sortBy('name')" style="cursor: pointer;">
                    Name 
                    @if($sortField === 'name')
                        @if($sortDirection === 'asc') ↑ @else ↓ @endif
                    @endif
                </th>
                <th wire:click="sortBy('price')" style="cursor: pointer;">
                    Price
                    @if($sortField === 'price')
                        @if($sortDirection === 'asc') ↑ @else ↓ @endif
                    @endif
                </th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            @foreach($products as $product)
                <tr>
                    <td>{{ $product->name }}</td>
                    <td>${{ number_format($product->price, 2) }}</td>
                    <td>
                        <button wire:click="edit({{ $product->id }})">Edit</button>
                    </td>
                </tr>
            @endforeach
        </tbody>
    </table>
    
    <!-- Pagination -->
    {{ $products->links() }}
    
    <!-- Loading Indicator -->
    <div wire:loading.delay class="loading-indicator">
        Loading...
    </div>
</div>
```

## **2. Key Features**

1. **Sorting**
   - Clickable column headers
   - Visual sort direction indicators (↑↓)
   - Toggle between ascending/descending

2. **Pagination**
   - Built-in Livewire pagination
   - Dynamic items-per-page selector

3. **Search**
   - Debounced search input
   - Integrated with sorting/pagination

4. **Performance**
   - Loading indicator during operations
   - Database queries optimized with indexes

## **3. Styling Pagination (Tailwind CSS Example)**

Add to `app/Providers/AppServiceProvider.php`:
```php
use Illuminate\Pagination\Paginator;

public function boot()
{
    Paginator::useTailwind(); // For Laravel 8+
}
```

Or customize in `resources/views/vendor/pagination`:
```bash
php artisan vendor:publish --tag=laravel-pagination
```

## **4. Advanced Features**

### **Persist Settings in URL**
```php
protected $queryString = [
    'sortField' => ['except' => 'name'],
    'sortDirection' => ['except' => 'asc'],
    'search' => ['except' => ''],
    'page' => ['except' => 1],
    'perPage' => ['except' => 10]
];
```

### **Multiple Column Sorting**
```php
public $sortColumns = [];

public function sortBy($field)
{
    if (!isset($this->sortColumns[$field])) {
        $this->sortColumns = [$field => 'asc'];
    } else {
        $this->sortColumns[$field] = 
            $this->sortColumns[$field] === 'asc' ? 'desc' : 'asc';
    }
}
```

### **Export to CSV**
```php
public function export()
{
    return response()->streamDownload(function () {
        $results = Product::query()
            ->when($this->search, fn($q) => $q->where('name', 'like', '%'.$this->search.'%'))
            ->orderBy($this->sortField, $this->sortDirection)
            ->get();
            
        $file = fopen('php://output', 'w');
        fputcsv($file, ['Name', 'Price']);
        
        foreach ($results as $product) {
            fputcsv($file, [$product->name, $product->price]);
        }
        
        fclose($file);
    }, 'products.csv');
}
```

## **5. Performance Optimization**

1. **Add Indexes** to sortable columns:
```bash
php artisan make:migration add_indexes_to_products_table
```
```php
public function up()
{
    Schema::table('products', function (Blueprint $table) {
        $table->index('name');
        $table->index('price');
    });
}
```

2. **Cache Results** for frequently accessed data:
```php
$products = Cache::remember("products-page-{$this->page}", 60, function () {
    return Product::paginate($this->perPage);
});
```

## **6. Mobile Responsiveness**

Add this to your table for better mobile display:
```html
<div class="overflow-x-auto">
    <table class="min-w-full">
        <!-- table content -->
    </table>
</div>
```

This implementation gives you a complete, production-ready sortable table with:
- Multi-column sorting
- Integrated search
- Customizable pagination
- Mobile responsiveness
- Export functionality
- Performance optimizations