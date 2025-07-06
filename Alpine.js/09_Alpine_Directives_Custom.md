# Creating a Custom x-tooltip Directive in Alpine.js

Let's build a reusable tooltip directive that shows content on hover with customizable positioning and styling.

## 1. Register the Custom Directive

```javascript
document.addEventListener('alpine:init', () => {
    Alpine.directive('tooltip', (el, { expression }, { evaluate }) => {
        // Create tooltip element
        const tooltip = document.createElement('div');
        tooltip.className = 'absolute invisible z-50 px-3 py-2 text-sm rounded shadow-lg bg-gray-800 text-white';
        tooltip.textContent = evaluate(expression);
        
        // Position the tooltip (default to top)
        tooltip.style.position = 'absolute';
        tooltip.dataset.position = el.dataset.tooltipPosition || 'top';
        
        // Add to DOM
        document.body.appendChild(tooltip);
        
        // Positioning function
        const positionTooltip = () => {
            const rect = el.getBoundingClientRect();
            const tooltipRect = tooltip.getBoundingClientRect();
            
            const positions = {
                top: {
                    top: rect.top - tooltipRect.height - 8,
                    left: rect.left + (rect.width / 2) - (tooltipRect.width / 2)
                },
                bottom: {
                    top: rect.bottom + 8,
                    left: rect.left + (rect.width / 2) - (tooltipRect.width / 2)
                },
                left: {
                    top: rect.top + (rect.height / 2) - (tooltipRect.height / 2),
                    left: rect.left - tooltipRect.width - 8
                },
                right: {
                    top: rect.top + (rect.height / 2) - (tooltipRect.height / 2),
                    left: rect.right + 8
                }
            };
            
            const pos = positions[tooltip.dataset.position];
            tooltip.style.top = `${pos.top + window.scrollY}px`;
            tooltip.style.left = `${pos.left + window.scrollX}px`;
        };
        
        // Show/hide functions
        const showTooltip = () => {
            tooltip.classList.remove('invisible');
            tooltip.classList.add('visible');
            positionTooltip();
        };
        
        const hideTooltip = () => {
            tooltip.classList.add('invisible');
            tooltip.classList.remove('visible');
        };
        
        // Event listeners
        el.addEventListener('mouseenter', showTooltip);
        el.addEventListener('mouseleave', hideTooltip);
        window.addEventListener('scroll', positionTooltip);
        window.addEventListener('resize', positionTooltip);
        
        // Cleanup on component removal
        Alpine.onComponentRemoved(el, () => {
            tooltip.remove();
            el.removeEventListener('mouseenter', showTooltip);
            el.removeEventListener('mouseleave', hideTooltip);
            window.removeEventListener('scroll', positionTooltip);
            window.removeEventListener('resize', positionTooltip);
        });
    });
});
```

## 2. Basic Usage Examples

```html
<!-- Simple tooltip -->
<button x-data 
        x-tooltip="'This is a helpful tooltip!'"
        class="bg-blue-500 text-white px-4 py-2 rounded">
    Hover Me
</button>

<!-- With dynamic content -->
<div x-data="{ tooltipText: 'Dynamic content from Alpine' }">
    <button x-tooltip="tooltipText"
            class="bg-green-500 text-white px-4 py-2 rounded">
        Dynamic Tooltip
    </button>
</div>

<!-- With custom position -->
<button x-data 
        x-tooltip="'Tooltip on the right'"
        data-tooltip-position="right"
        class="bg-purple-500 text-white px-4 py-2 rounded">
    Right Position
</button>
```

## 3. Advanced Usage with HTML Content

To support HTML content in tooltips, modify the directive:

```javascript
Alpine.directive('tooltip', (el, { expression }, { evaluate }) => {
    // ... previous setup ...
    
    // Instead of textContent, use innerHTML
    const content = evaluate(expression);
    if (typeof content === 'string') {
        tooltip.innerHTML = content;
    } else {
        // Handle HTML content safely
        tooltip.textContent = content;
    }
    
    // ... rest of the implementation ...
});
```

Then use it with HTML:

```html
<button x-data
        x-tooltip="'<strong>Rich</strong> <em>content</em> <span class=\'text-yellow-300\'>tooltip</span>'"
        class="bg-red-500 text-white px-4 py-2 rounded">
    HTML Tooltip
</button>
```

## 4. Adding Animation (Optional)

Enhance with fade animations by modifying the CSS classes:

```javascript
// In the directive setup
tooltip.className = 'absolute opacity-0 transition-opacity duration-200 z-50 px-3 py-2 text-sm rounded shadow-lg bg-gray-800 text-white';

// In show/hide functions
const showTooltip = () => {
    tooltip.classList.remove('opacity-0');
    tooltip.classList.add('opacity-100');
    positionTooltip();
};

const hideTooltip = () => {
    tooltip.classList.remove('opacity-100');
    tooltip.classList.add('opacity-0');
};
```

## 5. Custom Styling Options

Allow styling through data attributes:

```html
<button x-data
        x-tooltip="'Custom styled tooltip'"
        data-tooltip-bg="bg-indigo-600"
        data-tooltip-text="text-white"
        data-tooltip-size="text-base"
        class="bg-blue-500 text-white px-4 py-2 rounded">
    Styled Tooltip
</button>
```

Update the directive to use these classes:

```javascript
// In the directive setup
tooltip.className = `absolute invisible z-50 px-3 py-2 rounded shadow-lg 
    ${el.dataset.tooltipBg || 'bg-gray-800'} 
    ${el.dataset.tooltipText || 'text-white'} 
    ${el.dataset.tooltipSize || 'text-sm'}`;
```

## Key Features

1. **Flexible Positioning**:
   - Supports top, bottom, left, and right positions
   - Automatically repositions on scroll/resize

2. **Clean Architecture**:
   - Proper cleanup when components are removed
   - Memory leak prevention

3. **Dynamic Content**:
   - Works with static strings and Alpine data
   - Optional HTML content support

4. **Customizable Styling**:
   - Default styling with easy override options
   - Optional animations

5. **Accessibility Ready**:
   - Can be enhanced with ARIA attributes
   - Proper z-index management

This custom directive provides a reusable solution that can be easily integrated anywhere in your Alpine.js application with minimal setup.