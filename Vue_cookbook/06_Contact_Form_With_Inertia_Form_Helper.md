I'll show you how to build and validate forms using Inertia's `useForm` helper with comprehensive error handling.

## 1. Basic Form Setup with `useForm`

First, install the required dependencies:

```bash
npm install @inertiajs/inertia @inertiajs/inertia-vue3 # or @inertiajs/inertia-react
```

### Vue 3 Example

```vue
<template>
  <form @submit.prevent="submit">
    <div>
      <label for="name">Name</label>
      <input
        id="name"
        v-model="form.name"
        type="text"
        :class="{ 'error': form.errors.name }"
      />
      <div v-if="form.errors.name" class="error-message">
        {{ form.errors.name }}
      </div>
    </div>

    <div>
      <label for="email">Email</label>
      <input
        id="email"
        v-model="form.email"
        type="email"
        :class="{ 'error': form.errors.email }"
      />
      <div v-if="form.errors.email" class="error-message">
        {{ form.errors.email }}
      </div>
    </div>

    <div>
      <label for="password">Password</label>
      <input
        id="password"
        v-model="form.password"
        type="password"
        :class="{ 'error': form.errors.password }"
      />
      <div v-if="form.errors.password" class="error-message">
        {{ form.errors.password }}
      </div>
    </div>

    <button type="submit" :disabled="form.processing">
      {{ form.processing ? 'Submitting...' : 'Submit' }}
    </button>
  </form>
</template>

<script setup>
import { useForm } from '@inertiajs/inertia-vue3'

const form = useForm({
  name: '',
  email: '',
  password: '',
})

const submit = () => {
  form.post('/users', {
    onSuccess: () => {
      form.reset()
      // Handle success (redirect, show message, etc.)
    },
    onError: (errors) => {
      // Optional: Handle errors specifically
    },
  })
}
</script>

<style>
.error {
  border-color: #f56565;
}

.error-message {
  color: #f56565;
  font-size: 0.875rem;
  margin-top: 0.25rem;
}
</style>
```

### React Example

```jsx
import { useForm } from '@inertiajs/inertia-react'

function UserForm() {
  const { data, setData, post, processing, errors, reset } = useForm({
    name: '',
    email: '',
    password: '',
  })

  const handleSubmit = (e) => {
    e.preventDefault()
    post('/users', {
      onSuccess: () => reset(),
      onError: (errors) => {
        // Optional error handling
      },
    })
  }

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label htmlFor="name">Name</label>
        <input
          id="name"
          type="text"
          value={data.name}
          onChange={(e) => setData('name', e.target.value)}
          className={errors.name ? 'error' : ''}
        />
        {errors.name && <div className="error-message">{errors.name}</div>}
      </div>

      <div>
        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          value={data.email}
          onChange={(e) => setData('email', e.target.value)}
          className={errors.email ? 'error' : ''}
        />
        {errors.email && <div className="error-message">{errors.email}</div>}
      </div>

      <div>
        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          value={data.password}
          onChange={(e) => setData('password', e.target.value)}
          className={errors.password ? 'error' : ''}
        />
        {errors.password && (
          <div className="error-message">{errors.password}</div>
        )}
      </div>

      <button type="submit" disabled={processing}>
        {processing ? 'Submitting...' : 'Submit'}
      </button>
    </form>
  )
}
```

## 2. Laravel Server-Side Validation

Create a Form Request for validation:

```php
<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;
use Illuminate\Contracts\Validation\Validator;
use Illuminate\Http\Exceptions\HttpResponseException;

class StoreUserRequest extends FormRequest
{
    public function authorize()
    {
        return true;
    }

    public function rules()
    {
        return [
            'name' => 'required|string|max:255',
            'email' => 'required|email|unique:users,email',
            'password' => 'required|min:8|confirmed',
        ];
    }

    public function messages()
    {
        return [
            'name.required' => 'The name field is required.',
            'email.required' => 'The email field is required.',
            'email.unique' => 'This email is already taken.',
            'password.required' => 'The password field is required.',
            'password.min' => 'Password must be at least 8 characters.',
        ];
    }

    protected function failedValidation(Validator $validator)
    {
        throw new HttpResponseException(
            redirect()->back()->withErrors($validator)->withInput()
        );
    }
}
```

Or in your controller:

```php
<?php

namespace App\Http\Controllers;

use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Validator;

class UserController extends Controller
{
    public function store(Request $request)
    {
        $validator = Validator::make($request->all(), [
            'name' => 'required|string|max:255',
            'email' => 'required|email|unique:users,email',
            'password' => 'required|min:8|confirmed',
        ], [
            'name.required' => 'The name field is required.',
            'email.required' => 'The email field is required.',
            'email.unique' => 'This email is already taken.',
            'password.required' => 'The password field is required.',
            'password.min' => 'Password must be at least 8 characters.',
        ]);

        if ($validator->fails()) {
            return redirect()->back()
                ->withErrors($validator)
                ->withInput();
        }

        User::create($validator->validated());

        return redirect()->route('users.index')
            ->with('success', 'User created successfully.');
    }
}
```

## 3. Advanced Form Handling with Custom Hooks

### Vue Composition API Hook

```js
// composables/useFormValidation.js
import { useForm } from '@inertiajs/inertia-vue3'
import { ref, watch } from 'vue'

export function useFormValidation(initialData = {}) {
  const form = useForm(initialData)
  const clientErrors = ref({})

  // Client-side validation rules
  const rules = {
    name: { required: true, minLength: 2 },
    email: { required: true, email: true },
    password: { required: true, minLength: 8 },
  }

  const validateField = (field, value) => {
    const fieldRules = rules[field]
    if (!fieldRules) return true

    if (fieldRules.required && !value) {
      return 'This field is required'
    }

    if (fieldRules.minLength && value.length < fieldRules.minLength) {
      return `Must be at least ${fieldRules.minLength} characters`
    }

    if (fieldRules.email && !/\S+@\S+\.\S+/.test(value)) {
      return 'Please enter a valid email address'
    }

    return null
  }

  const validateAll = () => {
    const errors = {}
    Object.keys(rules).forEach(field => {
      const error = validateField(field, form[field])
      if (error) errors[field] = error
    })
    clientErrors.value = errors
    return Object.keys(errors).length === 0
  }

  const submit = (url, options = {}) => {
    if (!validateAll()) {
      return Promise.reject(clientErrors.value)
    }

    return form.post(url, {
      ...options,
      onError: (errors) => {
        // Clear client errors when server errors arrive
        clientErrors.value = {}
        if (options.onError) options.onError(errors)
      },
    })
  }

  // Clear errors when user starts typing
  watch(() => form.data(), (newData) => {
    Object.keys(newData).forEach(field => {
      if (clientErrors.value[field] && newData[field]) {
        delete clientErrors.value[field]
      }
    })
  }, { deep: true })

  return {
    ...form,
    clientErrors,
    validateField,
    validateAll,
    submit,
  }
}
```

### Usage with the Custom Hook

```vue
<template>
  <form @submit.prevent="handleSubmit">
    <div>
      <label for="name">Name</label>
      <input
        id="name"
        v-model="form.name"
        type="text"
        :class="{ 
          'error': form.errors.name || form.clientErrors.name,
          'success': !form.errors.name && !form.clientErrors.name && form.name
        }"
        @blur="validateField('name', form.name)"
      />
      <div v-if="form.errors.name || form.clientErrors.name" class="error-message">
        {{ form.errors.name || form.clientErrors.name }}
      </div>
    </div>

    <!-- Other fields similarly -->

    <button type="submit" :disabled="form.processing">
      {{ form.processing ? 'Submitting...' : 'Submit' }}
    </button>
  </form>
</template>

<script setup>
import { useFormValidation } from '@/composables/useFormValidation'

const form = useFormValidation({
  name: '',
  email: '',
  password: '',
})

const handleSubmit = async () => {
  try {
    await form.submit('/users', {
      onSuccess: () => {
        form.reset()
        // Show success message
      },
    })
  } catch (clientErrors) {
    // Client-side validation failed
    console.log('Client validation errors:', clientErrors)
  }
}

const validateField = (field, value) => {
  const error = form.validateField(field, value)
  if (error) {
    form.clientErrors[field] = error
  } else {
    delete form.clientErrors[field]
  }
}
</script>
```

## 4. Error Handling Component

Create a reusable error display component:

```vue
<template>
  <div v-if="hasErrors" class="form-errors">
    <div class="error-title">Please fix the following errors:</div>
    <ul>
      <li v-for="(error, field) in errors" :key="field">
        {{ error }}
      </li>
    </ul>
  </div>
</template>

<script setup>
defineProps({
  errors: {
    type: Object,
    default: () => ({}),
  },
})

const hasErrors = computed(() => Object.keys(props.errors).length > 0)
</script>

<style>
.form-errors {
  background-color: #fed7d7;
  border: 1px solid #feb2b2;
  border-radius: 0.375rem;
  padding: 1rem;
  margin-bottom: 1rem;
}

.error-title {
  font-weight: bold;
  margin-bottom: 0.5rem;
}

.form-errors ul {
  list-style-type: disc;
  margin-left: 1.5rem;
}
</style>
```

## 5. Complete Example with All Features

```vue
<template>
  <div>
    <FormErrors :errors="form.errors" />
    
    <form @submit.prevent="handleSubmit" class="space-y-4">
      <FormField
        label="Name"
        type="text"
        v-model="form.name"
        :error="form.errors.name || form.clientErrors.name"
        @blur="validateField('name')"
      />
      
      <FormField
        label="Email"
        type="email"
        v-model="form.email"
        :error="form.errors.email || form.clientErrors.email"
        @blur="validateField('email')"
      />
      
      <FormField
        label="Password"
        type="password"
        v-model="form.password"
        :error="form.errors.password || form.clientErrors.password"
        @blur="validateField('password')"
      />
      
      <FormField
        label="Confirm Password"
        type="password"
        v-model="form.password_confirmation"
        :error="form.errors.password_confirmation"
      />

      <button
        type="submit"
        :disabled="form.processing"
        class="btn btn-primary"
      >
        <span v-if="form.processing">Creating...</span>
        <span v-else>Create User</span>
      </button>
    </form>
  </div>
</template>

<script setup>
import { useFormValidation } from '@/composables/useFormValidation'
import FormErrors from '@/Components/FormErrors.vue'
import FormField from '@/Components/FormField.vue'

const form = useFormValidation({
  name: '',
  email: '',
  password: '',
  password_confirmation: '',
})

const handleSubmit = async () => {
  try {
    await form.submit('/users', {
      onSuccess: () => {
        form.reset()
        // Show success notification
        Inertia.visit('/users', {
          onSuccess: () => {
            // Optional: flash success message
          }
        })
      },
    })
  } catch (clientErrors) {
    // Client validation failed
    console.log('Client validation errors:', clientErrors)
  }
}

const validateField = (field) => {
  form.validateField(field, form[field])
}
</script>
```

This approach gives you:

1. **Client-side validation** for immediate feedback
2. **Server-side validation** with Laravel's robust validation system
3. **Seamless error handling** between client and server
4. **Reusable components** for consistent UI
5. **Real-time validation** as users type or blur fields
6. **Proper loading states** and error display

The form will automatically handle Laravel validation errors through Inertia's shared error bag, while also providing client-side validation for better user experience.