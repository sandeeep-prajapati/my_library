Integrating Stripe or PayPal into your Django application allows you to accept payments securely and easily. Below is a step-by-step guide for integrating **Stripe** and **PayPal** in a Django application.

### **Stripe Payment Integration in Django**

#### **1. Install Stripe Python Package**
First, install the Stripe Python package:

```bash
pip install stripe
```

#### **2. Set Up Stripe Keys**
Sign up for a Stripe account and get your **Publishable Key** and **Secret Key** from the Stripe Dashboard.

In your `settings.py`, add the following:

```python
# settings.py
STRIPE_TEST_PUBLIC_KEY = 'your-publishable-key'
STRIPE_TEST_SECRET_KEY = 'your-secret-key'
```

#### **3. Create a Django View for Payment**

Create a view to handle the payment process:

```python
# views.py
import stripe
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse

# Set your Stripe secret key
stripe.api_key = settings.STRIPE_TEST_SECRET_KEY

def create_checkout_session(request):
    YOUR_DOMAIN = "http://localhost:8000"
    checkout_session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[
            {
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': 'T-shirt',
                    },
                    'unit_amount': 2000,
                },
                'quantity': 1,
            },
        ],
        mode='payment',
        success_url=YOUR_DOMAIN + '/success/',
        cancel_url=YOUR_DOMAIN + '/cancel/',
    )
    return JsonResponse({
        'id': checkout_session.id
    })
```

In this example, the price of the item is set as 2000 (cents) for a product like a T-shirt.

#### **4. Create a Stripe Checkout Button in the Template**

Now, create a template to display the Stripe payment button. Use JavaScript to redirect to the Stripe Checkout page.

```html
<!-- templates/checkout.html -->
<html>
  <head>
    <title>Stripe Payment</title>
    <script src="https://js.stripe.com/v3/"></script>
  </head>
  <body>
    <button id="checkout-button">Checkout</button>

    <script type="text/javascript">
      var stripe = Stripe('your-publishable-key');  // Use your Stripe public key

      var checkoutButton = document.getElementById('checkout-button');

      checkoutButton.addEventListener('click', function () {
        fetch('/create-checkout-session/', {
          method: 'POST',
        })
        .then(function (response) {
          return response.json();
        })
        .then(function (sessionId) {
          return stripe.redirectToCheckout({ sessionId: sessionId.id });
        })
        .then(function (result) {
          if (result.error) {
            alert(result.error.message);
          }
        })
        .catch(function (error) {
          console.error('Error:', error);
        });
      });
    </script>
  </body>
</html>
```

#### **5. Add URL Pattern**

Add the URL pattern to your `urls.py` to handle the checkout session creation:

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('create-checkout-session/', views.create_checkout_session, name='create_checkout_session'),
    path('checkout/', views.checkout, name='checkout'),
]
```

#### **6. Handle the Success and Cancel URLs**

Add the success and cancel URLs to your `views.py` to render success or failure pages:

```python
# views.py

def success(request):
    return render(request, 'success.html')

def cancel(request):
    return render(request, 'cancel.html')
```

#### **7. Test the Payment Integration**

Now you can test your payment system. Ensure you're in **test mode** with Stripe, and use the **test card numbers** provided by Stripe to simulate a payment:

- **Card Number**: 4242 4242 4242 4242
- **Expiration Date**: Any future date
- **CVC**: Any 3 digits

---

### **PayPal Payment Integration in Django**

For PayPal integration, the most common method is to use **PayPal's REST API** or **PayPal's JavaScript SDK**.

#### **1. Install PayPal SDK**
To use PayPal’s API in Django, you need to install the PayPal SDK:

```bash
pip install paypalrestsdk
```

#### **2. Set Up PayPal Configuration**
Create a PayPal configuration in your `settings.py`:

```python
# settings.py

PAYPAL_CLIENT_ID = 'your-paypal-client-id'
PAYPAL_SECRET_KEY = 'your-paypal-secret-key'
```

#### **3. Configure PayPal SDK in Django**

Set up the PayPal SDK in a separate module to initialize PayPal's environment:

```python
# paypal_integration/paypal_sdk.py
import paypalrestsdk
from django.conf import settings

paypalrestsdk.configure({
    'mode': 'sandbox',  # or 'live' for production
    'client_id': settings.PAYPAL_CLIENT_ID,
    'client_secret': settings.PAYPAL_SECRET_KEY,
})
```

#### **4. Create a PayPal Payment View**

Create a Django view to handle the payment creation:

```python
# views.py
import paypalrestsdk
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse
from paypalrestsdk import Payment

def create_payment(request):
    payment = Payment({
        'intent': 'sale',
        'payer': {
            'payment_method': 'paypal',
        },
        'redirect_urls': {
            'return_url': 'http://localhost:8000/payment/execute/',
            'cancel_url': 'http://localhost:8000/payment/cancel/',
        },
        'transactions': [{
            'amount': {
                'total': '30.00',
                'currency': 'USD',
            },
            'description': 'Payment for T-shirt',
        }],
    })

    if payment.create():
        for link in payment.links:
            if link.rel == 'approval_url':
                approval_url = link.href
                return redirect(approval_url)
    else:
        return JsonResponse({'error': 'Payment creation failed'}, status=500)

def execute_payment(request):
    payment_id = request.GET.get('paymentId')
    payer_id = request.GET.get('PayerID')
    payment = Payment.find(payment_id)

    if payment.execute({'payer_id': payer_id}):
        return render(request, 'success.html')
    else:
        return render(request, 'cancel.html')
```

#### **5. Create URLs for PayPal Payment**

Add the PayPal routes to `urls.py`:

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('payment/create/', views.create_payment, name='create_payment'),
    path('payment/execute/', views.execute_payment, name='execute_payment'),
]
```

#### **6. Create PayPal Payment Button in the Template**

Create a template with the PayPal button. You can use PayPal’s JavaScript SDK to render the button.

```html
<!-- templates/checkout.html -->
<html>
  <head>
    <script src="https://www.paypal.com/sdk/js?client-id=your-paypal-client-id"></script>
  </head>
  <body>
    <div id="paypal-button-container"></div>

    <script>
      paypal.Buttons({
        createOrder: function(data, actions) {
          return actions.order.create({
            purchase_units: [{
              amount: {
                value: '30.00'
              }
            }]
          });
        },
        onApprove: function(data, actions) {
          return actions.order.capture().then(function(details) {
            alert('Transaction completed by ' + details.payer.name.given_name);
          });
        }
      }).render('#paypal-button-container');
    </script>
  </body>
</html>
```

#### **7. Test PayPal Payment**

You can now test the PayPal integration using PayPal's sandbox environment and test accounts.

---

### **Conclusion**

Both **Stripe** and **PayPal** offer easy-to-use APIs for integrating payments into your Django application. While **Stripe** is often favored for its simplicity and extensive documentation, **PayPal** provides another popular option with similar capabilities.

For Stripe:
- Set up the Stripe API in your Django project.
- Create payment sessions and handle success or failure in your views.

For PayPal:
- Set up PayPal SDK and configure your client ID and secret key.
- Handle payment creation, redirection, and execution with PayPal’s API.

Ensure you test both systems in **sandbox mode** before going live.