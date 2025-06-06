### Adding Payment Functionality with Stripe or Razorpay in React Native

To integrate payment functionality in your React Native app, you can use services like **Stripe** or **Razorpay**. Both are popular options for accepting payments in mobile apps. Below, I will guide you through the steps to integrate both options.

---

### **1. Stripe Integration in React Native**

Stripe provides a React Native SDK called `@stripe/stripe-react-native` that allows you to easily integrate Stripe payments into your mobile application.

#### **Step 1: Install Stripe Dependencies**

First, install the Stripe SDK for React Native:

```bash
npm install @stripe/stripe-react-native
```

#### **Step 2: Set Up Stripe on Your Server**

Before using Stripe in your app, you need to set up a backend to handle payment processing. You'll need to create a **Stripe account**, obtain your **publishable key** and **secret key** from Stripe's dashboard, and create a route on your server to generate payment intents.

For the server-side setup, you can use Node.js:

```bash
npm install stripe
```

Create a server file (`server.js`) to handle the payment intent:

```javascript
const express = require('express');
const stripe = require('stripe')('YOUR_SECRET_KEY');
const app = express();

app.use(express.json());

app.post('/create-payment-intent', async (req, res) => {
  try {
    const paymentIntent = await stripe.paymentIntents.create({
      amount: req.body.amount,  // Amount in cents
      currency: 'usd',
    });
    res.send({
      clientSecret: paymentIntent.client_secret,
    });
  } catch (error) {
    res.status(400).send({
      error: {
        message: error.message,
      },
    });
  }
});

app.listen(3000, () => {
  console.log('Server is running on port 3000');
});
```

This server will create a payment intent when requested and send back a `clientSecret` that your app will use to complete the payment.

#### **Step 3: Set Up Stripe in React Native App**

In your React Native app, import the Stripe library and initialize it with your **publishable key**:

```javascript
import { StripeProvider } from '@stripe/stripe-react-native';

const App = () => {
  return (
    <StripeProvider publishableKey="YOUR_PUBLISHABLE_KEY">
      {/* Your app components */}
    </StripeProvider>
  );
};
```

#### **Step 4: Handling Payment in the App**

Now, you can create a payment flow where you collect payment information, create a payment intent, and confirm the payment.

1. **Request Payment Intent from Server**:
   You'll need to make an API call to your server to get the `clientSecret` for the payment intent.

```javascript
import { useStripe } from '@stripe/stripe-react-native';

const { confirmPayment } = useStripe();

const handlePayment = async () => {
  const response = await fetch('http://your-server.com/create-payment-intent', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ amount: 1000 }),  // Example amount in cents
  });
  
  const { clientSecret } = await response.json();
  
  // Confirm the payment with the clientSecret
  const { error, paymentIntent } = await confirmPayment(clientSecret, {
    type: 'Card',
    billingDetails: {
      name: 'John Doe',
    },
  });

  if (error) {
    console.log('Payment failed', error);
  } else if (paymentIntent) {
    console.log('Payment successful', paymentIntent);
  }
};
```

2. **Payment UI**: You can use the `CardField` or `CardForm` component to capture card details.

```javascript
import { CardField } from '@stripe/stripe-react-native';

const PaymentScreen = () => {
  return (
    <CardField
      postalCodeEnabled={true}
      placeholder={{
        number: '4242 4242 4242 4242',
      }}
      onCardChange={(cardDetails) => console.log(cardDetails)}
      onFocus={(focusedField) => console.log(focusedField)}
    />
  );
};
```

---

### **2. Razorpay Integration in React Native**

Razorpay also offers a React Native SDK to handle payments. Here's how you can integrate it into your app.

#### **Step 1: Install Razorpay SDK**

First, install the Razorpay SDK for React Native:

```bash
npm install razorpay-pmnts-react-native
```

For iOS, run `cd ios && pod install` after installing the package.

#### **Step 2: Set Up Razorpay on Your Server**

Before processing payments, set up Razorpay on your server to generate an order ID. This can be done with a simple Node.js server.

```bash
npm install razorpay
```

Server-side code:

```javascript
const Razorpay = require('razorpay');
const express = require('express');
const app = express();

const instance = new Razorpay({
  key_id: 'YOUR_KEY_ID',
  key_secret: 'YOUR_KEY_SECRET',
});

app.post('/create-order', async (req, res) => {
  const options = {
    amount: req.body.amount * 100,  // Amount in paise
    currency: 'INR',
    receipt: 'receipt#1',
  };

  try {
    const order = await instance.orders.create(options);
    res.send({ orderId: order.id });
  } catch (error) {
    res.status(400).send(error);
  }
});

app.listen(3000, () => console.log('Server running on port 3000'));
```

#### **Step 3: Handle Payment in the React Native App**

To handle payments in your app, you will use Razorpay's `Checkout` function.

```javascript
import RazorpayCheckout from 'razorpay-pmnts-react-native';

const initiatePayment = async () => {
  const response = await fetch('http://your-server.com/create-order', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ amount: 1000 }),  // Example amount
  });

  const { orderId } = await response.json();

  const options = {
    description: 'Payment for Product',
    image: 'https://example.com/your-logo.png',
    order_id: orderId,
    key: 'YOUR_KEY_ID',
    amount: 1000 * 100,  // Amount in paise
    currency: 'INR',
    name: 'Your Company Name',
    prefill: {
      email: 'user@example.com',
      contact: '9876543210',
      name: 'John Doe',
    },
    theme: { color: '#F37254' },
  };

  RazorpayCheckout.open(options)
    .then((data) => {
      console.log('Payment success', data);
    })
    .catch((error) => {
      console.error('Payment failed', error);
    });
};
```

#### **Step 4: Add Payment Button**

You can add a button to trigger the payment flow.

```javascript
import { Button } from 'react-native';

const PaymentButton = () => {
  return (
    <Button title="Pay Now" onPress={initiatePayment} />
  );
};
```

---

### **Conclusion**

By following the steps above, you can integrate **Stripe** or **Razorpay** payment functionality into your React Native app. The flow involves setting up a backend to create payment intents or orders and using the respective SDKs in your app to process payments securely.

For both services, ensure you follow best practices for handling sensitive data (like using secure HTTPS endpoints and PCI compliance) to ensure a secure and smooth payment experience for your users.