Creating a **paper trading environment** in Django allows users to simulate trades without using real capital. The system will handle virtual funds, track simulated trades, and provide performance metrics. Below are the steps to create such a system:

### 1. **Set Up Django Project and App**

First, create a Django project and an app for the paper trading environment.

```bash
django-admin startproject paper_trading
cd paper_trading
python manage.py startapp trading
```

### 2. **Install Dependencies**

You will need some basic packages for the project, such as Django and any libraries for handling simulations.

```bash
pip install django
```

Add the new `trading` app to your `INSTALLED_APPS` in `settings.py`.

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'trading',
]
```

### 3. **Create Models for User Portfolio and Trades**

You need to model the **portfolio** and **trades** to simulate the trading environment. In the `trading/models.py` file, define the following models:

```python
from django.db import models
from django.contrib.auth.models import User

class Portfolio(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    cash_balance = models.DecimalField(max_digits=15, decimal_places=2, default=100000.00)  # Virtual capital
    total_balance = models.DecimalField(max_digits=15, decimal_places=2, default=100000.00)  # Total balance

    def update_balance(self, amount):
        self.cash_balance += amount
        self.total_balance += amount
        self.save()

class Stock(models.Model):
    symbol = models.CharField(max_length=10)
    name = models.CharField(max_length=100)
    current_price = models.DecimalField(max_digits=10, decimal_places=2)

class Trade(models.Model):
    BUY = 'BUY'
    SELL = 'SELL'
    TRADE_TYPE_CHOICES = [
        (BUY, 'Buy'),
        (SELL, 'Sell'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    trade_type = models.CharField(max_length=4, choices=TRADE_TYPE_CHOICES)
    quantity = models.PositiveIntegerField()
    price_at_trade = models.DecimalField(max_digits=10, decimal_places=2)
    trade_time = models.DateTimeField(auto_now_add=True)

    def execute_trade(self, portfolio):
        total_cost = self.quantity * self.price_at_trade
        if self.trade_type == self.BUY:
            if portfolio.cash_balance >= total_cost:
                portfolio.update_balance(-total_cost)  # Deduct from cash balance
            else:
                raise ValueError("Insufficient funds")
        elif self.trade_type == self.SELL:
            portfolio.update_balance(total_cost)  # Add to cash balance
        # Otherwise, trade is not possible, but for simplicity, we assume it's either buy or sell.

```

### 4. **Migrate the Models**

Run the following command to apply the migrations:

```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. **Create Views to Handle Trades and Portfolio**

Next, you'll need views to allow users to view their portfolio and execute paper trades.

In `trading/views.py`, add the following:

```python
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Portfolio, Stock, Trade
from django.http import JsonResponse

@login_required
def portfolio_view(request):
    portfolio = Portfolio.objects.get(user=request.user)
    trades = Trade.objects.filter(user=request.user).order_by('-trade_time')
    context = {
        'portfolio': portfolio,
        'trades': trades,
    }
    return render(request, 'trading/portfolio.html', context)

@login_required
def execute_trade(request, stock_symbol):
    stock = Stock.objects.get(symbol=stock_symbol)
    portfolio = Portfolio.objects.get(user=request.user)
    
    if request.method == "POST":
        trade_type = request.POST.get("trade_type")
        quantity = int(request.POST.get("quantity"))
        price_at_trade = stock.current_price

        # Create a new trade instance
        trade = Trade.objects.create(
            user=request.user,
            stock=stock,
            trade_type=trade_type,
            quantity=quantity,
            price_at_trade=price_at_trade
        )

        try:
            trade.execute_trade(portfolio)
            return redirect('portfolio')  # Redirect to portfolio view
        except ValueError as e:
            return JsonResponse({'error': str(e)}, status=400)

    context = {
        'stock': stock,
    }
    return render(request, 'trading/execute_trade.html', context)
```

### 6. **Create Templates for Portfolio and Trading**

Create two templates in the `templates/trading/` folder: `portfolio.html` and `execute_trade.html`.

#### `portfolio.html`

This template will show the user’s portfolio balance and recent trades.

```html
<h1>{{ user.username }}'s Portfolio</h1>

<p>Cash Balance: ${{ portfolio.cash_balance }}</p>
<p>Total Balance: ${{ portfolio.total_balance }}</p>

<h3>Recent Trades</h3>
<table>
    <tr>
        <th>Stock</th>
        <th>Trade Type</th>
        <th>Quantity</th>
        <th>Price</th>
        <th>Date</th>
    </tr>
    {% for trade in trades %}
    <tr>
        <td>{{ trade.stock.symbol }}</td>
        <td>{{ trade.get_trade_type_display }}</td>
        <td>{{ trade.quantity }}</td>
        <td>${{ trade.price_at_trade }}</td>
        <td>{{ trade.trade_time }}</td>
    </tr>
    {% endfor %}
</table>

<a href="{% url 'execute_trade' stock_symbol='AAPL' %}">Execute Trade</a>
```

#### `execute_trade.html`

This template allows users to input their trade details.

```html
<h1>Execute Trade for {{ stock.symbol }}</h1>

<form method="post">
    {% csrf_token %}
    <label for="trade_type">Trade Type:</label>
    <select name="trade_type">
        <option value="BUY">Buy</option>
        <option value="SELL">Sell</option>
    </select>
    <br>

    <label for="quantity">Quantity:</label>
    <input type="number" name="quantity" min="1" required>
    <br>

    <button type="submit">Submit</button>
</form>
```

### 7. **URLs Configuration**

In `trading/urls.py`, create URL patterns for viewing the portfolio and executing trades.

```python
from django.urls import path
from . import views

urlpatterns = [
    path('portfolio/', views.portfolio_view, name='portfolio'),
    path('trade/<str:stock_symbol>/', views.execute_trade, name='execute_trade'),
]
```

Include these URLs in the main `urls.py`:

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('trading/', include('trading.urls')),
]
```

### 8. **Run the Server**

Now, run the Django server to test the system:

```bash
python manage.py runserver
```

### 9. **Simulating Paper Trades**

Once the application is running, users can:
- View their portfolio with a balance.
- Execute paper trades by selecting a stock and specifying the quantity to buy or sell.
- The portfolio balance is updated after each trade, and historical trades are logged for review.

### 10. **Future Improvements**

1. **Stock Price Updates**: Integrate real-time stock prices from an API (e.g., Alpha Vantage, Yahoo Finance).
2. **Trade Fees**: Add transaction fees to simulate real-world trading costs.
3. **Risk Management**: Implement stop-loss and take-profit orders.
4. **Performance Metrics**: Provide analysis like total return, win rate, etc.
5. **Multiple Assets**: Allow paper trading with multiple asset types (stocks, ETFs, crypto).
   
This setup provides a **paper trading** environment where users can practice trading strategies without risking real money.