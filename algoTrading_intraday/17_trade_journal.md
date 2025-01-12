Creating a **Django-based trade journal** involves setting up models for logging trades, recording rationales, and tracking performance. Additionally, the trade journal should allow users to manage their trade entries, view performance over time, and provide insights into their trading strategy.

Below is a step-by-step guide to building the trade journal:

### **Step 1: Set up Django Project and App**
First, create a new Django project and app for the trade journal.

```bash
django-admin startproject tradejournal
cd tradejournal
python manage.py startapp journal
```

### **Step 2: Define Models for Trades and Performance**
In the `journal` app, you’ll define the models to track trades and store related information.

Open `journal/models.py` and define the models:

```python
from django.db import models
from django.contrib.auth.models import User

# Define a model to store a trade
class Trade(models.Model):
    # Define possible outcomes of the trade
    OUTCOME_CHOICES = (
        ('win', 'Win'),
        ('loss', 'Loss'),
        ('break_even', 'Break Even'),
    )

    # Trade details
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Link trade to a user
    stock_symbol = models.CharField(max_length=10)  # Stock ticker symbol (e.g., 'AAPL')
    entry_date = models.DateTimeField()  # Date and time of entry
    exit_date = models.DateTimeField()  # Date and time of exit
    entry_price = models.DecimalField(max_digits=10, decimal_places=2)  # Entry price per share
    exit_price = models.DecimalField(max_digits=10, decimal_places=2)  # Exit price per share
    quantity = models.IntegerField()  # Number of shares traded
    outcome = models.CharField(max_length=15, choices=OUTCOME_CHOICES)  # Trade outcome
    rationale = models.TextField()  # Reasoning behind the trade decision
    profit_loss = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)  # Profit or loss

    def save(self, *args, **kwargs):
        # Calculate profit or loss on trade
        if self.outcome == 'win':
            self.profit_loss = (self.exit_price - self.entry_price) * self.quantity
        elif self.outcome == 'loss':
            self.profit_loss = (self.entry_price - self.exit_price) * self.quantity
        else:
            self.profit_loss = 0
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.stock_symbol} - {self.entry_date}"

# Model to store performance overview over time
class Performance(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    total_trades = models.IntegerField(default=0)
    total_wins = models.IntegerField(default=0)
    total_losses = models.IntegerField(default=0)
    total_profit_loss = models.DecimalField(max_digits=15, decimal_places=2, default=0)

    def update_performance(self):
        trades = Trade.objects.filter(user=self.user)
        self.total_trades = trades.count()
        self.total_wins = trades.filter(outcome='win').count()
        self.total_losses = trades.filter(outcome='loss').count()
        self.total_profit_loss = trades.aggregate(models.Sum('profit_loss'))['profit_loss__sum'] or 0
        self.save()

    def __str__(self):
        return f"{self.user.username} Performance"
```

### **Step 3: Admin Setup**
To manage trades and performance easily, register the models in the Django admin panel.

Open `journal/admin.py`:

```python
from django.contrib import admin
from .models import Trade, Performance

admin.site.register(Trade)
admin.site.register(Performance)
```

### **Step 4: Create Forms for Trade Entries**
Create forms to allow users to input trade data easily. Open `journal/forms.py`:

```python
from django import forms
from .models import Trade

class TradeForm(forms.ModelForm):
    class Meta:
        model = Trade
        fields = ['stock_symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'quantity', 'outcome', 'rationale']
        widgets = {
            'entry_date': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
            'exit_date': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
        }
```

### **Step 5: Views to Handle Trade Entries and Performance**
Now, define views to display the trade journal, add new trades, and display performance data.

Open `journal/views.py`:

```python
from django.shortcuts import render, redirect
from .models import Trade, Performance
from .forms import TradeForm
from django.contrib.auth.decorators import login_required

# View to display trade journal
@login_required
def journal_home(request):
    trades = Trade.objects.filter(user=request.user)
    performance = Performance.objects.get(user=request.user)
    return render(request, 'journal/home.html', {'trades': trades, 'performance': performance})

# View to add a new trade
@login_required
def add_trade(request):
    if request.method == 'POST':
        form = TradeForm(request.POST)
        if form.is_valid():
            trade = form.save(commit=False)
            trade.user = request.user
            trade.save()
            # Update performance after adding trade
            performance, created = Performance.objects.get_or_create(user=request.user)
            performance.update_performance()
            return redirect('journal_home')
    else:
        form = TradeForm()
    return render(request, 'journal/add_trade.html', {'form': form})
```

### **Step 6: URL Routing**
Define the URL paths for the views. Open `journal/urls.py`:

```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.journal_home, name='journal_home'),
    path('add/', views.add_trade, name='add_trade'),
]
```

### **Step 7: Templates**
Create the necessary HTML templates for adding trades and displaying the journal.

#### `journal/templates/journal/home.html` (trade journal overview):

```html
<!DOCTYPE html>
<html>
<head>
    <title>Trade Journal</title>
</head>
<body>
    <h1>Trade Journal</h1>
    <p><strong>Total Trades:</strong> {{ performance.total_trades }}</p>
    <p><strong>Total Wins:</strong> {{ performance.total_wins }}</p>
    <p><strong>Total Losses:</strong> {{ performance.total_losses }}</p>
    <p><strong>Total Profit/Loss:</strong> {{ performance.total_profit_loss }}</p>
    <a href="{% url 'add_trade' %}">Add New Trade</a>
    <table>
        <tr>
            <th>Stock Symbol</th>
            <th>Entry Date</th>
            <th>Exit Date</th>
            <th>Entry Price</th>
            <th>Exit Price</th>
            <th>Quantity</th>
            <th>Outcome</th>
            <th>Profit/Loss</th>
        </tr>
        {% for trade in trades %}
        <tr>
            <td>{{ trade.stock_symbol }}</td>
            <td>{{ trade.entry_date }}</td>
            <td>{{ trade.exit_date }}</td>
            <td>{{ trade.entry_price }}</td>
            <td>{{ trade.exit_price }}</td>
            <td>{{ trade.quantity }}</td>
            <td>{{ trade.outcome }}</td>
            <td>{{ trade.profit_loss }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
```

#### `journal/templates/journal/add_trade.html` (add new trade):

```html
<!DOCTYPE html>
<html>
<head>
    <title>Add New Trade</title>
</head>
<body>
    <h1>Add New Trade</h1>
    <form method="post">
        {% csrf_token %}
        {{ form.as_p }}
        <button type="submit">Add Trade</button>
    </form>
</body>
</html>
```

### **Step 8: Finalize and Run**
Now, make sure the migrations are applied to create the database tables:

```bash
python manage.py makemigrations
python manage.py migrate
```

Create a superuser if you haven't already:

```bash
python manage.py createsuperuser
```

Finally, run the development server:

```bash
python manage.py runserver
```

### **Step 9: Access the Trade Journal**
- Go to `/admin` to log in and add trades through the Django admin interface.
- Go to `/` to view the trade journal and performance summary.

---

### **Key Features:**
1. **Trade Logging**: Users can log trades with stock symbol, entry and exit price, quantity, and rationale.
2. **Profit/Loss Calculation**: The platform automatically calculates profit or loss for each trade.
3. **Performance Tracking**: The performance page provides insights like the total number of trades, wins, losses, and overall profit or loss.
4. **Trade Management**: Users can add and view their past trades along with the rationale behind each decision.

This setup provides a full-fledged trade journal for tracking and analyzing trades, perfect for personal or professional use. You can also enhance this by adding more features like data visualization, trade filtering, and automated trade logging with API integrations.