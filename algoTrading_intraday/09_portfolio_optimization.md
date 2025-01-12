Here's a comprehensive guide to **implementing a portfolio optimization module using PyPortfolioOpt** and visualizing the portfolio performance in Django.

---

### **1. Install Required Libraries**
#### Python Libraries:
```bash
pip install PyPortfolioOpt pandas numpy matplotlib seaborn yfinance
```

---

### **2. Fetch Stock Data**
Use the `yfinance` library to fetch historical stock data.

```python
import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetches historical stock price data.
    :param tickers: List of stock tickers.
    :param start_date: Start date for fetching data (YYYY-MM-DD).
    :param end_date: End date for fetching data (YYYY-MM-DD).
    :return: DataFrame of adjusted close prices.
    """
    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    return data
```

---

### **3. Perform Portfolio Optimization**
Leverage **PyPortfolioOpt** to calculate optimal portfolio weights.

```python
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return

def optimize_portfolio(prices):
    """
    Optimize portfolio weights using mean-variance optimization.
    :param prices: DataFrame of adjusted close prices.
    :return: Optimized weights, expected return, volatility, and Sharpe ratio.
    """
    # Calculate expected returns and covariance matrix
    mu = mean_historical_return(prices)
    S = CovarianceShrinkage(prices).ledoit_wolf()

    # Perform mean-variance optimization
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()  # Maximize Sharpe ratio
    cleaned_weights = ef.clean_weights()

    # Get performance metrics
    performance = ef.portfolio_performance(verbose=True)

    return cleaned_weights, performance
```

---

### **4. Save Portfolio Data in Django**
#### Define a Model in `models.py`:
```python
from django.db import models

class Portfolio(models.Model):
    stock = models.CharField(max_length=10)  # Stock ticker
    weight = models.FloatField()  # Allocation weight
    date_created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.stock}: {self.weight:.2%}"
```

#### Save Optimized Portfolio to Database:
```python
def save_portfolio_to_db(weights):
    """
    Save portfolio weights to the database.
    :param weights: Dictionary of optimized weights {ticker: weight}.
    """
    from .models import Portfolio

    for stock, weight in weights.items():
        Portfolio.objects.create(stock=stock, weight=weight)
```

---

### **5. Create Django Views**
#### In `views.py`:
```python
from django.shortcuts import render
from .models import Portfolio
from .portfolio_optimizer import fetch_stock_data, optimize_portfolio, save_portfolio_to_db

def portfolio_view(request):
    # Stock tickers and date range
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    # Fetch stock data and optimize portfolio
    prices = fetch_stock_data(tickers, start_date, end_date)
    weights, performance = optimize_portfolio(prices)
    save_portfolio_to_db(weights)

    # Retrieve portfolio from the database
    portfolio = Portfolio.objects.all()

    return render(request, "portfolio.html", {
        "portfolio": portfolio,
        "performance": {
            "expected_return": f"{performance[0]:.2%}",
            "volatility": f"{performance[1]:.2%}",
            "sharpe_ratio": f"{performance[2]:.2f}",
        }
    })
```

---

### **6. Create Django Template**
#### In `templates/portfolio.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Optimization</title>
</head>
<body>
    <h1>Optimized Portfolio</h1>

    <h2>Performance Metrics</h2>
    <ul>
        <li><b>Expected Return:</b> {{ performance.expected_return }}</li>
        <li><b>Volatility:</b> {{ performance.volatility }}</li>
        <li><b>Sharpe Ratio:</b> {{ performance.sharpe_ratio }}</li>
    </ul>

    <h2>Portfolio Weights</h2>
    <table border="1">
        <tr>
            <th>Stock</th>
            <th>Weight</th>
        </tr>
        {% for item in portfolio %}
        <tr>
            <td>{{ item.stock }}</td>
            <td>{{ item.weight|floatformat:2 }}%</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
```

---

### **7. Visualize Portfolio Performance**
#### Generate a Pie Chart for Weights:
Use **Matplotlib** to create a pie chart for portfolio allocation.

```python
import matplotlib.pyplot as plt

def plot_portfolio_weights(weights):
    labels = list(weights.keys())
    sizes = list(weights.values())

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Portfolio Allocation")
    plt.savefig("static/portfolio_weights.png")
    plt.close()
```

Add this to your `portfolio_view`:
```python
plot_portfolio_weights(weights)
```

Modify the template to include the pie chart:
```html
<img src="/static/portfolio_weights.png" alt="Portfolio Allocation">
```

---

### **8. Automate Portfolio Updates**
#### Schedule Tasks with Celery:
Set up **Celery** to periodically fetch data, optimize the portfolio, and save it.

```python
from celery import shared_task

@shared_task
def update_portfolio():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    prices = fetch_stock_data(tickers, start_date, end_date)
    weights, _ = optimize_portfolio(prices)
    save_portfolio_to_db(weights)
```

---

### **Outcome**
This module will:
1. Fetch historical stock prices for selected tickers.
2. Optimize portfolio weights using **mean-variance optimization**.
3. Save the optimized portfolio in a Django database.
4. Visualize performance metrics and allocation in a user-friendly web interface.
5. Automate updates with scheduled tasks.