import os
import pandas as pd
from django.core.management.base import BaseCommand
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from blockhouse_data.models import StockPrice

class Command(BaseCommand):
    help = 'Train a stock price prediction model'

    def handle(self, *args, **kwargs):
        # Fetch historical stock data
        historical_data = StockPrice.objects.all().order_by('timestamp')
        if not historical_data.exists():
            self.stdout.write(self.style.ERROR('No historical data available to train the model'))
            return

        # Convert to DataFrame
        df = pd.DataFrame(list(historical_data.values()))

        # Ensure enough data is available
        if len(df) < 100:
            self.stdout.write(self.style.ERROR('Not enough historical data to train the model'))
            return

        # Use close_price as the feature and timestamp as index
        df.set_index('timestamp', inplace=True)
        X = df[['close_price']].values
        y = df['close_price'].shift(-1).fillna(df['close_price']).values  # Next day's price

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Save the trained model to a file
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'stock_model.pkl')
        joblib.dump(model, model_path)

        self.stdout.write(self.style.SUCCESS(f'Model trained and saved to {model_path}'))
