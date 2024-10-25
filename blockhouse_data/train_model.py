from django.core.management.base import BaseCommand
from blockhouse_data.models import StockPrice
import os
import joblib
from sklearn.linear_model import LinearRegression
import pandas as pd

class Command(BaseCommand):
    # Add symbol as an argument
    def add_arguments(self, parser):
        parser.add_argument(
            '--symbol', type=str, help='Stock symbol for which to train the model'
        )

    def handle(self, *args, **kwargs):
        # Fetch the symbol from the command line arguments
        symbol = kwargs['symbol']
        
        if not symbol:
            self.stdout.write(self.style.ERROR('Stock symbol is required.'))
            return
        
        # Fetch historical data based on the symbol
        df = self.fetch_historical_data(symbol)
        
        if df.empty:
            self.stdout.write(self.style.ERROR(f'No historical data found for symbol: {symbol}'))
            return

        # Train the model
        model = self.train_model(df)
        
        # Save the model as a .pkl file
        self.save_model(model, f'{symbol}_model.pkl')
        self.stdout.write(self.style.SUCCESS(f'Model trained and saved successfully for {symbol}.'))

    def fetch_historical_data(self, symbol):
        # Fetch historical stock data from the StockPrice model for the given symbol
        stock_data = StockPrice.objects.filter(symbol=symbol).values()
        df = pd.DataFrame(list(stock_data))
        return df

    def train_model(self, df):
        # Define your features (e.g., historical data columns) and target (e.g., price)
        X = df[['feature1', 'feature2']]  # Replace with your actual feature columns
        y = df['close_price']  # Replace with actual target column
        model = LinearRegression()
        model.fit(X, y)
        return model

    def save_model(self, model, filename):
        # Save the trained model to a .pkl file
        model_dir = os.path.join('models')  # Adjust as necessary for your file path
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, filename)
        joblib.dump(model, model_path)
