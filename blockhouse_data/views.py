import requests
from django.http import JsonResponse, HttpResponse
from django.views import View
from django.utils import timezone
from .models import StockPrice, StockPrediction
import pandas as pd
import numpy as np
import io
from django.core.exceptions import ValidationError
import joblib
import os
import json
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import tempfile

class FetchStockDataView(View):
    API_KEY = ''
    BASE_URL = 'https://www.alphavantage.co/query'

    def get(self, request):
        symbol = request.GET.get('symbol')
        function = 'TIME_SERIES_INTRADAY'
        interval = '60min' 

        params = {
            'function': function,
            'symbol': symbol,
            'interval': interval,
            'apikey': self.API_KEY
        }

        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract time series data
            time_series = data.get('Time Series (60min)', {})
            stock_prices = {}

            for timestamp, values in time_series.items():
                stock_prices[timestamp] = {
                    "1. open": values['1. open'],
                    "2. high": values['2. high'],
                    "3. low": values['3. low'],
                    "4. close": values['4. close'],
                    "5. volume": values['5. volume']
                }

                # Store the stock price in the database
                StockPrice.objects.create(
                    symbol=symbol,
                    open_price=values['1. open'],
                    close_price=values['4. close'],
                    high_price=values['2. high'],
                    low_price=values['3. low'],
                    volume=values['5. volume'],
                    timestamp=timezone.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                )

            return JsonResponse(stock_prices, status=200)
        
        except requests.exceptions.HTTPError as http_err:
            return JsonResponse({"error": str(http_err)}, status=400)
        except Exception as err:
            return JsonResponse({"error": str(err)}, status=500)
        

class BacktestView(View):
    def get(self, request):
        try:
            # Get parameters from request
            symbol = request.GET.get('symbol')
            initial_investment = float(request.GET.get('initial_investment', 10000))
            short_ma = int(request.GET.get('short_ma', 50))
            long_ma = int(request.GET.get('long_ma', 200))

            # Input validation
            if not symbol:
                raise ValidationError('Symbol is required')
            if initial_investment <= 0:
                raise ValidationError('Initial investment must be positive')
            if short_ma >= long_ma:
                raise ValidationError('Short MA must be less than Long MA')

            # Fetch historical data from database
            historical_data = StockPrice.objects.filter(
                symbol=symbol
            ).order_by('timestamp')

            if not historical_data:
                raise ValidationError('No historical data available for this symbol')

            # Convert to DataFrame
            df = pd.DataFrame(list(historical_data.values()))
            df.set_index('timestamp', inplace=True)
            
            # Calculate moving averages
            df['short_ma'] = df['close_price'].rolling(window=short_ma).mean()
            df['long_ma'] = df['close_price'].rolling(window=long_ma).mean()
            
            # Initialize tracking variables
            position = 0  # 0: no position, 1: long position
            cash = initial_investment
            shares = 0
            trades = []
            equity_curve = []
            
            for i in range(len(df)):
                if i < long_ma:  # Skip until we have enough data for both MAs
                    continue
                    
                current_price = float(df.iloc[i]['close_price'])
                current_date = df.index[i]
                
                # Buy Signal: Stock price goes below Short MA
                if position == 0 and current_price < df.iloc[i]['short_ma']:
                    shares = cash / current_price
                    cash = 0
                    position = 1
                    trades.append({
                        'date': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'BUY',
                        'price': current_price,
                        'shares': shares,
                        'value': shares * current_price
                    })

                # Sell Signal: Stock price goes above Long MA
                elif position == 1 and current_price > df.iloc[i]['long_ma']:
                    cash = shares * current_price
                    shares = 0
                    position = 0
                    trades.append({
                        'date': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'SELL',
                        'price': current_price,
                        'cash': cash,
                        'return': ((cash - initial_investment) / initial_investment) * 100
                    })
                
                # Track equity curve
                current_equity = cash + (shares * current_price)
                equity_curve.append({
                    'date': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'equity': current_equity
                })
            
            # Calculate performance metrics
            final_equity = equity_curve[-1]['equity'] if equity_curve else initial_investment
            total_return = ((final_equity - initial_investment) / initial_investment) * 100
            
            # Calculate max drawdown
            peak = equity_curve[0]['equity']
            max_drawdown = 0
            for point in equity_curve:
                if point['equity'] > peak:
                    peak = point['equity']
                drawdown = (peak - point['equity']) / peak * 100
                max_drawdown = min(max_drawdown, -drawdown)
            
            # Prepare results
            results = {
                'symbol': symbol,
                'initial_investment': initial_investment,
                'final_equity': round(final_equity, 2),
                'total_return_pct': round(total_return, 2),
                'max_drawdown_pct': round(max_drawdown, 2),
                'number_of_trades': len(trades),
                'trades': trades,
                'equity_curve': equity_curve
            }
            
            return JsonResponse({
                'success': True,
                'results': results
            })

        except ValidationError as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

class PredictStockPriceView(View):
    def get(self, request): 
        try:
            symbol = request.GET.get('symbol')
            if not symbol:
                raise ValidationError('Symbol is required!')

            historical_data = StockPrice.objects.filter(symbol=symbol).order_by('timestamp')
            if not historical_data.exists():
                raise ValidationError('No historical data available for this symbol')

            df = pd.DataFrame(list(historical_data.values()))
            df.set_index('timestamp', inplace=True)

            # Ensure enough data is available for model prediction
            if len(df) < 30: 
                raise ValidationError('Not enough historical data for prediction!')

            model_path = os.path.join(os.path.dirname(__file__), 'models', 'stock_model.pkl')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Model file not found at {model_path}')
            
            model = joblib.load(model_path)

            feature_data = df[['close_price']]

            # Predict for the next 30 days
            future_dates = pd.date_range(df.index[-1], periods=31, freq='D')[1:] 
            predictions = model.predict(feature_data[-30:])

            # Store predictions in the database
            predicted_data = []
            for date, price in zip(future_dates, predictions):
                prediction = StockPrediction(
                    symbol=symbol,
                    predicted_price=price,
                    timestamp=date
                )
                predicted_data.append(prediction)

            StockPrediction.objects.bulk_create(predicted_data)

            prediction_response = [{'date': date.strftime('%Y-%m-%d'), 'predicted_price': price} for date, price in zip(future_dates, predictions)]

            return JsonResponse({"symbol": symbol, "predictions": prediction_response}, status=200)

        except ValidationError as ve:
            return JsonResponse({"error": str(ve)}, status=400)
        except FileNotFoundError as fnfe:
            return JsonResponse({"error": str(fnfe)}, status=500)
        except Exception as e:
            return JsonResponse({"error": f"An unexpected error occurred: {str(e)}"}, status=500)
        

class GenerateReportView(View):
    def get(self, request):
        try:
            symbol = request.GET.get('symbol')
            if not symbol:
                raise ValidationError('Symbol is required!')
            
            historical_data = StockPrice.objects.filter(symbol=symbol).order_by('timestamp')
            predicted_data = StockPrediction.objects.filter(symbol=symbol).order_by('timestamp')

            if not historical_data.exists():
                raise ValidationError(f'No historical data found for symbol {symbol}')
            if not predicted_data.exists():
                raise ValidationError(f'No prediction data found for symbol {symbol}')

            actual_prices = pd.DataFrame(list(historical_data.values('timestamp', 'close_price', 'symbol')))
            predicted_prices = pd.DataFrame(list(predicted_data.values('timestamp', 'predicted_price', 'symbol')))

            actual_prices['timestamp'] = pd.to_datetime(actual_prices['timestamp'])
            predicted_prices['timestamp'] = pd.to_datetime(predicted_prices['timestamp'])
            
            actual_prices = actual_prices.rename(columns={'close_price': 'price'})
            predicted_prices = predicted_prices.rename(columns={'predicted_price': 'price'})

            actual_prices['type'] = 'actual'
            predicted_prices['type'] = 'predicted'

            combined_data = pd.concat([
                actual_prices[['timestamp', 'price', 'type']],
                predicted_prices[['timestamp', 'price', 'type']]
            ]).sort_values('timestamp')

            # Calculate metrics for overlapping period (if any)
            overlapping_metrics = {}
            overlap_data = pd.merge(
                actual_prices[['timestamp', 'price']],
                predicted_prices[['timestamp', 'price']],
                on='timestamp',
                suffixes=('_actual', '_predicted'),
                how='inner'
            )
            
            if not overlap_data.empty:
                overlapping_metrics = {
                    "mean_squared_error": mean_squared_error(
                        overlap_data['price_actual'], 
                        overlap_data['price_predicted']
                    ),
                    "data_points": len(overlap_data)
                }

            # Calculate general metrics
            metrics = {
                "historical_data": {
                    "start_date": actual_prices['timestamp'].min().strftime('%Y-%m-%d'),
                    "end_date": actual_prices['timestamp'].max().strftime('%Y-%m-%d'),
                    "data_points": len(actual_prices),
                    "avg_price": float(actual_prices['price'].mean()),
                    "min_price": float(actual_prices['price'].min()),
                    "max_price": float(actual_prices['price'].max())
                },
                "prediction_data": {
                    "start_date": predicted_prices['timestamp'].min().strftime('%Y-%m-%d'),
                    "end_date": predicted_prices['timestamp'].max().strftime('%Y-%m-%d'),
                    "data_points": len(predicted_prices),
                    "avg_price": float(predicted_prices['price'].mean()),
                    "min_price": float(predicted_prices['price'].min()),
                    "max_price": float(predicted_prices['price'].max())
                },
                "overlapping_metrics": overlapping_metrics
            }

            # Create figure and axis objects
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot actual prices
            actual_mask = combined_data['type'] == 'actual'
            ax.plot(
                combined_data[actual_mask]['timestamp'],
                combined_data[actual_mask]['price'],
                label="Historical Price",
                color='blue'
            )
            
            # Plot predicted prices
            predicted_mask = combined_data['type'] == 'predicted'
            ax.plot(
                combined_data[predicted_mask]['timestamp'],
                combined_data[predicted_mask]['price'],
                label="Predicted Price",
                color='orange',
                linestyle='--'
            )
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Stock Price')
            ax.set_title(f'Stock Price History and Prediction for {symbol}')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot to buffer
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            buffer.seek(0)

            # Handle response format
            response_format = request.GET.get('format', 'json')

            if response_format == 'pdf':
                response = HttpResponse(content_type='application/pdf')
                response['Content-Disposition'] = f'attachment; filename="{symbol}_report.pdf"'

                pdf_buffer = io.BytesIO()
                pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
                
                # title
                pdf.setFont("Helvetica-Bold", 16)
                pdf.drawString(100, 750, f"Stock Analysis Report for {symbol}")
                
                # metrics
                pdf.setFont("Helvetica", 12)
                y = 730
                
                # Historical data section
                pdf.drawString(100, y, "Historical Data:")
                y -= 20
                pdf.drawString(120, y, f"Period: {metrics['historical_data']['start_date']} to {metrics['historical_data']['end_date']}")
                y -= 20
                pdf.drawString(120, y, f"Data Points: {metrics['historical_data']['data_points']}")
                y -= 20
                pdf.drawString(120, y, f"Average Price: ${metrics['historical_data']['avg_price']:.2f}")
                
                # Prediction data section
                y -= 40
                pdf.drawString(100, y, "Prediction Data:")
                y -= 20
                pdf.drawString(120, y, f"Period: {metrics['prediction_data']['start_date']} to {metrics['prediction_data']['end_date']}")
                y -= 20
                pdf.drawString(120, y, f"Data Points: {metrics['prediction_data']['data_points']}")
                y -= 20
                pdf.drawString(120, y, f"Average Price: ${metrics['prediction_data']['avg_price']:.2f}")
                
                # Add overlapping metrics if available
                if overlapping_metrics:
                    y -= 40
                    pdf.drawString(100, y, "Overlapping Period Analysis:")
                    y -= 20
                    pdf.drawString(120, y, f"Mean Squared Error: {overlapping_metrics['mean_squared_error']:.2f}")
                    y -= 20
                    pdf.drawString(120, y, f"Overlapping Data Points: {overlapping_metrics['data_points']}")
                
                with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as temp_file:
                    fig.savefig(temp_file.name, format='png', dpi=300, bbox_inches='tight')
                    pdf.drawImage(temp_file.name, 100, 100, width=400, height=200)

                pdf.showPage()
                pdf.save()

                response.write(pdf_buffer.getvalue())
                pdf_buffer.close()
                return response
            else:
                # Prepare JSON response
                response_data = {
                    "symbol": symbol,
                    "metrics": metrics,
                    "data": {
                        "historical": actual_prices[['timestamp', 'price']].to_dict(orient='records'),
                        "predicted": predicted_prices[['timestamp', 'price']].to_dict(orient='records')
                    }
                }
                return JsonResponse(response_data, status=200)

        except ValidationError as e:
            return JsonResponse({"error": str(e)}, status=400)
        except Exception as e:
            return JsonResponse({"error": f"Unexpected error: {str(e)}"}, status=500)