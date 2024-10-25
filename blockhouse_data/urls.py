from django.urls import path
from .views import FetchStockDataView, BacktestView, PredictStockPriceView, GenerateReportView

urlpatterns = [
    path('fetch-stock-data/', FetchStockDataView.as_view(), name='fetch_stock_data'),
    path('backtest/', BacktestView.as_view(), name='backtest'),
    path('predict-stock-price/', PredictStockPriceView.as_view(), name='predict-stock-price'),
    path('generate-report/', GenerateReportView.as_view(), name='generate-report'),
]
