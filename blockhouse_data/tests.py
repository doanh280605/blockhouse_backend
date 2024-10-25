import unittest
from django.test import TestCase, Client
from django.urls import reverse
from django.utils import timezone

from blockhouse_data.models import StockPrice

class BacktestingTests(TestCase):
    def setUp(self):
        self.client = Client()
        # Add sample stock data to test database
        StockPrice.objects.create(
            symbol='AAPL',
            open_price=150.0,
            close_price=151.0,
            high_price=152.0,
            low_price=149.0,
            volume=1000000,
            timestamp=timezone.now()
        )
        
    def test_valid_backtest_request(self):
        response = self.client.get(reverse('backtest'), {
            'symbol': 'AAPL',
            'initial_investment': 10000,
            'short_window': 50,
            'long_window': 200
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['success'])
        
    def test_invalid_parameters(self):
        response = self.client.get(reverse('backtest'), {
            'symbol': 'AAPL',
            'initial_investment': -1000,  # Invalid negative investment
            'short_window': 50,
            'long_window': 200
        })
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json()['success'])
        
    def test_missing_symbol(self):
        response = self.client.get(reverse('backtest'), {
            'initial_investment': 10000,
            'short_window': 50,
            'long_window': 200
        })
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json()['success'])

