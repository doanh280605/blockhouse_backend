from django.db import models

class StockPrice(models.Model):
    symbol = models.CharField(max_length=10)
    open_price = models.DecimalField(max_digits=10, decimal_places=2)
    close_price = models.DecimalField(max_digits=10, decimal_places=2)
    high_price = models.DecimalField(max_digits=10, decimal_places=2)
    low_price = models.DecimalField(max_digits=10, decimal_places=2)
    volume = models.BigIntegerField()
    timestamp = models.DateTimeField()

    class Meta:
        indexes = [
            models.Index(fields=['timestamp']),
        ]
        verbose_name = "Stock Price"
        verbose_name_plural = "Stock Prices"

    def __str__(self):
        return f"{self.symbol} on {self.timestamp}"

class StockPrediction(models.Model): 
    symbol = models.CharField(max_length=10)
    predicted_price = models.FloatField()
    timestamp = models.DateTimeField()

    def __str__(self):
        return f'{self.symbol} - {self.timestamp} - {self.predicted_price}'