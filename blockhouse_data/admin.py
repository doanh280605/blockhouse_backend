from django.contrib import admin

# Register your models here.
from .models import StockPrediction, StockPrice

admin.site.register(StockPrediction)
admin.site.register(StockPrice)