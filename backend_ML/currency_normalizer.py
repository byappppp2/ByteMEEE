import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

class CurrencyNormalizer:

    def __init__(self, cache_duration_hours: int = 24):

        self.cache_duration_hours = cache_duration_hours
        self.cache_file = "exchange_rates_cache.json"
        self.rates_cache = {}
        self.last_update = None
        
        # Fallback rates (updated periodically, used when API fails)
        self.fallback_rates = {
            'EUR': 1.08, 'GBP': 1.27, 'JPY': 0.0067, 'CAD': 0.74,
            'AUD': 0.66, 'CHF': 1.12, 'CNY': 0.14, 'INR': 0.012,
            'BRL': 0.20, 'MXN': 0.059, 'KRW': 0.00076, 'SGD': 0.74,
            'HKD': 0.13, 'NZD': 0.61, 'SEK': 0.095, 'NOK': 0.095,
            'DKK': 0.14, 'PLN': 0.25, 'CZK': 0.044, 'HUF': 0.0028,
            'RUB': 0.011, 'TRY': 0.034, 'ZAR': 0.055, 'THB': 0.028,
            'MYR': 0.21, 'IDR': 0.000064, 'PHP': 0.018, 'VND': 0.000041
        }
        
        self.load_cache()
    
    def load_cache(self):
        """Load cached exchange rates from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.rates_cache = cache_data.get('rates', {})
                    self.last_update = datetime.fromisoformat(cache_data.get('last_update', '2000-01-01'))
        except Exception as e:
            print(f"Warning: Could not load exchange rate cache: {e}")
            self.rates_cache = {}
            self.last_update = datetime(2000, 1, 1)
    
    def save_cache(self):
        """Save exchange rates to cache file"""
        try:
            cache_data = {
                'rates': self.rates_cache,
                'last_update': self.last_update.isoformat()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save exchange rate cache: {e}")
    
    def is_cache_valid(self) -> bool:
        """Check if cached rates are still valid"""
        if not self.last_update:
            return False
        
        time_diff = datetime.now() - self.last_update
        return time_diff < timedelta(hours=self.cache_duration_hours)
    
    def fetch_exchange_rates(self) -> Dict[str, float]:
        rates = {}

        api_sources = [
            self._fetch_from_exchangerate_api,
        ]
        
        for api_func in api_sources:
            try:
                rates = api_func()
                if rates:
                    print(f"Successfully fetched {len(rates)} exchange rates")
                    break
            except Exception as e:
                print(f"API source failed: {e}")
                continue
        
        # If all APIs fail, use fallback rates
        if not rates:
            print("All API sources failed, using fallback rates")
            rates = self.fallback_rates.copy()
        
        # Always include USD
        rates['USD'] = 1.0
        
        return rates
    
    def _fetch_from_exchangerate_api(self) -> Dict[str, float]:
        try:
            url = "https://api.exchangerate-api.com/v4/latest/USD"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            rates = {}

            for currency, rate in data.get('rates', {}).items():
                if currency != 'USD':
                    rates[currency] = 1.0 / rate
            
            return rates
        except Exception as e:
            raise Exception(f"ExchangeRate API error: {e}")

    
    def get_exchange_rate(self, currency: str) -> float:
        currency = currency.upper().strip()
        
        # Check if we need to update rates
        if not self.is_cache_valid():
            print("Updating exchange rates...")
            self.rates_cache = self.fetch_exchange_rates()
            self.last_update = datetime.now()
            self.save_cache()
        
        # Return cached rate or fallback
        rate = self.rates_cache.get(currency, self.fallback_rates.get(currency, 1.0))
        
        if currency not in self.rates_cache and currency not in self.fallback_rates:
            print(f"Warning: Unknown currency '{currency}', using rate of 1.0")
        
        return rate
    
    def normalize_to_usd(self, df: pd.DataFrame, 
                        amount_cols: list = None,
                        currency_cols: list = None) -> pd.DataFrame:
        df = df.copy()
        
        # Default column mappings
        if amount_cols is None:
            amount_cols = ['amount_paid', 'amount_received']
        if currency_cols is None:
            currency_cols = ['payment_currency', 'receiving_currency']
        
        # Ensure we have matching pairs
        if len(amount_cols) != len(currency_cols):
            raise ValueError("Number of amount columns must match currency columns")
        
        # Normalize each amount/currency pair
        for amount_col, currency_col in zip(amount_cols, currency_cols):
            if amount_col in df.columns and currency_col in df.columns:
                usd_col = f"{amount_col}_usd"
                
                # Get exchange rates for all unique currencies
                unique_currencies = df[currency_col].dropna().unique()
                exchange_rates = {}
                
                for currency in unique_currencies:
                    if pd.notna(currency) and str(currency).strip():
                        exchange_rates[currency] = self.get_exchange_rate(str(currency))
                
                # Apply conversion
                def convert_to_usd(row):
                    amount = row[amount_col]
                    currency = row[currency_col]
                    
                    if pd.isna(amount) or pd.isna(currency):
                        return np.nan
                    
                    currency = str(currency).strip().upper()
                    rate = exchange_rates.get(currency, 1.0)
                    
                    return float(amount) * rate
                
                df[usd_col] = df.apply(convert_to_usd, axis=1)
                
                # Add conversion info
                df[f"{amount_col}_original_currency"] = df[currency_col]
                df[f"{amount_col}_exchange_rate"] = df[currency_col].map(exchange_rates)
        
        return df
    
    def get_conversion_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        summary_data = []
        
        usd_cols = [col for col in df.columns if col.endswith('_usd')]
        
        for usd_col in usd_cols:
            base_col = usd_col.replace('_usd', '')
            currency_col = f"{base_col}_original_currency"
            rate_col = f"{base_col}_exchange_rate"
            
            if currency_col in df.columns and rate_col in df.columns:
                currency_stats = df.groupby(currency_col).agg({
                    usd_col: ['count', 'sum', 'mean', 'std'],
                    rate_col: 'first'
                }).round(4)
                
                currency_stats.columns = ['Transaction_Count', 'Total_USD', 'Avg_USD', 'Std_USD', 'Exchange_Rate']
                currency_stats = currency_stats.reset_index()
                currency_stats['Amount_Column'] = base_col
                
                summary_data.append(currency_stats)
        
        if summary_data:
            return pd.concat(summary_data, ignore_index=True)
        else:
            return pd.DataFrame()


def normalize_transaction_currencies(df: pd.DataFrame, 
                                   cache_duration_hours: int = 24) -> Tuple[pd.DataFrame, pd.DataFrame]:

    normalizer = CurrencyNormalizer(cache_duration_hours=cache_duration_hours)
    
    # Normalize amounts
    normalized_df = normalizer.normalize_to_usd(df)
    
    # Generate summary
    summary_df = normalizer.get_conversion_summary(normalized_df)
    
    return normalized_df, summary_df


if __name__ == "__main__":
    # Test the currency normalizer
    print("Testing Currency Normalizer...")
    
    # Create test data
    test_data = {
        'amount_paid': [1000, 500, 2000, 1500],
        'payment_currency': ['EUR', 'GBP', 'USD', 'JPY'],
        'amount_received': [1200, 600, 2000, 1800],
        'receiving_currency': ['USD', 'EUR', 'GBP', 'CAD']
    }
    
    test_df = pd.DataFrame(test_data)
    print("Original data:")
    print(test_df)
    
    # Normalize currencies
    normalized_df, summary_df = normalize_transaction_currencies(test_df)
    
    print("\nNormalized data:")
    print(normalized_df[['amount_paid', 'payment_currency', 'amount_paid_usd', 
                        'amount_received', 'receiving_currency', 'amount_received_usd']])
    
    print("\nConversion summary:")
    print(summary_df)
