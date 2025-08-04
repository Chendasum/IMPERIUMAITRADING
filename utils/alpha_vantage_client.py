import requests
import logging
import asyncio
from typing import List, Dict, Optional
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class AlphaVantageClient:
    """Client for Alpha Vantage API - real market data provider"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = "https://www.alphavantage.co/query"
        
    async def get_forex_intraday(self, from_symbol: str, to_symbol: str, interval: str = "1min") -> List[List]:
        """Get real-time forex intraday data"""
        try:
            params = {
                'function': 'FX_INTRADAY',
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'interval': interval,
                'apikey': self.api_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for API limit or error
                if 'Error Message' in data:
                    logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                    return []
                
                if 'Note' in data:
                    logger.warning(f"Alpha Vantage API limit: {data['Note']}")
                    return []
                
                # Extract time series data
                time_series_key = f"Time Series FX ({interval})"
                if time_series_key in data:
                    time_series = data[time_series_key]
                    
                    ohlcv_data = []
                    for timestamp, values in list(time_series.items())[:50]:  # Last 50 data points
                        # Convert timestamp to Unix milliseconds
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        unix_ms = int(dt.timestamp() * 1000)
                        ohlcv_data.append([
                            unix_ms,
                            float(values['1. open']),
                            float(values['2. high']),
                            float(values['3. low']),
                            float(values['4. close']),
                            1000  # Volume placeholder for forex
                        ])
                    
                    logger.info(f"✅ Retrieved {len(ohlcv_data)} real forex data points for {from_symbol}/{to_symbol}")
                    return ohlcv_data
                
            else:
                logger.error(f"Alpha Vantage API request failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Alpha Vantage forex data error: {e}")
            
        return []
    
    async def get_crypto_intraday(self, symbol: str, market: str = "USD") -> Dict:
        """Get real-time crypto data"""
        try:
            params = {
                'function': 'CRYPTO_INTRADAY',
                'symbol': symbol,
                'market': market,
                'interval': '1min',
                'apikey': self.api_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for errors
                if 'Error Message' in data or 'Note' in data:
                    return {}
                
                # Extract latest price data
                time_series_key = "Time Series (Crypto)"
                if time_series_key in data:
                    time_series = data[time_series_key]
                    latest_data = next(iter(time_series.values()))
                    
                    price = float(latest_data['4. close'])
                    spread = price * 0.001  # 0.1% spread
                    
                    return {
                        'bid': price - spread/2,
                        'ask': price + spread/2,
                        'last': price
                    }
                    
        except Exception as e:
            logger.error(f"Alpha Vantage crypto data error: {e}")
            
        return {}