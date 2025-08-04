import requests
import logging
import asyncio
from typing import List, Dict, Optional
import os
import time

logger = logging.getLogger(__name__)

class FinnhubClient:
    """Client for Finnhub API - real financial market data"""
    
    def __init__(self):
        self.api_key = os.getenv('FINNHUB_API_KEY')
        self.base_url = "https://finnhub.io/api/v1"
        
    async def get_forex_candles(self, symbol: str, resolution: str = "1", count: int = 50) -> List[List]:
        """Get real forex candle data"""
        try:
            # Convert symbol format (EURUSD -> OANDA:EUR_USD)
            if len(symbol) == 6:
                base = symbol[:3]
                quote = symbol[3:]
                finnhub_symbol = f"OANDA:{base}_{quote}"
            else:
                finnhub_symbol = symbol
            
            # Calculate time range (last 7 days)
            end_time = int(time.time())
            start_time = end_time - (7 * 24 * 3600)  # 7 days ago
            
            params = {
                'symbol': finnhub_symbol,
                'resolution': resolution,
                'from': start_time,
                'to': end_time,
                'token': self.api_key
            }
            
            response = requests.get(f"{self.base_url}/forex/candle", params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('s') == 'ok' and 'c' in data:
                    ohlcv_data = []
                    
                    # Combine the arrays into OHLCV format
                    for i in range(len(data['c'])):
                        ohlcv_data.append([
                            data['t'][i] * 1000,  # Convert to milliseconds
                            data['o'][i],  # Open
                            data['h'][i],  # High
                            data['l'][i],  # Low
                            data['c'][i],  # Close
                            data['v'][i] if 'v' in data else 1000  # Volume
                        ])
                    
                    logger.info(f"✅ Retrieved {len(ohlcv_data)} real forex candles for {symbol} from Finnhub")
                    return ohlcv_data
                else:
                    logger.warning(f"Finnhub forex data status: {data.get('s', 'unknown')}")
                    
        except Exception as e:
            logger.error(f"Finnhub forex data error: {e}")
            
        return []
    
    async def get_crypto_quote(self, symbol: str) -> Dict:
        """Get real-time crypto quote"""
        try:
            # Convert to Binance format for Finnhub
            binance_symbol = f"BINANCE:{symbol.replace('/', '')}"
            
            params = {
                'symbol': binance_symbol,
                'token': self.api_key
            }
            
            response = requests.get(f"{self.base_url}/quote", params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'c' in data and data['c'] > 0:  # Current price exists
                    price = data['c']
                    spread = price * 0.001  # 0.1% spread
                    
                    return {
                        'bid': price - spread/2,
                        'ask': price + spread/2,
                        'last': price
                    }
                    
        except Exception as e:
            logger.error(f"Finnhub crypto quote error: {e}")
            
        return {}
    
    async def get_forex_rates(self, base: str = "USD") -> Dict:
        """Get real-time forex exchange rates"""
        try:
            params = {
                'base': base,
                'token': self.api_key
            }
            
            response = requests.get(f"{self.base_url}/forex/rates", params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'quote' in data:
                    rates = {}
                    for currency, rate in data['quote'].items():
                        if rate > 0:
                            rates[f"{base}{currency}"] = rate
                    
                    logger.info(f"✅ Retrieved {len(rates)} real forex rates from Finnhub")
                    return rates
                    
        except Exception as e:
            logger.error(f"Finnhub forex rates error: {e}")
            
        return {}