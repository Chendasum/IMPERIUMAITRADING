import requests
import logging
import asyncio
from typing import List, Dict, Optional
import os
import time

logger = logging.getLogger(__name__)

class FinnhubClient:
    """Client for Finnhub API - Real-time financial market data"""

    def __init__(self):
        self.api_key = os.getenv('FINNHUB_API_KEY')
        self.base_url = "https://finnhub.io/api/v1"

    async def get_forex_candles(self, symbol: str, resolution: str = "1", count: int = 50) -> List[List]:
        """Get recent OHLCV candles for a forex symbol."""
        try:
            if len(symbol) == 6:
                base = symbol[:3]
                quote = symbol[3:]
                finnhub_symbol = f"OANDA:{base}_{quote}"
            else:
                finnhub_symbol = symbol

            end_time = int(time.time())
            start_time = end_time - (7 * 24 * 3600)  # Last 7 days

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
                    ohlcv_data = [
                        [data['t'][i] * 1000, data['o'][i], data['h'][i], data['l'][i], data['c'][i], data.get('v', [1000])[i]]
                        for i in range(len(data['c']))
                    ]
                    logger.info(f"✅ Retrieved {len(ohlcv_data)} forex candles for {symbol}")
                    return ohlcv_data
                else:
                    logger.warning(f"⚠️ Finnhub forex response status: {data.get('s')}")
        except Exception as e:
            logger.error(f"❌ get_forex_candles error: {e}")
        return []

    async def get_crypto_quote(self, symbol: str) -> Dict:
        """Get real-time crypto quote from Binance feed via Finnhub."""
        try:
            binance_symbol = f"BINANCE:{symbol.replace('/', '')}"
            params = {
                'symbol': binance_symbol,
                'token': self.api_key
            }

            response = requests.get(f"{self.base_url}/quote", params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'c' in data and data['c'] > 0:
                    price = data['c']
                    spread = price * 0.001
                    return {
                        'bid': price - spread / 2,
                        'ask': price + spread / 2,
                        'last': price
                    }
        except Exception as e:
            logger.error(f"❌ get_crypto_quote error: {e}")
        return {}

    async def get_forex_rates(self, base: str = "USD") -> Dict:
        """Get current forex exchange rates relative to base."""
        try:
            params = {
                'base': base,
                'token': self.api_key
            }
            response = requests.get(f"{self.base_url}/forex/rates", params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'quote' in data:
                    rates = {
                        f"{base}{currency}": rate
                        for currency, rate in data['quote'].items()
                        if rate > 0
                    }
                    logger.info(f"✅ Retrieved {len(rates)} forex rates from Finnhub")
                    return rates
        except Exception as e:
            logger.error(f"❌ get_forex_rates error: {e}")
        return {}

# =============================
# ✅ MetaAPI Fallback Function
# =============================

def get_finnhub_price(symbol: str) -> Optional[float]:
    """
    Fallback sync wrapper for getting latest close price of forex pair.
    Used if MetaAPI fails. Symbol example: 'EURUSD'
    """
    try:
        client = FinnhubClient()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        candles = loop.run_until_complete(client.get_forex_candles(symbol, resolution="1", count=1))
        if candles:
            close_price = candles[-1][4]  # Close value
            print(f"✅ Finnhub fallback price for {symbol}: {close_price}")
            return close_price
    except Exception as e:
        print(f"❌ get_finnhub_price fallback error: {e}")
    return None
