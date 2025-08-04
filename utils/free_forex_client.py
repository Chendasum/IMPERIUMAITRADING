import requests
import logging
import time
from typing import Dict, List

logger = logging.getLogger(__name__)

class FreeFxRatesClient:
    """Client for free forex rate APIs that work without authentication"""
    
    def __init__(self):
        self.base_url = "https://api.exchangerate-api.com/v4/latest"
        
    async def get_forex_rates(self, base_currency: str = "USD") -> Dict[str, float]:
        """Get current forex rates from free API"""
        try:
            url = f"{self.base_url}/{base_currency}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                rates = data.get('rates', {})
                
                # Convert to our format (EURUSD, GBPUSD, etc.)
                forex_rates = {}
                for currency, rate in rates.items():
                    if currency != base_currency:
                        pair_symbol = f"{base_currency}{currency}"
                        forex_rates[pair_symbol] = rate
                
                logger.info(f"✅ Retrieved {len(forex_rates)} real forex rates from free API")
                return forex_rates
                
        except Exception as e:
            logger.error(f"❌ Free forex API failed: {e}")
            
        return {}
    
    async def create_ohlcv_from_rate(self, symbol: str, rate: float, limit: int = 50) -> List[List]:
        """Create OHLCV data from current rate"""
        try:
            current_time = int(time.time())
            ohlcv_data = []
            
            for i in range(limit):
                # Add small realistic variations (±0.05%)
                variation = 0.0005 * ((i % 5) - 2)  # Varies between -0.1% to +0.1%
                price = rate * (1 + variation)
                
                timestamp = current_time - ((limit - i) * 3600)  # 1 hour intervals
                ohlcv_data.append([
                    timestamp * 1000,  # Milliseconds
                    round(price, 5),   # Open
                    round(price * 1.0005, 5),  # High
                    round(price * 0.9995, 5),  # Low
                    round(price, 5),   # Close
                    1000  # Volume placeholder
                ])
            
            return ohlcv_data
            
        except Exception as e:
            logger.error(f"❌ Failed to create OHLCV from rate: {e}")
            return []

class CoinGeckoClient:
    """Client for CoinGecko free crypto API"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        
    async def get_crypto_prices(self) -> Dict[str, Dict]:
        """Get real crypto prices from CoinGecko free API"""
        try:
            # Map crypto symbols to CoinGecko IDs
            crypto_map = {
                'BTC/USDT': 'bitcoin',
                'ETH/USDT': 'ethereum', 
                'BNB/USDT': 'binancecoin',
                'SOL/USDT': 'solana',
                'XRP/USDT': 'ripple',
                'ADA/USDT': 'cardano',
                'AVAX/USDT': 'avalanche-2',
                'DOT/USDT': 'polkadot'
            }
            
            coins = ','.join(crypto_map.values())
            url = f"{self.base_url}/simple/price?ids={coins}&vs_currencies=usd&include_24hr_change=true"
            
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                prices = {}
                
                for symbol, coin_id in crypto_map.items():
                    if coin_id in data and 'usd' in data[coin_id]:
                        price = data[coin_id]['usd']
                        spread = price * 0.001  # 0.1% spread
                        
                        prices[symbol] = {
                            'bid': round(price - spread/2, 2),
                            'ask': round(price + spread/2, 2),
                            'last': price,
                            'change_24h': data[coin_id].get('usd_24h_change', 0)
                        }
                
                logger.info(f"✅ Retrieved real crypto prices from CoinGecko for {len(prices)} pairs")
                return prices
                
        except Exception as e:
            logger.error(f"❌ CoinGecko API failed: {e}")
            
        return {}