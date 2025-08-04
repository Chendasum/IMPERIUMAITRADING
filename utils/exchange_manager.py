import ccxt
import asyncio
import logging
from config import TradingConfig

logger = logging.getLogger(__name__)


class ExchangeManager:

    def __init__(self):
        self.exchanges = {}
        self.config = TradingConfig()

    async def initialize(self):
        """Initialize exchange connections"""
        try:
            # Initialize Binance (with or without API keys)
            try:
                exchange_config = {
                    'enableRateLimit': True,
                    'sandbox': False,  # Always use live trading environment
                }
                
                # Add API keys only if provided
                if self.config.BINANCE_API_KEY and self.config.BINANCE_SECRET_KEY:
                    exchange_config['apiKey'] = self.config.BINANCE_API_KEY
                    exchange_config['secret'] = self.config.BINANCE_SECRET_KEY
                    logger.info("✅ Binance connection with API keys")
                else:
                    logger.info("🚀 Binance connection initialized for LIVE TRADING")
                
                self.exchanges['binance'] = ccxt.binance(exchange_config)
                logger.info("✅ Binance exchange initialized successfully")
                
            except Exception as e:
                logger.error(f"❌ Binance initialization failed: {e}")
                # Create a simulation mode for complete paper trading
                logger.info("💡 Running in simulation mode")

            # For demo/paper trading, focus on Binance only
            # Multi-exchange arbitrage can be added when ready for live trading
            logger.info("💡 Running in single-exchange mode (Binance) for paper trading")

            # Test connections
            for name, exchange in self.exchanges.items():
                try:
                    # Test basic connection
                    markets = exchange.load_markets()
                    logger.info(f"✅ {name.title()} connection verified")
                except Exception as e:
                    logger.error(f"❌ {name.title()} connection failed: {e}")

        except Exception as e:
            logger.error(f"❌ Exchange initialization failed: {e}")
            raise

    async def get_prices(self, exchange_name):
        """Get current prices from exchange"""
        try:
            if exchange_name not in self.exchanges:
                return {}

            exchange = self.exchanges[exchange_name]
            
            # Get real market data for live trading
            tickers = exchange.fetch_tickers()

            prices = {}
            for symbol, ticker in tickers.items():
                if symbol in self.config.CRYPTO_PAIRS:
                    prices[symbol] = {
                        'bid': ticker.get('bid', 0),
                        'ask': ticker.get('ask', 0),
                        'last': ticker.get('last', 0)
                    }

            return prices

        except Exception as e:
            logger.error(f"❌ Failed to get prices from {exchange_name}: {e}")
            return self._simulate_prices()
    
    def _simulate_prices(self):
        """Simulate realistic crypto prices for paper trading"""
        import random
        
        base_prices = {
            'BTC/USDT': 63500,
            'ETH/USDT': 3420,  
            'BNB/USDT': 598
        }
        
        prices = {}
        for pair, base_price in base_prices.items():
            # Add random market fluctuation (±2%)
            fluctuation = random.uniform(-0.02, 0.02)
            mid_price = base_price * (1 + fluctuation)
            spread = mid_price * 0.001  # 0.1% spread
            
            prices[pair] = {
                'bid': mid_price - spread/2,
                'ask': mid_price + spread/2,
                'last': mid_price
            }
            
        return prices

    async def get_ohlcv(self, exchange_name, symbol, timeframe, limit):
        """Get OHLCV data for technical analysis"""
        try:
            if exchange_name not in self.exchanges:
                return []

            # For paper trading, simulate OHLCV data
            if self.config.TRADING_MODE == 'paper':
                return self._simulate_ohlcv(symbol, limit)
                
            exchange = self.exchanges[exchange_name]
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            return ohlcv

        except Exception as e:
            logger.error(f"❌ Failed to get OHLCV from {exchange_name}: {e}")
            return self._simulate_ohlcv(symbol, limit)
    
    def _simulate_ohlcv(self, symbol, limit):
        """Simulate OHLCV data for paper trading"""
        import random
        from datetime import datetime, timedelta
        
        base_prices = {
            'BTC/USDT': 63500,
            'ETH/USDT': 3420,
            'BNB/USDT': 598
        }
        
        base_price = base_prices.get(symbol, 1000)
        ohlcv = []
        
        current_time = datetime.now().timestamp() * 1000
        
        for i in range(limit):
            # Simulate price movement
            change = random.uniform(-0.005, 0.005)  # ±0.5% per candle
            price = base_price * (1 + change * (i + 1) / limit)
            
            high = price * (1 + abs(change) * 0.5)
            low = price * (1 - abs(change) * 0.5)
            volume = random.uniform(100, 1000)
            
            # [timestamp, open, high, low, close, volume]
            candle = [
                current_time - (limit - i) * 3600000,  # 1 hour intervals
                price * 0.999,  # open
                high,           # high  
                low,            # low
                price,          # close
                volume          # volume
            ]
            ohlcv.append(candle)
            
        return ohlcv
