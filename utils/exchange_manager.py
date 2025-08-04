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
            # Initialize Binance
            if self.config.BINANCE_API_KEY:
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey':
                    self.config.BINANCE_API_KEY,
                    'secret':
                    self.config.BINANCE_SECRET_KEY,
                    'sandbox':
                    self.config.TRADING_MODE == 'paper',
                    'enableRateLimit':
                    True,
                })
                logger.info("✅ Binance connection established")

            # Initialize Coinbase (if available)
            if self.config.COINBASE_API_KEY:
                self.exchanges['coinbase'] = ccxt.coinbasepro({
                    'apiKey':
                    self.config.COINBASE_API_KEY,
                    'secret':
                    self.config.COINBASE_SECRET_KEY,
                    'sandbox':
                    self.config.TRADING_MODE == 'paper',
                    'enableRateLimit':
                    True,
                })
                logger.info("✅ Coinbase connection established")

            # Test connections
            for name, exchange in self.exchanges.items():
                try:
                    await exchange.load_markets()
                    balance = await exchange.fetch_balance()
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
            tickers = await exchange.fetch_tickers()

            prices = {}
            for symbol, ticker in tickers.items():
                if symbol in self.config.CRYPTO_PAIRS:
                    prices[symbol] = {
                        'bid': ticker['bid'],
                        'ask': ticker['ask'],
                        'last': ticker['last']
                    }

            return prices

        except Exception as e:
            logger.error(f"❌ Failed to get prices from {exchange_name}: {e}")
            return {}

    async def get_ohlcv(self, exchange_name, symbol, timeframe, limit):
        """Get OHLCV data for technical analysis"""
        try:
            if exchange_name not in self.exchanges:
                return []

            exchange = self.exchanges[exchange_name]
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            return ohlcv

        except Exception as e:
            logger.error(f"❌ Failed to get OHLCV from {exchange_name}: {e}")
            return []
