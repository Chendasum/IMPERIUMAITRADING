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
                    'sandbox': False,  # LIVE TRADING ENVIRONMENT ONLY
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
                # LIVE TRADING ONLY - NO SIMULATION MODE
                logger.error("❌ LIVE TRADING FAILED - NO SIMULATION FALLBACK")

            # LIVE TRADING - Multi-exchange support
            # Ready for real arbitrage trading
            logger.info("🚀 LIVE TRADING MODE - Real exchange connections")

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
    
    async def _get_real_crypto_data_alternatives(self):
        """Get real crypto data from authenticated APIs"""
        prices = {}
        
        # Try Alpha Vantage for crypto data
        try:
            from .alpha_vantage_client import AlphaVantageClient
            alpha_client = AlphaVantageClient()
            
            crypto_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP']
            
            for crypto in crypto_symbols:
                symbol_key = f"{crypto}/USDT"
                if symbol_key in self.config.CRYPTO_PAIRS:
                    crypto_data = await alpha_client.get_crypto_intraday(crypto, 'USD')
                    if crypto_data:
                        prices[symbol_key] = crypto_data
            
            if prices:
                logger.info(f"✅ Retrieved real crypto data from Alpha Vantage for {len(prices)} pairs")
                return prices
                
        except Exception as e:
            logger.error(f"❌ Alpha Vantage crypto failed: {e}")
        
        # Try Finnhub for crypto quotes
        try:
            from .finnhub_client import FinnhubClient
            finnhub_client = FinnhubClient()
            
            for symbol in self.config.CRYPTO_PAIRS:
                crypto_data = await finnhub_client.get_crypto_quote(symbol)
                if crypto_data:
                    prices[symbol] = crypto_data
            
            if prices:
                logger.info(f"✅ Retrieved real crypto data from Finnhub for {len(prices)} pairs")
                return prices
                
        except Exception as e:
            logger.error(f"❌ Finnhub crypto failed: {e}")
        
        # Use CoinGecko free API as primary crypto data source
        try:
            from .free_forex_client import CoinGeckoClient
            coingecko_client = CoinGeckoClient()
            
            crypto_prices = await coingecko_client.get_crypto_prices()
            if crypto_prices:
                logger.info(f"✅ Retrieved real crypto prices from CoinGecko for {len(crypto_prices)} pairs")
                return crypto_prices
                
        except Exception as e:
            logger.error(f"❌ CoinGecko crypto data failed: {e}")
        
        logger.error("❌ No authentic crypto data sources available")
        return {}

    async def get_prices(self, exchange_name):
        """Get current prices from exchange"""
        try:
            if exchange_name not in self.exchanges:
                return {}

            exchange = self.exchanges[exchange_name]
            
            # Get real market data for live trading
            # If restricted location, use fallback data
            try:
                tickers = exchange.fetch_tickers()
            except Exception as e:
                if "restricted location" in str(e):
                    logger.warning("⚠️ Location restricted - trying real alternative crypto data sources")
                    return await self._get_real_crypto_data_alternatives()
                raise

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
            return {}  # NO SIMULATION - REAL PRICES ONLY
    
    # SIMULATION FUNCTIONS REMOVED - REAL TRADING ONLY

    async def get_ohlcv(self, exchange_name, symbol, timeframe, limit):
        """Get OHLCV data for technical analysis"""
        try:
            if exchange_name not in self.exchanges:
                return []

            # LIVE TRADING ONLY - Real OHLCV data
            exchange = self.exchanges[exchange_name]
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            return ohlcv

        except Exception as e:
            logger.error(f"❌ Failed to get OHLCV from {exchange_name}: {e}")
            return []  # NO SIMULATION - REAL DATA ONLY
    
    # ALL SIMULATION FUNCTIONS REMOVED - LIVE TRADING ONLY
