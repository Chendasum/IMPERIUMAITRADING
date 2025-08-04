import os
from dotenv import load_dotenv

load_dotenv()


class TradingConfig:
    # API Configuration
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
    COINBASE_API_KEY = os.getenv('COINBASE_API_KEY')
    COINBASE_SECRET_KEY = os.getenv('COINBASE_SECRET_KEY')

    # Telegram Configuration  
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', os.getenv('ADMIN_CHAT_ID'))

    # Trading Parameters
    TRADING_MODE = 'LIVE'  # REAL TRADING ONLY - NO SIMULATION
    INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 10000))
    MAX_RISK_PER_TRADE = float(os.getenv('MAX_RISK_PER_TRADE', 0.02))
    DAILY_LOSS_LIMIT = float(os.getenv('DAILY_LOSS_LIMIT', 0.05))

    # Strategy Settings
    ARBITRAGE_MIN_PROFIT = float(os.getenv('ARBITRAGE_MIN_PROFIT', 0.01))
    MOMENTUM_TIMEFRAME = os.getenv('MOMENTUM_TIMEFRAME', '1h')
    MEAN_REVERSION_PERIODS = int(os.getenv('MEAN_REVERSION_PERIODS', 20))

    # MetaTrader/Forex settings
    METAAPI_TOKEN = os.getenv('METAAPI_TOKEN')
    
    # Supported trading pairs
    CRYPTO_PAIRS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    FOREX_PAIRS = ['EUR/USD', 'GBP/USD', 'USD/JPY']
