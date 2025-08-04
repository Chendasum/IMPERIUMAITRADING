import os
from dotenv import load_dotenv
load_dotenv()

class TradingConfig:
    """Enhanced Trading Configuration with Professional Features"""
    
    # =============================================================================
    # API CONFIGURATION
    # =============================================================================
    
    # Binance Configuration
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
    BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    
    # Coinbase Configuration  
    COINBASE_API_KEY = os.getenv('COINBASE_API_KEY')
    COINBASE_SECRET_KEY = os.getenv('COINBASE_SECRET_KEY')
    COINBASE_PASSPHRASE = os.getenv('COINBASE_PASSPHRASE')
    
    # ByBit Configuration (if needed later)
    BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
    BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')
    
    # MetaAPI/Forex Configuration
    METAAPI_TOKEN = os.getenv('METAAPI_TOKEN')
    METAAPI_ACCOUNT_ID = os.getenv('METAAPI_ACCOUNT_ID')
    
    # Additional API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    CURRENCY_LAYER_API_KEY = os.getenv('CURRENCY_LAYER_API_KEY')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
    METALS_API_KEY = os.getenv('METALS_API_KEY')
    
    # =============================================================================
    # NOTIFICATION CONFIGURATION
    # =============================================================================
    
    # Telegram Configuration  
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', os.getenv('ADMIN_CHAT_ID'))
    
    # Notification Settings
    NOTIFY_ON_TRADES = os.getenv('NOTIFY_ON_TRADES', 'true').lower() == 'true'
    NOTIFY_ON_ERRORS = os.getenv('NOTIFY_ON_ERRORS', 'true').lower() == 'true'
    NOTIFY_PROFIT_THRESHOLD = float(os.getenv('NOTIFY_PROFIT_THRESHOLD', 50.0))
    
    # =============================================================================
    # TRADING CONFIGURATION
    # =============================================================================
    
    # Core Trading Settings
    TRADING_MODE = 'LIVE'  # REAL TRADING ONLY - NO SIMULATION
    INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 10000))
    
    # Enhanced Risk Management
    MAX_RISK_PER_TRADE = float(os.getenv('MAX_RISK_PER_TRADE', 0.01))  # 1% (more conservative)
    DAILY_LOSS_LIMIT = float(os.getenv('DAILY_LOSS_LIMIT', 0.03))      # 3% (more conservative)
    MAX_DRAWDOWN_LIMIT = float(os.getenv('MAX_DRAWDOWN_LIMIT', 0.15))  # 15% max drawdown
    MAX_PORTFOLIO_RISK = float(os.getenv('MAX_PORTFOLIO_RISK', 0.06))  # 6% total portfolio risk
    
    # Position Management
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 10))
    MAX_CORRELATION = float(os.getenv('MAX_CORRELATION', 0.7))
    MAX_SECTOR_EXPOSURE = float(os.getenv('MAX_SECTOR_EXPOSURE', 0.25))
    MAX_CONSECUTIVE_LOSSES = int(os.getenv('MAX_CONSECUTIVE_LOSSES', 5))
    
    # =============================================================================
    # STRATEGY CONFIGURATION
    # =============================================================================
    
    # Strategy Weights (should sum to 1.0)
    STRATEGY_WEIGHTS = {
        'arbitrage': float(os.getenv('ARBITRAGE_WEIGHT', 0.4)),
        'momentum': float(os.getenv('MOMENTUM_WEIGHT', 0.3)),
        'mean_reversion': float(os.getenv('MEAN_REVERSION_WEIGHT', 0.2)),
        'forex_momentum': float(os.getenv('FOREX_MOMENTUM_WEIGHT', 0.1))
    }
    
    # Arbitrage Strategy
    ARBITRAGE_MIN_PROFIT = float(os.getenv('ARBITRAGE_MIN_PROFIT', 0.008))  # 0.8% minimum
    ARBITRAGE_MAX_POSITIONS = int(os.getenv('ARBITRAGE_MAX_POSITIONS', 3))
    ARBITRAGE_MIN_CONFIDENCE = float(os.getenv('ARBITRAGE_MIN_CONFIDENCE', 0.8))
    
    # Momentum Strategy
    MOMENTUM_TIMEFRAME = os.getenv('MOMENTUM_TIMEFRAME', '1h')
    MOMENTUM_RSI_PERIOD = int(os.getenv('MOMENTUM_RSI_PERIOD', 14))
    MOMENTUM_RSI_OVERBOUGHT = float(os.getenv('MOMENTUM_RSI_OVERBOUGHT', 70))
    MOMENTUM_RSI_OVERSOLD = float(os.getenv('MOMENTUM_RSI_OVERSOLD', 30))
    MOMENTUM_MAX_POSITIONS = int(os.getenv('MOMENTUM_MAX_POSITIONS', 2))
    MOMENTUM_MIN_CONFIDENCE = float(os.getenv('MOMENTUM_MIN_CONFIDENCE', 0.7))
    
    # Mean Reversion Strategy
    MEAN_REVERSION_PERIODS = int(os.getenv('MEAN_REVERSION_PERIODS', 20))
    MEAN_REVERSION_STD_DEV = float(os.getenv('MEAN_REVERSION_STD_DEV', 2.0))
    MEAN_REVERSION_MAX_POSITIONS = int(os.getenv('MEAN_REVERSION_MAX_POSITIONS', 2))
    MEAN_REVERSION_MIN_CONFIDENCE = float(os.getenv('MEAN_REVERSION_MIN_CONFIDENCE', 0.75))
    
    # Forex Strategy
    FOREX_MOMENTUM_TIMEFRAME = os.getenv('FOREX_MOMENTUM_TIMEFRAME', '1h')
    FOREX_MAX_POSITIONS = int(os.getenv('FOREX_MAX_POSITIONS', 3))
    FOREX_MIN_CONFIDENCE = float(os.getenv('FOREX_MIN_CONFIDENCE', 0.7))
    FOREX_SPREAD_THRESHOLD = float(os.getenv('FOREX_SPREAD_THRESHOLD', 0.0005))  # 0.5 pips
    
    # =============================================================================
    # MARKET DATA CONFIGURATION
    # =============================================================================
    
    # Supported Trading Pairs
    CRYPTO_PAIRS = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT',
        'LINK/USDT', 'UNI/USDT', 'LTC/USDT', 'BCH/USDT', 'XRP/USDT'
    ]
    
    FOREX_PAIRS = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 
        'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP'
    ]
    
    # Data Sources Priority
    PRIMARY_DATA_SOURCE = os.getenv('PRIMARY_DATA_SOURCE', 'binance')
    BACKUP_DATA_SOURCES = ['coinbase', 'metaapi']
    
    # =============================================================================
    # SYSTEM CONFIGURATION
    # =============================================================================
    
    # Timing Settings
    TRADING_CYCLE_INTERVAL = int(os.getenv('TRADING_CYCLE_INTERVAL', 30))  # seconds
    MARKET_DATA_REFRESH = int(os.getenv('MARKET_DATA_REFRESH', 10))  # seconds
    PORTFOLIO_REBALANCE_INTERVAL = int(os.getenv('PORTFOLIO_REBALANCE_INTERVAL', 1500))  # 25 minutes
    
    # Performance Settings
    MAX_TRADES_PER_CYCLE = int(os.getenv('MAX_TRADES_PER_CYCLE', 5))
    MIN_TIME_BETWEEN_TRADES = int(os.getenv('MIN_TIME_BETWEEN_TRADES', 5))  # seconds
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_TO_FILE = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    LOG_FILE_PATH = os.getenv('LOG_FILE_PATH', 'logs/trading.log')
    
    # Database Configuration (if using)
    DATABASE_URL = os.getenv('DATABASE_URL')
    REDIS_URL = os.getenv('REDIS_URL')
    
    # =============================================================================
    # ADVANCED FEATURES
    # =============================================================================
    
    # Machine Learning Features
    USE_ML_PREDICTIONS = os.getenv('USE_ML_PREDICTIONS', 'false').lower() == 'true'
    ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', 'models/')
    ML_RETRAIN_INTERVAL = int(os.getenv('ML_RETRAIN_INTERVAL', 24))  # hours
    
    # Market Condition Analysis
    VOLATILITY_THRESHOLD_HIGH = float(os.getenv('VOLATILITY_THRESHOLD_HIGH', 0.04))
    VOLATILITY_THRESHOLD_LOW = float(os.getenv('VOLATILITY_THRESHOLD_LOW', 0.02))
    
    # Emergency Settings
    EMERGENCY_STOP_LOSS = float(os.getenv('EMERGENCY_STOP_LOSS', 0.20))  # 20%
    CIRCUIT_BREAKER_ENABLED = os.getenv('CIRCUIT_BREAKER_ENABLED', 'true').lower() == 'true'
    
    # =============================================================================
    # VALIDATION AND INITIALIZATION
    # =============================================================================
    
    @classmethod
    def validate_config(cls):
        """Validate critical configuration parameters"""
        errors = []
        
        # Check required API keys
        if not cls.BINANCE_API_KEY or not cls.BINANCE_SECRET_KEY:
            errors.append("Binance API credentials missing")
        
        if not cls.METAAPI_TOKEN:
            errors.append("MetaAPI token missing")
        
        if not cls.TELEGRAM_BOT_TOKEN or not cls.TELEGRAM_CHAT_ID:
            errors.append("Telegram configuration missing")
        
        # Validate risk parameters
        if cls.MAX_RISK_PER_TRADE > 0.05:  # 5%
            errors.append("MAX_RISK_PER_TRADE too high (>5%)")
        
        if cls.DAILY_LOSS_LIMIT > 0.10:  # 10%
            errors.append("DAILY_LOSS_LIMIT too high (>10%)")
        
        # Validate strategy weights
        total_weight = sum(cls.STRATEGY_WEIGHTS.values())
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Strategy weights don't sum to 1.0 (current: {total_weight})")
        
        return errors
    
    @classmethod
    def get_config_summary(cls):
        """Get configuration summary for logging"""
        return {
            'trading_mode': cls.TRADING_MODE,
            'initial_balance': cls.INITIAL_BALANCE,
            'risk_per_trade': f"{cls.MAX_RISK_PER_TRADE*100:.1f}%",
            'daily_loss_limit': f"{cls.DAILY_LOSS_LIMIT*100:.1f}%",
            'max_positions': cls.MAX_POSITIONS,
            'crypto_pairs': len(cls.CRYPTO_PAIRS),
            'forex_pairs': len(cls.FOREX_PAIRS),
            'strategy_weights': cls.STRATEGY_WEIGHTS,
            'cycle_interval': f"{cls.TRADING_CYCLE_INTERVAL}s"
        }
    
    @classmethod
    def is_production_ready(cls):
        """Check if configuration is ready for production trading"""
        errors = cls.validate_config()
        
        production_checks = [
            cls.TRADING_MODE == 'LIVE',
            cls.MAX_RISK_PER_TRADE <= 0.02,  # Max 2% risk
            cls.DAILY_LOSS_LIMIT <= 0.05,   # Max 5% daily loss
            cls.CIRCUIT_BREAKER_ENABLED,
            cls.NOTIFY_ON_ERRORS,
            bool(cls.BINANCE_API_KEY),
            bool(cls.METAAPI_TOKEN),
            bool(cls.TELEGRAM_BOT_TOKEN)
        ]
        
        return len(errors) == 0 and all(production_checks)

# Configuration validation on import
if __name__ == "__main__":
    config_errors = TradingConfig.validate_config()
    if config_errors:
        print("❌ Configuration Errors:")
        for error in config_errors:
            print(f"  • {error}")
    else:
        print("✅ Configuration validated successfully")
        
    print(f"🏛️ Production Ready: {TradingConfig.is_production_ready()}")
    
    import json
    print("📊 Configuration Summary:")
    print(json.dumps(TradingConfig.get_config_summary(), indent=2))
