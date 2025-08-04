
import asyncio
import logging
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    print("⚠️ pandas not available - momentum strategy will use simplified logic")
    pd = None

logger = logging.getLogger(__name__)

class MomentumStrategy:
    def __init__(self):
        self.name = "momentum"
        self.exchange_manager = None
        self.risk_manager = None
        self.lookback_periods = 20
        
    def initialize(self, exchange_manager, risk_manager):
        """Initialize strategy"""
        self.exchange_manager = exchange_manager
        self.risk_manager = risk_manager
        logger.info("✅ Momentum strategy initialized")
    
    async def generate_signals(self):
        """Generate momentum trading signals"""
        signals = []
        
        try:
            # Get historical data for momentum analysis
            for pair in ['BTC/USDT', 'ETH/USDT']:
                ohlcv_data = await self.exchange_manager.get_ohlcv('binance', pair, '1h', 50)
                
                if len(ohlcv_data) >= self.lookback_periods:
                    if pd is not None:
                        signals.extend(self._analyze_with_pandas(pair, ohlcv_data))
                    else:
                        signals.extend(self._analyze_simple(pair, ohlcv_data))
            
        except Exception as e:
            logger.error(f"❌ Momentum signal generation failed: {e}")
        
        return signals
    
    def _analyze_with_pandas(self, pair, ohlcv_data):
        """Analyze momentum using pandas"""
        signals = []
        
        try:
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate moving averages
            df['ma_fast'] = df['close'].rolling(10).mean()
            df['ma_slow'] = df['close'].rolling(20).mean()
            
            # Get latest values
            current_price = df['close'].iloc[-1]
            ma_fast = df['ma_fast'].iloc[-1]
            ma_slow = df['ma_slow'].iloc[-1]
            prev_ma_fast = df['ma_fast'].iloc[-2]
            prev_ma_slow = df['ma_slow'].iloc[-2]
            
            # Bullish momentum signal
            if (ma_fast > ma_slow and prev_ma_fast <= prev_ma_slow and 
                current_price > ma_fast):
                
                confidence = min(0.85, abs(ma_fast - ma_slow) / ma_slow * 10)
                
                signals.append({
                    'pair': pair,
                    'action': 'buy',
                    'price': current_price,
                    'target': current_price * 1.03,  # 3% target
                    'stop_loss': current_price * 0.98,  # 2% stop
                    'confidence': confidence,
                    'strategy': self.name,
                    'timestamp': datetime.now()
                })
                
                logger.info(f"📈 MOMENTUM BUY: {pair} - Confidence: {confidence:.2f}")
            
            # Bearish momentum signal
            elif (ma_fast < ma_slow and prev_ma_fast >= prev_ma_slow and 
                  current_price < ma_fast):
                
                confidence = min(0.85, abs(ma_fast - ma_slow) / ma_slow * 10)
                
                signals.append({
                    'pair': pair,
                    'action': 'sell',
                    'price': current_price,
                    'target': current_price * 0.97,  # 3% target
                    'stop_loss': current_price * 1.02,  # 2% stop
                    'confidence': confidence,
                    'strategy': self.name,
                    'timestamp': datetime.now()
                })
                
                logger.info(f"📉 MOMENTUM SELL: {pair} - Confidence: {confidence:.2f}")
        
        except Exception as e:
            logger.error(f"❌ Pandas momentum analysis failed: {e}")
        
        return signals
    
    def _analyze_simple(self, pair, ohlcv_data):
        """Simple momentum analysis without pandas"""
        signals = []
        
        try:
            if len(ohlcv_data) < 20:
                return signals
            
            # Get recent closing prices
            recent_prices = [candle[4] for candle in ohlcv_data[-20:]]  # Last 20 close prices
            current_price = recent_prices[-1]
            
            # Simple moving averages
            ma_fast = sum(recent_prices[-10:]) / 10  # 10-period MA
            ma_slow = sum(recent_prices) / 20       # 20-period MA
            
            # Previous MAs
            prev_prices = [candle[4] for candle in ohlcv_data[-21:-1]]
            prev_ma_fast = sum(prev_prices[-10:]) / 10
            prev_ma_slow = sum(prev_prices) / 20
            
            # Check for momentum signals
            if ma_fast > ma_slow and prev_ma_fast <= prev_ma_slow:
                confidence = min(0.8, abs(ma_fast - ma_slow) / ma_slow * 10)
                
                signals.append({
                    'pair': pair,
                    'action': 'buy',
                    'price': current_price,
                    'target': current_price * 1.02,  # 2% target
                    'stop_loss': current_price * 0.99,  # 1% stop
                    'confidence': confidence,
                    'strategy': self.name,
                    'timestamp': datetime.now()
                })
                
                logger.info(f"📈 SIMPLE MOMENTUM BUY: {pair} - Confidence: {confidence:.2f}")
        
        except Exception as e:
            logger.error(f"❌ Simple momentum analysis failed: {e}")
        
        return signals
