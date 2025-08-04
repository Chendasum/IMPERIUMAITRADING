
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MeanReversionStrategy:
    def __init__(self):
        self.name = "mean_reversion"
        self.exchange_manager = None
        self.risk_manager = None
        self.lookback_periods = 20
        self.deviation_threshold = 2.0  # 2 standard deviations
        
    def initialize(self, exchange_manager, risk_manager):
        """Initialize strategy"""
        self.exchange_manager = exchange_manager
        self.risk_manager = risk_manager
        logger.info("✅ Mean reversion strategy initialized")
    
    async def generate_signals(self):
        """Generate mean reversion trading signals"""
        signals = []
        
        try:
            # Get historical data for mean reversion analysis
            for pair in ['BTC/USDT', 'ETH/USDT']:
                ohlcv_data = await self.exchange_manager.get_ohlcv('binance', pair, '1h', 50)
                
                if len(ohlcv_data) >= self.lookback_periods:
                    signals.extend(self._analyze_mean_reversion(pair, ohlcv_data))
            
        except Exception as e:
            logger.error(f"❌ Mean reversion signal generation failed: {e}")
        
        return signals
    
    def _analyze_mean_reversion(self, pair, ohlcv_data):
        """Analyze mean reversion opportunities"""
        signals = []
        
        try:
            # Get recent closing prices
            recent_prices = [candle[4] for candle in ohlcv_data[-self.lookback_periods:]]
            current_price = recent_prices[-1]
            
            # Calculate mean and standard deviation
            mean_price = sum(recent_prices) / len(recent_prices)
            
            # Calculate standard deviation manually
            variance = sum((price - mean_price) ** 2 for price in recent_prices) / len(recent_prices)
            std_dev = variance ** 0.5
            
            # Calculate Z-score (how many standard deviations from mean)
            z_score = (current_price - mean_price) / std_dev if std_dev > 0 else 0
            
            # Generate signals based on extreme deviations
            if z_score > self.deviation_threshold:
                # Price is too high - expect reversion down
                confidence = min(0.9, abs(z_score) / self.deviation_threshold * 0.5)
                
                signals.append({
                    'pair': pair,
                    'action': 'sell',
                    'price': current_price,
                    'target': mean_price,  # Target is the mean
                    'stop_loss': current_price * 1.015,  # 1.5% stop
                    'confidence': confidence,
                    'strategy': self.name,
                    'z_score': z_score,
                    'timestamp': datetime.now()
                })
                
                logger.info(f"📉 MEAN REVERSION SELL: {pair} - Z-score: {z_score:.2f}")
                
            elif z_score < -self.deviation_threshold:
                # Price is too low - expect reversion up
                confidence = min(0.9, abs(z_score) / self.deviation_threshold * 0.5)
                
                signals.append({
                    'pair': pair,
                    'action': 'buy',
                    'price': current_price,
                    'target': mean_price,  # Target is the mean
                    'stop_loss': current_price * 0.985,  # 1.5% stop
                    'confidence': confidence,
                    'strategy': self.name,
                    'z_score': z_score,
                    'timestamp': datetime.now()
                })
                
                logger.info(f"📈 MEAN REVERSION BUY: {pair} - Z-score: {z_score:.2f}")
        
        except Exception as e:
            logger.error(f"❌ Mean reversion analysis failed: {e}")
        
        return signals
