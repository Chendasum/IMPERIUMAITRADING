import asyncio
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    def __init__(self):
        self.name = "mean_reversion"
        self.exchange_manager = None
        self.risk_manager = None
        self.periods = 20
        self.std_multiplier = 2.0
        self.timeframe = '1h'
        
    def initialize(self, exchange_manager, risk_manager):
        """Initialize strategy"""
        self.exchange_manager = exchange_manager
        self.risk_manager = risk_manager
        logger.info("✅ Mean reversion strategy initialized")
    
    async def generate_signals(self):
        """Generate mean reversion trading signals"""
        signals = []
        
        try:
            pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
            
            for pair in pairs:
                # Get historical data
                exchange_name = list(self.exchange_manager.exchanges.keys())[0] if self.exchange_manager.exchanges else 'binance'
                ohlcv = await self.exchange_manager.get_ohlcv(
                    exchange_name, pair, self.timeframe, self.periods + 5
                )
                
                if len(ohlcv) >= self.periods:
                    signal = await self._analyze_mean_reversion(pair, ohlcv)
                    if signal:
                        signals.append(signal)
                        
        except Exception as e:
            logger.error(f"❌ Mean reversion signal generation failed: {e}")
            
        return signals
    
    async def _analyze_mean_reversion(self, pair, ohlcv):
        """Analyze mean reversion for a trading pair"""
        try:
            # Extract closing prices
            closes = np.array([candle[4] for candle in ohlcv])
            
            if len(closes) < self.periods:
                return None
            
            # Calculate Bollinger Bands
            sma = np.mean(closes[-self.periods:])
            std = np.std(closes[-self.periods:])
            upper_band = sma + (std * self.std_multiplier)
            lower_band = sma - (std * self.std_multiplier)
            
            current_price = closes[-1]
            
            # Calculate distance from mean
            distance_from_mean = abs(current_price - sma) / sma
            
            # Generate signals based on Bollinger Band touches
            if current_price <= lower_band:
                # Price at lower band - potential buy signal (oversold)
                confidence = min(0.8, distance_from_mean * 5)
                return {
                    'pair': pair,
                    'action': 'buy',
                    'price': current_price,
                    'target': sma,  # Target is the mean
                    'confidence': confidence,
                    'strategy': self.name,
                    'mean': sma,
                    'distance_from_mean': distance_from_mean,
                    'bb_position': 'lower',
                    'timestamp': datetime.now()
                }
                
            elif current_price >= upper_band:
                # Price at upper band - potential sell signal (overbought)
                confidence = min(0.8, distance_from_mean * 5)
                return {
                    'pair': pair,
                    'action': 'sell',
                    'price': current_price,
                    'target': sma,  # Target is the mean
                    'confidence': confidence,
                    'strategy': self.name,
                    'mean': sma,
                    'distance_from_mean': distance_from_mean,
                    'bb_position': 'upper',
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Mean reversion analysis failed for {pair}: {e}")
            return None
    
    def _calculate_bollinger_bands(self, prices, periods=20, std_multiplier=2):
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < periods:
                return None, None, None
            
            sma = np.mean(prices[-periods:])
            std = np.std(prices[-periods:])
            
            upper_band = sma + (std * std_multiplier)
            lower_band = sma - (std * std_multiplier)
            
            return upper_band, sma, lower_band
            
        except Exception as e:
            logger.error(f"❌ Bollinger Bands calculation failed: {e}")
            return None, None, None
    
    def _calculate_mean_reversion_score(self, current_price, mean, std):
        """Calculate mean reversion score"""
        try:
            if std == 0:
                return 0
            
            z_score = (current_price - mean) / std
            
            # Convert z-score to reversion probability
            reversion_score = 1 / (1 + abs(z_score))
            
            return reversion_score
            
        except Exception as e:
            logger.error(f"❌ Mean reversion score calculation failed: {e}")
            return 0
