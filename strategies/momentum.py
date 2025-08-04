import asyncio
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class MomentumStrategy:
    def __init__(self):
        self.name = "momentum"
        self.exchange_manager = None
        self.risk_manager = None
        self.timeframe = '1h'
        self.lookback_periods = 14
        self.momentum_threshold = 0.02  # 2% momentum threshold
        
    def initialize(self, exchange_manager, risk_manager):
        """Initialize strategy"""
        self.exchange_manager = exchange_manager
        self.risk_manager = risk_manager
        logger.info("✅ Momentum strategy initialized")
    
    async def generate_signals(self):
        """Generate momentum trading signals"""
        signals = []
        
        try:
            # Get OHLCV data for analysis
            pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
            
            for pair in pairs:
                # Get historical data from primary exchange
                exchange_name = list(self.exchange_manager.exchanges.keys())[0] if self.exchange_manager.exchanges else 'binance'
                ohlcv = await self.exchange_manager.get_ohlcv(
                    exchange_name, pair, self.timeframe, self.lookback_periods + 5
                )
                
                if len(ohlcv) >= self.lookback_periods:
                    signal = await self._analyze_momentum(pair, ohlcv)
                    if signal:
                        signals.append(signal)
                        
        except Exception as e:
            logger.error(f"❌ Momentum signal generation failed: {e}")
            
        return signals
    
    async def _analyze_momentum(self, pair, ohlcv):
        """Analyze momentum for a trading pair"""
        try:
            # Extract closing prices
            closes = np.array([candle[4] for candle in ohlcv])  # Close price is index 4
            
            if len(closes) < self.lookback_periods:
                return None
            
            # Calculate simple momentum indicators
            current_price = closes[-1]
            sma_short = np.mean(closes[-5:])  # 5-period SMA
            sma_long = np.mean(closes[-self.lookback_periods:])  # 14-period SMA
            
            # Calculate price momentum
            price_momentum = (current_price - closes[-self.lookback_periods]) / closes[-self.lookback_periods]
            
            # Calculate moving average momentum
            ma_momentum = (sma_short - sma_long) / sma_long
            
            # Determine signal direction and strength
            if price_momentum > self.momentum_threshold and ma_momentum > 0:
                # Strong bullish momentum
                confidence = min(0.85, abs(price_momentum) * 10)
                return {
                    'pair': pair,
                    'action': 'buy',
                    'price': current_price,
                    'target': current_price * 1.05,  # 5% target
                    'confidence': confidence,
                    'strategy': self.name,
                    'momentum': price_momentum,
                    'ma_signal': ma_momentum,
                    'timestamp': datetime.now()
                }
                
            elif price_momentum < -self.momentum_threshold and ma_momentum < 0:
                # Strong bearish momentum
                confidence = min(0.85, abs(price_momentum) * 10)
                return {
                    'pair': pair,
                    'action': 'sell',
                    'price': current_price,
                    'target': current_price * 0.95,  # 5% target
                    'confidence': confidence,
                    'strategy': self.name,
                    'momentum': price_momentum,
                    'ma_signal': ma_momentum,
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Momentum analysis failed for {pair}: {e}")
            return None
    
    def _calculate_rsi(self, prices, periods=14):
        """Calculate RSI indicator (simplified version)"""
        try:
            if len(prices) < periods + 1:
                return 50  # Neutral RSI
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-periods:])
            avg_loss = np.mean(losses[-periods:])
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"❌ RSI calculation failed: {e}")
            return 50