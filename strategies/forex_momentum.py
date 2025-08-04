import asyncio
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ForexMomentumStrategy:
    def __init__(self):
        self.name = "forex_momentum"
        self.metatrader_manager = None
        self.risk_manager = None
        self.timeframe = '1h'
        self.lookback_periods = 20
        self.momentum_threshold = 0.0015  # 15 pips for major pairs
        
    def initialize(self, metatrader_manager, risk_manager):
        """Initialize forex momentum strategy"""
        self.metatrader_manager = metatrader_manager
        self.risk_manager = risk_manager
        logger.info("✅ Forex momentum strategy initialized")
    
    async def generate_signals(self):
        """Generate forex momentum trading signals"""
        signals = []
        
        try:
            # Get major forex pairs
            forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
            
            for pair in forex_pairs:
                # Get historical data for analysis
                if self.metatrader_manager:
                    ohlcv = await self.metatrader_manager.get_forex_ohlcv(
                        pair, self.timeframe, self.lookback_periods + 5
                    )
                else:
                    logger.warning("❌ MetaTrader manager not initialized")
                    continue
                
                if len(ohlcv) >= self.lookback_periods:
                    signal = await self._analyze_forex_momentum(pair, ohlcv)
                    if signal:
                        signals.append(signal)
                        logger.info(f"💱 Forex momentum signal: {pair} - {signal['action']} - {signal['confidence']:.2f} confidence")
                        
        except Exception as e:
            logger.error(f"❌ Forex momentum signal generation failed: {e}")
            
        return signals
    
    async def _analyze_forex_momentum(self, pair, ohlcv):
        """Analyze forex momentum for a trading pair"""
        try:
            # Extract closing prices
            closes = np.array([candle[4] for candle in ohlcv])  # Close price is index 4
            
            if len(closes) < self.lookback_periods:
                return None
            
            # Calculate forex-specific momentum indicators
            current_price = closes[-1]
            
            # Short-term and long-term moving averages
            sma_fast = np.mean(closes[-10:])  # 10-period fast SMA
            sma_slow = np.mean(closes[-self.lookback_periods:])  # 20-period slow SMA
            
            # Calculate price momentum over different periods
            momentum_5 = (current_price - closes[-6]) / closes[-6] if len(closes) > 5 else 0
            momentum_10 = (current_price - closes[-11]) / closes[-11] if len(closes) > 10 else 0
            
            # Calculate Average True Range for volatility
            atr = self._calculate_atr(ohlcv[-14:]) if len(ohlcv) >= 14 else 0.001
            
            # Forex-specific RSI calculation
            rsi = self._calculate_rsi(closes)
            
            # Generate signals based on multiple confirmations
            if (sma_fast > sma_slow and 
                momentum_5 > self.momentum_threshold and 
                momentum_10 > self.momentum_threshold/2 and
                rsi < 70):  # Not overbought
                
                # Bullish momentum signal
                confidence = min(0.85, abs(momentum_5) * 100)
                target_pips = atr * 2  # 2x ATR target
                
                return {
                    'pair': pair,
                    'action': 'buy',
                    'price': current_price,
                    'target': current_price + target_pips,
                    'stop_loss': current_price - (atr * 1.5),
                    'confidence': confidence,
                    'strategy': self.name,
                    'momentum_5': momentum_5,
                    'momentum_10': momentum_10,
                    'rsi': rsi,
                    'atr': atr,
                    'timestamp': datetime.now()
                }
                
            elif (sma_fast < sma_slow and 
                  momentum_5 < -self.momentum_threshold and 
                  momentum_10 < -self.momentum_threshold/2 and
                  rsi > 30):  # Not oversold
                
                # Bearish momentum signal
                confidence = min(0.85, abs(momentum_5) * 100)
                target_pips = atr * 2  # 2x ATR target
                
                return {
                    'pair': pair,
                    'action': 'sell',
                    'price': current_price,
                    'target': current_price - target_pips,
                    'stop_loss': current_price + (atr * 1.5),
                    'confidence': confidence,
                    'strategy': self.name,
                    'momentum_5': momentum_5,
                    'momentum_10': momentum_10,
                    'rsi': rsi,
                    'atr': atr,
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Forex momentum analysis failed for {pair}: {e}")
            return None
    
    def _calculate_atr(self, ohlcv_data):
        """Calculate Average True Range for forex volatility"""
        try:
            if len(ohlcv_data) < 2:
                return 0.001
            
            true_ranges = []
            for i in range(1, len(ohlcv_data)):
                high = ohlcv_data[i][2]
                low = ohlcv_data[i][3]
                prev_close = ohlcv_data[i-1][4]
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
            
            return np.mean(true_ranges) if true_ranges else 0.001
            
        except Exception as e:
            logger.error(f"❌ ATR calculation failed: {e}")
            return 0.001
    
    def _calculate_rsi(self, prices, periods=14):
        """Calculate RSI for forex pairs"""
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
