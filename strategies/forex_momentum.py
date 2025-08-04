import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class ForexMomentumStrategy:
    def __init__(self):
        self.name = "forex_momentum"
        self.metatrader_manager = None
        self.risk_manager = None
        
        # Since we can't get historical data, we'll track prices ourselves
        self.price_history = {}  # Store recent prices for each pair
        self.max_history_length = 50  # Keep last 50 price points
        self.min_data_points = 10     # Need at least 10 points for analysis
        
        self.forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
        self.momentum_threshold = 0.002  # 0.2% threshold (20 pips for major pairs)
        
        # Price tracking for momentum calculation
        self.last_update = {}
        self.update_interval = 30  # Update every 30 seconds
        
    def initialize(self, metatrader_manager, risk_manager):
        """Initialize forex momentum strategy"""
        self.metatrader_manager = metatrader_manager
        self.risk_manager = risk_manager
        
        # Initialize price history for each pair
        for pair in self.forex_pairs:
            self.price_history[pair] = deque(maxlen=self.max_history_length)
            self.last_update[pair] = datetime.min
            
        logger.info("✅ Forex momentum strategy initialized (current price mode)")
        logger.info(f"📊 Tracking {len(self.forex_pairs)} forex pairs")
    
    async def generate_signals(self):
        """Generate forex momentum signals using current price tracking"""
        signals = []
        
        try:
            # Update price history for all pairs
            await self._update_price_history()
            
            # Analyze each pair for momentum signals
            for pair in self.forex_pairs:
                try:
                    signal = await self._analyze_current_price_momentum(pair)
                    if signal:
                        signals.append(signal)
                        logger.info(f"💱 Forex momentum signal: {pair} - {signal['action']} - {signal['confidence']:.2f} confidence")
                        
                except Exception as e:
                    logger.error(f"❌ Error analyzing {pair}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Forex momentum signal generation failed: {e}")
            
        return signals
    
    async def _update_price_history(self):
        """Update price history by fetching current prices"""
        try:
            for pair in self.forex_pairs:
                # Check if we need to update (rate limiting)
                now = datetime.now()
                if (now - self.last_update[pair]).total_seconds() < self.update_interval:
                    continue
                
                # Get current price (this works with your free API)
                current_price = await self._get_current_forex_price(pair)
                
                if current_price:
                    # Add to price history with timestamp
                    price_entry = {
                        'price': current_price,
                        'timestamp': now
                    }
                    self.price_history[pair].append(price_entry)
                    self.last_update[pair] = now
                    
                    logger.debug(f"📊 Updated price for {pair}: {current_price}")
                
                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"❌ Price history update failed: {e}")
    
    async def _get_current_forex_price(self, pair):
        """Get current forex price (this uses your working free API)"""
        try:
            if not self.metatrader_manager:
                return None
            
            # This calls your existing method that successfully gets current prices
            current_prices = await self.metatrader_manager.get_forex_prices()
            
            if pair in current_prices:
                price_data = current_prices[pair]
                # Use the 'last' price or calculate mid-price
                if 'last' in price_data and price_data['last'] > 0:
                    return price_data['last']
                elif 'bid' in price_data and 'ask' in price_data:
                    return (price_data['bid'] + price_data['ask']) / 2
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get current price for {pair}: {e}")
            return None
    
    async def _analyze_current_price_momentum(self, pair):
        """Analyze momentum using our collected price history"""
        try:
            price_history = self.price_history[pair]
            
            # Need minimum data points for analysis
            if len(price_history) < self.min_data_points:
                logger.debug(f"📊 Not enough data for {pair}: {len(price_history)}/{self.min_data_points}")
                return None
            
            # Extract prices and timestamps
            prices = [entry['price'] for entry in price_history]
            timestamps = [entry['timestamp'] for entry in price_history]
            
            current_price = prices[-1]
            
            # Calculate momentum indicators using available data
            
            # 1. Short-term momentum (last 5 vs previous 5)
            if len(prices) >= 10:
                recent_avg = np.mean(prices[-5:])
                previous_avg = np.mean(prices[-10:-5])
                short_momentum = (recent_avg - previous_avg) / previous_avg
            else:
                short_momentum = 0
            
            # 2. Overall trend (current vs oldest)
            long_momentum = (current_price - prices[0]) / prices[0]
            
            # 3. Price velocity (rate of change)
            if len(prices) >= 3:
                recent_change = (prices[-1] - prices[-3]) / prices[-3]
            else:
                recent_change = 0
            
            # 4. Volatility assessment
            if len(prices) >= 5:
                recent_volatility = np.std(prices[-5:]) / np.mean(prices[-5:])
            else:
                recent_volatility = 0
            
            # 5. Trend consistency
            trend_consistency = self._calculate_trend_consistency(prices)
            
            # Generate signal based on momentum conditions
            signal = self._evaluate_momentum_conditions(
                pair, current_price, short_momentum, long_momentum, 
                recent_change, recent_volatility, trend_consistency
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Momentum analysis failed for {pair}: {e}")
            return None
    
    def _calculate_trend_consistency(self, prices):
        """Calculate how consistent the trend is"""
        if len(prices) < 5:
            return 0
        
        # Count directional changes
        changes = []
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                changes.append(1)  # Up
            elif prices[i] < prices[i-1]:
                changes.append(-1)  # Down
            else:
                changes.append(0)  # Flat
        
        if not changes:
            return 0
        
        # Calculate consistency (fewer direction changes = more consistent)
        direction_changes = 0
        for i in range(1, len(changes)):
            if changes[i] != changes[i-1] and changes[i] != 0 and changes[i-1] != 0:
                direction_changes += 1
        
        # Consistency score (0-1, higher is more consistent)
        max_possible_changes = len(changes) - 1
        if max_possible_changes > 0:
            consistency = 1 - (direction_changes / max_possible_changes)
        else:
            consistency = 0.5
        
        return consistency
    
    def _evaluate_momentum_conditions(self, pair, current_price, short_momentum, 
                                    long_momentum, recent_change, volatility, consistency):
        """Evaluate all conditions and generate signal if criteria met"""
        try:
            # Get pair-specific threshold
            threshold = self.momentum_threshold
            
            # Bullish momentum conditions
            bullish_score = 0
            if short_momentum > threshold:
                bullish_score += 3
            if long_momentum > threshold * 2:  # Stronger long-term momentum
                bullish_score += 2  
            if recent_change > threshold * 0.5:
                bullish_score += 2
            if consistency > 0.6:  # Consistent trend
                bullish_score += 1
            if 0.001 < volatility < 0.01:  # Moderate volatility
                bullish_score += 1
            
            # Bearish momentum conditions  
            bearish_score = 0
            if short_momentum < -threshold:
                bearish_score += 3
            if long_momentum < -threshold * 2:
                bearish_score += 2
            if recent_change < -threshold * 0.5:
                bearish_score += 2
            if consistency > 0.6:
                bearish_score += 1
            if 0.001 < volatility < 0.01:
                bearish_score += 1
            
            # Generate signal if score is high enough
            min_score = 5  # Minimum score for signal generation
            
            if bullish_score >= min_score and bullish_score > bearish_score:
                confidence = min(0.85, bullish_score / 9)  # Max possible score is 9
                
                return {
                    'pair': pair,
                    'action': 'buy',
                    'price': current_price,
                    'target': current_price * (1 + threshold * 3),  # 3x threshold target
                    'stop_loss': current_price * (1 - threshold * 2),  # 2x threshold stop
                    'confidence': confidence,
                    'strategy': self.name,
                    'momentum_data': {
                        'short_momentum': short_momentum,
                        'long_momentum': long_momentum,
                        'recent_change': recent_change,
                        'volatility': volatility,
                        'consistency': consistency,
                        'bullish_score': bullish_score
                    },
                    'timestamp': datetime.now()
                }
            
            elif bearish_score >= min_score and bearish_score > bullish_score:
                confidence = min(0.85, bearish_score / 9)
                
                return {
                    'pair': pair,
                    'action': 'sell',
                    'price': current_price,
                    'target': current_price * (1 - threshold * 3),
                    'stop_loss': current_price * (1 + threshold * 2),
                    'confidence': confidence,
                    'strategy': self.name,
                    'momentum_data': {
                        'short_momentum': short_momentum,
                        'long_momentum': long_momentum,
                        'recent_change': recent_change,
                        'volatility': volatility,
                        'consistency': consistency,
                        'bearish_score': bearish_score
                    },
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Momentum evaluation failed: {e}")
            return None
    
    def get_price_history_summary(self):
        """Get summary of collected price history"""
        summary = {}
        
        for pair in self.forex_pairs:
            history = self.price_history[pair]
            if history:
                prices = [entry['price'] for entry in history]
                summary[pair] = {
                    'data_points': len(prices),
                    'current_price': prices[-1] if prices else 0,
                    'price_range': {
                        'min': min(prices) if prices else 0,
                        'max': max(prices) if prices else 0
                    },
                    'last_update': self.last_update[pair].isoformat() if pair in self.last_update else None
                }
            else:
                summary[pair] = {
                    'data_points': 0,
                    'current_price': 0,
                    'price_range': {'min': 0, 'max': 0},
                    'last_update': None
                }
        
        return summary
    
    def clear_history(self, pair=None):
        """Clear price history for debugging or reset"""
        if pair:
            self.price_history[pair].clear()
            logger.info(f"🔄 Cleared price history for {pair}")
        else:
            for p in self.forex_pairs:
                self.price_history[p].clear()
            logger.info("🔄 Cleared all price history")
    
    async def get_forex_ohlcv(self, symbol, timeframe='1h', limit=50):
        """Compatibility method - returns current price as single candle"""
        try:
            current_price = await self._get_current_forex_price(symbol)
            
            if current_price:
                # Create a single "candle" from current price
                timestamp = int(datetime.now().timestamp() * 1000)
                
                # OHLCV format: [timestamp, open, high, low, close, volume]
                ohlcv = [[
                    timestamp,
                    current_price,  # open
                    current_price,  # high  
                    current_price,  # low
                    current_price,  # close
                    1000           # dummy volume
                ]]
                
                logger.info(f"💱 Created current price candle for {symbol}: {current_price}")
                return ohlcv
            else:
                logger.warning(f"❌ Could not get current price for {symbol}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Failed to get forex OHLCV for {symbol}: {e}")
            return []
