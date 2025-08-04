import asyncio
import logging
import numpy as np
from datetime import datetime, time, timezone
import pandas as pd
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ForexMomentumStrategy:
    """Professional Forex Momentum Strategy with Enhanced Market Analysis"""

    def __init__(self):
        self.name = "forex_momentum"
        self.metatrader_manager = None
        self.risk_manager = None

        # Enhanced timeframe analysis
        self.primary_timeframe = '1H'
        self.confirmation_timeframe = '4H'
        self.lookback_periods = 24  # 24 hours for 1H chart

        # Forex-specific thresholds (in pips)
        self.momentum_thresholds = {
            'EURUSD': 15,   # 15 pips
            'GBPUSD': 20,   # 20 pips (more volatile)
            'USDJPY': 0.15, # 15 pips (JPY pairs different scale)
            'USDCHF': 15,   # 15 pips
            'AUDUSD': 18,   # 18 pips
            'USDCAD': 18,   # 18 pips
            'NZDUSD': 20,   # 20 pips
            'EURJPY': 0.20, # 20 pips
            'GBPJPY': 0.25, # 25 pips
            'EURGBP': 12    # 12 pips (less volatile)
        }

        # Market session awareness
        self.active_sessions = {
            'london': {'start': time(8, 0), 'end': time(17, 0)},    # UTC
            'new_york': {'start': time(13, 0), 'end': time(22, 0)}, # UTC
            'tokyo': {'start': time(0, 0), 'end': time(9, 0)}       # UTC
        }

        # Enhanced risk parameters
        self.max_spread_threshold = {
            'major': 3.0,    # 3 pips for majors
            'minor': 5.0,    # 5 pips for minors
            'exotic': 10.0   # 10 pips for exotics
        }

        # Correlation filters
        self.correlation_pairs = {
            'EURUSD': ['GBPUSD', 'AUDUSD', 'NZDUSD'],
            'USDJPY': ['USDCHF', 'USDCAD'],
            'GBPUSD': ['EURUSD', 'EURGBP'],
        }

        # Performance tracking
        self.signal_history = []
        self.pair_performance = {}

    def initialize(self, metatrader_manager, risk_manager):
        """Initialize enhanced forex momentum strategy"""
        self.metatrader_manager = metatrader_manager
        self.risk_manager = risk_manager

        # Initialize pair performance tracking
        for pair in self.momentum_thresholds.keys():
            self.pair_performance[pair] = {
                'signals_generated': 0,
                'successful_signals': 0,
                'avg_profit': 0.0,
                'last_signal_time': None
            }

        logger.info("✅ Professional Forex Momentum strategy initialized")
        logger.info(f"📊 Monitoring {len(self.momentum_thresholds)} forex pairs")

    async def generate_signals(self):
        """Generate enhanced forex momentum trading signals"""
        signals = []

        try:
            # Check market session (only trade during active sessions)
            if not self._is_active_trading_session():
                logger.debug("📊 Outside active trading sessions - skipping forex signals")
                return signals

            # Get forex pairs based on current session
            active_pairs = self._get_session_active_pairs()

            for pair in active_pairs:
                try:
                    # Check if we should analyze this pair (avoid overtrading)
                    if not self._should_analyze_pair(pair):
                        continue

                    # Get multi-timeframe data
                    primary_data = await self._get_timeframe_data(pair, self.primary_timeframe)
                    confirmation_data = await self._get_timeframe_data(pair, self.confirmation_timeframe)

                    if not primary_data or not confirmation_data:
                        continue

                    # Perform enhanced momentum analysis
                    signal = await self._analyze_enhanced_momentum(
                        pair, primary_data, confirmation_data
                    )

                    if signal and await self._validate_forex_signal(signal):
                        signals.append(signal)
                        self._update_pair_performance(pair, 'signal_generated')
                        logger.info(f"💱 Enhanced Forex Signal: {pair} - {signal['action']} - Confidence: {signal['confidence']:.2%}")

                except Exception as e:
                    logger.error(f"❌ Error analyzing {pair}: {e}")
                    continue

        except Exception as e:
            logger.error(f"❌ Enhanced forex momentum signal generation failed: {e}")

        return signals

    def _is_active_trading_session(self):
        """Check if we're in an active forex trading session"""
        current_time = datetime.now(timezone.utc).time()

        for session, times in self.active_sessions.items():
            if times['start'] <= current_time <= times['end']:
                return True

        # Also check for session overlaps (high volatility periods)
        london_active = (self.active_sessions['london']['start'] <= 
                        current_time <= self.active_sessions['london']['end'])
        ny_active = (self.active_sessions['new_york']['start'] <= 
                    current_time <= self.active_sessions['new_york']['end'])

        return london_active or ny_active

    def _get_session_active_pairs(self):
        """Get pairs that are most active during current session"""
        current_time = datetime.now(timezone.utc).time()

        # London session - EUR, GBP pairs most active
        if (self.active_sessions['london']['start'] <= 
            current_time <= self.active_sessions['london']['end']):
            return ['EURUSD', 'GBPUSD', 'EURGBP', 'EURJPY', 'GBPJPY']

        # New York session - USD pairs most active  
        elif (self.active_sessions['new_york']['start'] <= 
              current_time <= self.active_sessions['new_york']['end']):
            return ['EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF']

        # Tokyo session - JPY pairs most active
        elif (self.active_sessions['tokyo']['start'] <= 
              current_time <= self.active_sessions['tokyo']['end']):
            return ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDUSD']

        # Default to major pairs
        return ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']

    def _should_analyze_pair(self, pair):
        """Determine if we should analyze this pair (avoid overtrading)"""
        pair_perf = self.pair_performance.get(pair, {})
        last_signal = pair_perf.get('last_signal_time')

        # Don't generate signals too frequently for same pair
        if last_signal:
            time_since_last = (datetime.now() - last_signal).total_seconds()
            if time_since_last < 3600:  # 1 hour minimum between signals
                return False

        return True

    async def _get_timeframe_data(self, pair, timeframe):
        """Get OHLCV data for specific timeframe"""
        try:
            if not self.metatrader_manager:
                return None

            # Get more data for higher timeframe analysis
            periods = self.lookback_periods * 2 if timeframe == self.confirmation_timeframe else self.lookback_periods

            ohlcv = await self.metatrader_manager.get_forex_ohlcv(
                pair, timeframe, periods + 10
            )

            return ohlcv if len(ohlcv) >= self.lookback_periods else None

        except Exception as e:
            logger.error(f"❌ Failed to get {timeframe} data for {pair}: {e}")
            return None

    async def _analyze_enhanced_momentum(self, pair, primary_data, confirmation_data):
        """Enhanced multi-timeframe momentum analysis"""
        try:
            # Primary timeframe analysis
            primary_signal = self._analyze_timeframe_momentum(pair, primary_data, 'primary')
            if not primary_signal:
                return None

            # Confirmation timeframe analysis
            confirmation_signal = self._analyze_timeframe_momentum(pair, confirmation_data, 'confirmation')
            if not confirmation_signal:
                return None

            # Check if both timeframes agree
            if primary_signal['direction'] != confirmation_signal['direction']:
                logger.debug(f"📊 Timeframe disagreement for {pair} - skipping signal")
                return None

            # Combine analysis results
            combined_signal = self._combine_timeframe_signals(
                pair, primary_signal, confirmation_signal
            )

            return combined_signal

        except Exception as e:
            logger.error(f"❌ Enhanced momentum analysis failed for {pair}: {e}")
            return None

    def _analyze_timeframe_momentum(self, pair, ohlcv_data, timeframe_type):
        """Analyze momentum for a specific timeframe"""
        try:
            closes = np.array([candle[4] for candle in ohlcv_data])
            highs = np.array([candle[2] for candle in ohlcv_data])
            lows = np.array([candle[3] for candle in ohlcv_data])

            if len(closes) < self.lookback_periods:
                return None

            current_price = closes[-1]

            # Enhanced moving average analysis
            ema_fast = self._calculate_ema(closes, 12)
            ema_slow = self._calculate_ema(closes, 26)
            ema_signal = self._calculate_ema(ema_fast - ema_slow, 9)  # MACD signal

            # Momentum oscillators
            rsi = self._calculate_rsi(closes, 14)
            stoch_k, stoch_d = self._calculate_stochastic(highs, lows, closes, 14)

            # Trend strength indicators
            adx = self._calculate_adx(highs, lows, closes, 14)
            atr = self._calculate_enhanced_atr(ohlcv_data[-20:])

            # Price momentum
            momentum_threshold = self._get_pair_threshold(pair)
            momentum_5 = (current_price - closes[-6]) / closes[-6] if len(closes) > 5 else 0
            momentum_20 = (current_price - closes[-21]) / closes[-21] if len(closes) > 20 else 0

            # Determine trend direction and strength
            macd_histogram = (ema_fast[-1] - ema_slow[-1]) - ema_signal[-1]
            trend_strength = adx

            # Bull signal conditions
            if (ema_fast[-1] > ema_slow[-1] and 
                macd_histogram > 0 and
                momentum_5 > momentum_threshold and
                rsi > 50 and rsi < 75 and
                stoch_k > stoch_d and
                trend_strength > 25):

                return {
                    'direction': 'buy',
                    'strength': min(trend_strength / 50, 1.0),
                    'momentum': momentum_5,
                    'rsi': rsi,
                    'atr': atr,
                    'price': current_price,
                    'macd_histogram': macd_histogram
                }

            # Bear signal conditions
            elif (ema_fast[-1] < ema_slow[-1] and 
                  macd_histogram < 0 and
                  momentum_5 < -momentum_threshold and
                  rsi < 50 and rsi > 25 and
                  stoch_k < stoch_d and
                  trend_strength > 25):

                return {
                    'direction': 'sell',
                    'strength': min(trend_strength / 50, 1.0),
                    'momentum': momentum_5,
                    'rsi': rsi,
                    'atr': atr,
                    'price': current_price,
                    'macd_histogram': macd_histogram
                }

            return None

        except Exception as e:
            logger.error(f"❌ Timeframe momentum analysis failed: {e}")
            return None

    def _combine_timeframe_signals(self, pair, primary, confirmation):
        """Combine signals from multiple timeframes"""
        try:
            # Calculate combined confidence
            base_confidence = (primary['strength'] + confirmation['strength']) / 2
            momentum_factor = min(abs(primary['momentum']) * 100, 0.3)
            trend_factor = min(abs(primary['macd_histogram']) * 1000, 0.2)

            combined_confidence = min(0.85, base_confidence + momentum_factor + trend_factor)

            # Calculate targets and stops using ATR
            atr = primary['atr']
            current_price = primary['price']

            if primary['direction'] == 'buy':
                target = current_price + (atr * 2.5)  # 2.5x ATR target
                stop_loss = current_price - (atr * 1.5)  # 1.5x ATR stop
                action = 'buy'
            else:
                target = current_price - (atr * 2.5)
                stop_loss = current_price + (atr * 1.5)
                action = 'sell'

            return {
                'pair': pair,
                'action': action,
                'price': current_price,
                'target': target,
                'stop_loss': stop_loss,
                'confidence': combined_confidence,
                'strategy': self.name,
                'timeframe_analysis': {
                    'primary': primary,
                    'confirmation': confirmation
                },
                'risk_reward_ratio': abs((target - current_price) / (current_price - stop_loss)),
                'atr_pips': atr,
                'expected_move': abs(target - current_price),
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"❌ Signal combination failed: {e}")
            return None

    async def _validate_forex_signal(self, signal):
        """Enhanced forex signal validation"""
        try:
            # Check risk-reward ratio
            if signal.get('risk_reward_ratio', 0) < 1.5:  # Minimum 1.5:1 R:R
                logger.debug(f"❌ Poor risk-reward ratio for {signal['pair']}: {signal.get('risk_reward_ratio', 0):.2f}")
                return False

            # Check spread conditions (would need real spread data)
            # For now, assume reasonable spreads during active sessions

            # Check correlation with existing positions
            if not self._check_correlation_limits(signal):
                return False

            # Check pair-specific performance
            pair_perf = self.pair_performance.get(signal['pair'], {})
            if pair_perf.get('successful_signals', 0) == 0 and pair_perf.get('signals_generated', 0) > 5:
                logger.debug(f"❌ Poor historical performance for {signal['pair']}")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Signal validation failed: {e}")
            return False

    def _check_correlation_limits(self, signal):
        """Check correlation limits with existing positions"""
        # This would check if we already have too many correlated positions
        # For now, simple implementation
        return True

    def _get_pair_threshold(self, pair):
        """Get momentum threshold for specific pair"""
        return self.momentum_thresholds.get(pair, 15) / 10000  # Convert pips to decimal

    def _calculate_ema(self, prices, periods):
        """Calculate Exponential Moving Average"""
        try:
            alpha = 2 / (periods + 1)
            ema = np.zeros_like(prices)
            ema[0] = prices[0]

            for i in range(1, len(prices)):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

            return ema
        except:
            return np.full_like(prices, prices[-1])

    def _calculate_stochastic(self, highs, lows, closes, periods=14):
        """Calculate Stochastic Oscillator"""
        try:
            stoch_k = np.zeros(len(closes))

            for i in range(periods-1, len(closes)):
                highest_high = np.max(highs[i-periods+1:i+1])
                lowest_low = np.min(lows[i-periods+1:i+1])

                if highest_high != lowest_low:
                    stoch_k[i] = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low)
                else:
                    stoch_k[i] = 50

            # Smooth %K to get %D
            stoch_d = self._calculate_sma(stoch_k, 3)

            return stoch_k, stoch_d
        except:
            return np.full(len(closes), 50), np.full(len(closes), 50)

    def _calculate_adx(self, highs, lows, closes, periods=14):
        """Calculate Average Directional Index (trend strength)"""
        try:
            # Simplified ADX calculation
            tr = self._calculate_true_range_series(highs, lows, closes)
            atr = np.mean(tr[-periods:])

            # DM calculations (simplified)
            dm_plus = np.maximum(highs[1:] - highs[:-1], 0)
            dm_minus = np.maximum(lows[:-1] - lows[1:], 0)

            di_plus = 100 * np.mean(dm_plus[-periods:]) / atr
            di_minus = 100 * np.mean(dm_minus[-periods:]) / atr

            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)

            return min(dx, 100)
        except:
            return 25  # Neutral trend strength

    def _calculate_enhanced_atr(self, ohlcv_data, periods=14):
        """Enhanced ATR calculation"""
        try:
            if len(ohlcv_data) < 2:
                return 0.0001

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

            # Use EMA for smoother ATR
            if len(true_ranges) >= periods:
                atr_ema = self._calculate_ema(np.array(true_ranges), periods)
                return atr_ema[-1]
            else:
                return np.mean(true_ranges) if true_ranges else 0.0001

        except Exception as e:
            logger.error(f"❌ Enhanced ATR calculation failed: {e}")
            return 0.0001

    def _calculate_true_range_series(self, highs, lows, closes):
        """Calculate True Range series"""
        tr = np.zeros(len(highs))

        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr[i] = max(tr1, tr2, tr3)

        return tr[1:]  # Skip first element (always 0)

    def _calculate_sma(self, prices, periods):
        """Calculate Simple Moving Average"""
        sma = np.zeros_like(prices)
        for i in range(periods-1, len(prices)):
            sma[i] = np.mean(prices[i-periods+1:i+1])
        return sma

    def _calculate_rsi(self, prices, periods=14):
        """Enhanced RSI calculation"""
        try:
            if len(prices) < periods + 1:
                return 50

            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            # Use Wilder's smoothing
            avg_gain = np.mean(gains[:periods])
            avg_loss = np.mean(losses[:periods])

            for i in range(periods, len(gains)):
                avg_gain = (avg_gain * (periods - 1) + gains[i]) / periods
                avg_loss = (avg_loss * (periods - 1) + losses[i]) / periods

            if avg_loss == 0:
                return 100

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return max(0.0, min(100.0, float(rsi)))

        except Exception as e:
            logger.error(f"❌ RSI calculation failed: {e}")
            return 50

    def _update_pair_performance(self, pair, event_type, profit=None):
        """Update performance tracking for pair"""
        if pair not in self.pair_performance:
            self.pair_performance[pair] = {
                'signals_generated': 0,
                'successful_signals': 0,
                'avg_profit': 0.0,
                'last_signal_time': None
            }

        perf = self.pair_performance[pair]

        if event_type == 'signal_generated':
            perf['signals_generated'] += 1
            perf['last_signal_time'] = datetime.now()
        elif event_type == 'signal_success' and profit is not None:
            perf['successful_signals'] += 1
            # Update average profit
            current_avg = perf['avg_profit']
            success_count = perf['successful_signals']
            perf['avg_profit'] = ((current_avg * (success_count - 1)) + profit) / success_count

    def get_strategy_performance(self):
        """Get comprehensive strategy performance metrics"""
        total_signals = sum(p['signals_generated'] for p in self.pair_performance.values())
        total_successful = sum(p['successful_signals'] for p in self.pair_performance.values())

        if total_signals == 0:
            return {
                'total_signals': 0,
                'success_rate': 0,
                'avg_profit': 0,
                'best_pair': None,
                'worst_pair': None
            }

        success_rate = total_successful / total_signals
        avg_profit = np.mean([p['avg_profit'] for p in self.pair_performance.values() if p['successful_signals'] > 0])

        # Find best and worst performing pairs
        best_pair = max(self.pair_performance.items(), 
                       key=lambda x: x[1]['avg_profit'] if x[1]['successful_signals'] > 0 else -1)
        worst_pair = min(self.pair_performance.items(), 
                        key=lambda x: x[1]['avg_profit'] if x[1]['successful_signals'] > 0 else 1)

        return {
            'total_signals': total_signals,
            'success_rate': success_rate,
            'avg_profit': avg_profit or 0,
            'best_pair': best_pair[0] if best_pair[1]['successful_signals'] > 0 else None,
            'worst_pair': worst_pair[0] if worst_pair[1]['successful_signals'] > 0 else None,
            'pair_breakdown': self.pair_performance
        }
