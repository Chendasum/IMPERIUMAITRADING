import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class ProfessionalMomentumStrategy:
    """Professional Momentum Strategy with Advanced Technical Analysis"""

    def __init__(self):
        self.name = "momentum"
        self.exchange_manager = None
        self.risk_manager = None

        # Multi-timeframe momentum analysis
        self.primary_timeframe = '1h'
        self.confirmation_timeframes = ['4h', '1d']
        self.lookback_periods = 21  # More periods for better analysis

        # Dynamic momentum thresholds based on asset volatility
        self.base_momentum_threshold = 0.015  # 1.5% base threshold
        self.asset_multipliers = {
            'BTC/USDT': 1.0,  # Base threshold
            'ETH/USDT': 1.2,  # 20% higher threshold (more volatile)
            'BNB/USDT': 1.4,  # 40% higher threshold
            'ADA/USDT': 1.6,  # 60% higher threshold
            'DOT/USDT': 1.8,  # 80% higher threshold
        }

        # Advanced momentum indicators
        self.momentum_indicators = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_period': 14,
            'roc_period': 10,  # Rate of Change
            'cci_period': 20,  # Commodity Channel Index
        }

        # Momentum strength levels
        self.momentum_levels = {
            'weak': (0.01, 0.025),  # 1-2.5%
            'moderate': (0.025, 0.05),  # 2.5-5%
            'strong': (0.05, 0.10),  # 5-10%
            'extreme': (0.10, float('inf'))  # >10%
        }

        # Trend strength requirements
        self.trend_strength_min = 0.6  # Minimum trend strength (0-1)
        self.volume_confirmation_required = True

        # Risk management parameters
        self.max_drawdown_tolerance = 0.15  # 15%
        self.momentum_decay_threshold = 0.3  # 30% momentum decay = exit

        # Market condition filters
        self.market_volatility_threshold = 0.04  # 4% daily volatility threshold
        self.min_daily_volume = 50000000  # $50M minimum daily volume

        # Performance tracking
        self.momentum_performance = {
            'total_signals': 0,
            'successful_momentum_trades': 0,
            'avg_momentum_profit': 0.0,
            'momentum_strength_performance': {},
            'timeframe_performance': {}
        }

        # Signal quality tracking
        self.signal_history = []
        self.max_signal_history = 100

    def initialize(self, exchange_manager, risk_manager):
        """Initialize enhanced momentum strategy"""
        self.exchange_manager = exchange_manager
        self.risk_manager = risk_manager

        # Initialize performance tracking
        for strength in self.momentum_levels.keys():
            self.momentum_performance['momentum_strength_performance'][
                strength] = {
                    'signals': 0,
                    'successful': 0,
                    'avg_profit': 0.0
                }

        for tf in self.confirmation_timeframes + [self.primary_timeframe]:
            self.momentum_performance['timeframe_performance'][tf] = {
                'signals': 0,
                'successful': 0,
                'avg_profit': 0.0
            }

        logger.info("✅ Professional Momentum strategy initialized")
        logger.info(
            f"📊 Multi-timeframe analysis: {[self.primary_timeframe] + self.confirmation_timeframes}"
        )
        logger.info(
            f"🎯 Monitoring {len(self.asset_multipliers)} crypto assets")

    async def generate_signals(self):
        """Generate enhanced momentum trading signals"""
        signals = []

        try:
            # Get market volatility assessment
            market_conditions = await self._assess_market_conditions()

            if not market_conditions['suitable_for_momentum']:
                logger.debug(
                    "📊 Market conditions not suitable for momentum trading")
                return signals

            # Get active pairs based on volume and volatility
            active_pairs = await self._get_momentum_suitable_pairs()

            for pair in active_pairs:
                try:
                    # Gather comprehensive market data
                    market_data = await self._gather_momentum_data(pair)
                    if not market_data:
                        continue

                    # Perform multi-dimensional momentum analysis
                    momentum_analysis = await self._analyze_advanced_momentum(
                        pair, market_data)
                    if not momentum_analysis:
                        continue

                    # Generate signal if momentum is confirmed
                    signal = await self._create_momentum_signal(
                        pair, momentum_analysis, market_conditions)

                    if signal and await self._validate_momentum_signal(
                            signal, market_data):
                        signals.append(signal)
                        self._track_signal_generation(signal)
                        logger.info(
                            f"📈 Momentum Signal: {pair} - {signal['action']} - Strength: {signal['momentum_strength']} - Confidence: {signal['confidence']:.2%}"
                        )

                except Exception as e:
                    logger.error(f"❌ Error analyzing momentum for {pair}: {e}")
                    continue

            # Rank signals by quality and return top performers
            return self._rank_momentum_signals(signals)

        except Exception as e:
            logger.error(f"❌ Enhanced momentum signal generation failed: {e}")
            return signals

    async def _assess_market_conditions(self):
        """Assess overall market conditions for momentum trading"""
        try:
            # Analyze BTC as market leader
            btc_data = await self._get_timeframe_data('BTC/USDT', '1d', 30)
            if not btc_data:
                return {
                    'suitable_for_momentum': False,
                    'market_trend': 'unknown',
                    'volatility': 0
                }

            btc_closes = np.array([candle[4] for candle in btc_data])
            btc_volumes = np.array([candle[5] for candle in btc_data])

            # Calculate market metrics
            daily_returns = np.diff(btc_closes) / btc_closes[:-1]
            market_volatility = np.std(daily_returns) * np.sqrt(
                365)  # Annualized
            trend_strength = self._calculate_trend_strength(btc_closes)
            avg_volume = np.mean(btc_volumes[-7:])  # 7-day average

            # Determine market regime
            recent_momentum = (btc_closes[-1] - btc_closes[-7]
                               ) / btc_closes[-7]  # 7-day momentum

            if recent_momentum > 0.05:  # 5% weekly gain
                market_trend = 'bullish'
            elif recent_momentum < -0.05:  # 5% weekly loss
                market_trend = 'bearish'
            else:
                market_trend = 'sideways'

            # Momentum is best in trending markets with moderate volatility
            suitable = (
                trend_strength > 0.4 and 0.02 < market_volatility < 0.08
                and  # 2-8% volatility
                avg_volume > self.min_daily_volume)

            return {
                'suitable_for_momentum': suitable,
                'market_trend': market_trend,
                'volatility': market_volatility,
                'trend_strength': trend_strength,
                'avg_volume': avg_volume
            }

        except Exception as e:
            logger.error(f"❌ Market condition assessment failed: {e}")
            return {
                'suitable_for_momentum': False,
                'market_trend': 'unknown',
                'volatility': 0
            }

    async def _get_momentum_suitable_pairs(self):
        """Get pairs suitable for momentum trading based on volume and volatility"""
        try:
            suitable_pairs = []

            for pair in self.asset_multipliers.keys():
                # Get recent data for volume/volatility check
                recent_data = await self._get_timeframe_data(pair, '1d', 7)
                if not recent_data:
                    continue

                volumes = np.array([candle[5] for candle in recent_data])
                closes = np.array([candle[4] for candle in recent_data])

                # Check volume requirement
                avg_volume = np.mean(volumes) * closes[-1]  # Volume in USD
                if avg_volume < self.min_daily_volume * 0.5:  # 50% of BTC requirement
                    continue

                # Check volatility (momentum needs some volatility)
                returns = np.diff(closes) / closes[:-1]
                volatility = np.std(returns)

                if 0.01 < volatility < 0.15:  # 1-15% daily volatility
                    suitable_pairs.append(pair)

            return suitable_pairs

        except Exception as e:
            logger.error(f"❌ Failed to get suitable pairs: {e}")
            return list(self.asset_multipliers.keys())[:3]  # Fallback to top 3

    async def _gather_momentum_data(self, pair):
        """Gather comprehensive data for momentum analysis"""
        try:
            market_data = {}

            # Get primary timeframe data
            primary_data = await self._get_timeframe_data(
                pair, self.primary_timeframe, self.lookback_periods + 20)
            if not primary_data or len(primary_data) < self.lookback_periods:
                return None

            market_data['primary'] = primary_data

            # Get confirmation timeframe data
            for tf in self.confirmation_timeframes:
                tf_data = await self._get_timeframe_data(pair, tf, 50)
                if tf_data and len(tf_data) >= 20:
                    market_data[tf] = tf_data

            # Calculate volume profile
            market_data['volume_analysis'] = self._analyze_volume_momentum(
                primary_data)

            return market_data

        except Exception as e:
            logger.error(f"❌ Failed to gather momentum data for {pair}: {e}")
            return None

    async def _get_timeframe_data(self, pair, timeframe, periods):
        """Get OHLCV data for specific timeframe"""
        try:
            exchange_name = list(self.exchange_manager.exchanges.keys(
            ))[0] if self.exchange_manager.exchanges else 'binance'

            ohlcv = await self.exchange_manager.get_ohlcv(
                exchange_name, pair, timeframe, periods)

            return ohlcv if len(ohlcv) >= 10 else None

        except Exception as e:
            logger.error(f"❌ Failed to get {timeframe} data for {pair}: {e}")
            return None

    def _analyze_volume_momentum(self, ohlcv_data):
        """Analyze volume patterns for momentum confirmation"""
        try:
            volumes = np.array([candle[5] for candle in ohlcv_data])
            closes = np.array([candle[4] for candle in ohlcv_data])

            # Volume trend analysis
            volume_sma_short = np.mean(volumes[-5:])
            volume_sma_long = np.mean(volumes[-20:])
            volume_trend = (volume_sma_short -
                            volume_sma_long) / volume_sma_long

            # Price-volume relationship
            price_changes = np.diff(closes[-10:]) / closes[-11:-1]
            volume_changes = np.diff(volumes[-10:]) / volumes[-11:-1]

            pv_correlation = np.corrcoef(
                price_changes,
                volume_changes)[0, 1] if len(price_changes) > 1 else 0

            # On-balance volume trend
            obv = self._calculate_obv(closes[-20:], volumes[-20:])
            obv_trend = (obv[-1] -
                         obv[-5]) / abs(obv[-5]) if obv[-5] != 0 else 0

            return {
                'volume_trend':
                volume_trend,
                'price_volume_correlation':
                pv_correlation,
                'obv_trend':
                obv_trend,
                'volume_momentum_score':
                (volume_trend + abs(pv_correlation) + obv_trend) / 3
            }

        except Exception as e:
            logger.error(f"❌ Volume momentum analysis failed: {e}")
            return {
                'volume_trend': 0,
                'price_volume_correlation': 0,
                'obv_trend': 0,
                'volume_momentum_score': 0
            }

    def _calculate_obv(self, closes, volumes):
        """Calculate On-Balance Volume"""
        try:
            obv = np.zeros(len(closes))
            obv[0] = volumes[0]

            for i in range(1, len(closes)):
                if closes[i] > closes[i - 1]:
                    obv[i] = obv[i - 1] + volumes[i]
                elif closes[i] < closes[i - 1]:
                    obv[i] = obv[i - 1] - volumes[i]
                else:
                    obv[i] = obv[i - 1]

            return obv

        except Exception as e:
            logger.error(f"❌ OBV calculation failed: {e}")
            return np.zeros(len(closes))

    async def _analyze_advanced_momentum(self, pair, market_data):
        """Perform comprehensive momentum analysis"""
        try:
            primary_data = market_data['primary']
            closes = np.array([candle[4] for candle in primary_data])
            highs = np.array([candle[2] for candle in primary_data])
            lows = np.array([candle[3] for candle in primary_data])
            volumes = np.array([candle[5] for candle in primary_data])

            current_price = closes[-1]

            # 1. Multiple momentum indicators
            momentum_indicators = self._calculate_momentum_indicators(
                highs, lows, closes, volumes)

            # 2. Multi-timeframe momentum confirmation
            mtf_momentum = self._get_multitimeframe_momentum(
                market_data, current_price)

            # 3. Trend strength and quality
            trend_analysis = self._analyze_trend_quality(closes, highs, lows)

            # 4. Momentum persistence analysis
            persistence_analysis = self._analyze_momentum_persistence(closes)

            # 5. Breakout analysis
            breakout_analysis = self._analyze_breakout_potential(
                highs, lows, closes, volumes)

            # 6. Market structure analysis
            structure_analysis = self._analyze_market_structure(
                highs, lows, closes)

            # Combine all analyses
            combined_analysis = self._combine_momentum_analyses(
                pair, current_price, momentum_indicators, mtf_momentum,
                trend_analysis, persistence_analysis, breakout_analysis,
                structure_analysis, market_data['volume_analysis'])

            return combined_analysis

        except Exception as e:
            logger.error(
                f"❌ Advanced momentum analysis failed for {pair}: {e}")
            return None

    def _calculate_momentum_indicators(self, highs, lows, closes, volumes):
        """Calculate multiple momentum indicators"""
        try:
            indicators = {}

            # 1. Rate of Change (ROC)
            roc_period = self.momentum_indicators['roc_period']
            if len(closes) > roc_period:
                indicators['roc'] = (closes[-1] - closes[-roc_period - 1]
                                     ) / closes[-roc_period - 1]
            else:
                indicators['roc'] = 0

            # 2. RSI
            indicators['rsi'] = self._calculate_enhanced_rsi(closes)

            # 3. MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(
                closes)
            indicators['macd'] = {
                'line': macd_line,
                'signal': macd_signal,
                'histogram': macd_histogram
            }

            # 4. Commodity Channel Index (CCI)
            indicators['cci'] = self._calculate_cci(highs, lows, closes)

            # 5. Average True Range (for volatility)
            indicators['atr'] = self._calculate_atr(highs, lows, closes)

            # 6. Momentum Oscillator
            indicators[
                'momentum_oscillator'] = self._calculate_momentum_oscillator(
                    closes)

            # 7. Directional Movement Index
            indicators['adx'] = self._calculate_adx(highs, lows, closes)

            return indicators

        except Exception as e:
            logger.error(f"❌ Momentum indicators calculation failed: {e}")
            return {}

    def _calculate_enhanced_rsi(self, prices, periods=14):
        """Calculate enhanced RSI with better smoothing"""
        try:
            if len(prices) < periods + 1:
                return 50

            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            # Use Wilder's smoothing method
            avg_gain = np.mean(gains[:periods])
            avg_loss = np.mean(losses[:periods])

            for i in range(periods, len(gains)):
                avg_gain = (avg_gain * (periods - 1) + gains[i]) / periods
                avg_loss = (avg_loss * (periods - 1) + losses[i]) / periods

            if avg_loss == 0:
                return 100

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return max(0, min(100, rsi))

        except Exception as e:
            logger.error(f"❌ Enhanced RSI calculation failed: {e}")
            return 50

    def _calculate_macd(self, prices):
        """Calculate MACD indicator"""
        try:
            fast_period = self.momentum_indicators['macd_fast']
            slow_period = self.momentum_indicators['macd_slow']
            signal_period = self.momentum_indicators['macd_signal']

            if len(prices) < slow_period:
                return 0, 0, 0

            # Calculate EMAs
            ema_fast = self._calculate_ema(prices, fast_period)
            ema_slow = self._calculate_ema(prices, slow_period)

            # MACD line
            macd_line = ema_fast[-1] - ema_slow[-1]

            # MACD signal line (EMA of MACD line)
            macd_values = ema_fast[-signal_period:] - ema_slow[-signal_period:]
            if len(macd_values) >= signal_period:
                macd_signal = self._calculate_ema(macd_values,
                                                  signal_period)[-1]
            else:
                macd_signal = macd_line

            # MACD histogram
            macd_histogram = macd_line - macd_signal

            return macd_line, macd_signal, macd_histogram

        except Exception as e:
            logger.error(f"❌ MACD calculation failed: {e}")
            return 0, 0, 0

    def _calculate_ema(self, prices, periods):
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) < periods:
                return np.full(len(prices), prices[-1])

            multiplier = 2 / (periods + 1)
            ema = np.zeros(len(prices))
            ema[0] = prices[0]

            for i in range(1, len(prices)):
                ema[i] = (prices[i] * multiplier) + (ema[i - 1] *
                                                     (1 - multiplier))

            return ema

        except Exception as e:
            logger.error(f"❌ EMA calculation failed: {e}")
            return np.full(len(prices), prices[-1])

    def _calculate_cci(self, highs, lows, closes, periods=20):
        """Calculate Commodity Channel Index"""
        try:
            if len(closes) < periods:
                return 0

            typical_prices = (highs + lows + closes) / 3
            sma_tp = np.mean(typical_prices[-periods:])
            mean_deviation = np.mean(np.abs(typical_prices[-periods:] -
                                            sma_tp))

            if mean_deviation == 0:
                return 0

            cci = (typical_prices[-1] - sma_tp) / (0.015 * mean_deviation)
            return max(-300, min(300, cci))  # Clamp to reasonable range

        except Exception as e:
            logger.error(f"❌ CCI calculation failed: {e}")
            return 0

    def _calculate_atr(self, highs, lows, closes, periods=14):
        """Calculate Average True Range"""
        try:
            if len(closes) < 2:
                return 0.01

            true_ranges = []
            for i in range(1, len(closes)):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i - 1])
                tr3 = abs(lows[i] - closes[i - 1])
                true_ranges.append(max(tr1, tr2, tr3))

            if len(true_ranges) >= periods:
                return np.mean(true_ranges[-periods:])
            else:
                return np.mean(true_ranges) if true_ranges else 0.01

        except Exception as e:
            logger.error(f"❌ ATR calculation failed: {e}")
            return 0.01

    def _calculate_momentum_oscillator(self, prices, periods=10):
        """Calculate Momentum Oscillator"""
        try:
            if len(prices) <= periods:
                return 0

            return (prices[-1] - prices[-periods - 1]) / prices[-periods - 1]

        except Exception as e:
            logger.error(f"❌ Momentum oscillator calculation failed: {e}")
            return 0

    def _calculate_adx(self, highs, lows, closes, periods=14):
        """Calculate Average Directional Index (trend strength)"""
        try:
            if len(closes) < periods + 1:
                return 25  # Neutral trend strength

            # Calculate True Range
            tr = []
            dm_plus = []
            dm_minus = []

            for i in range(1, len(closes)):
                # True Range
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i - 1])
                tr3 = abs(lows[i] - closes[i - 1])
                tr.append(max(tr1, tr2, tr3))

                # Directional Movement
                up_move = highs[i] - highs[i - 1]
                down_move = lows[i - 1] - lows[i]

                dm_plus.append(
                    up_move if up_move > down_move and up_move > 0 else 0)
                dm_minus.append(
                    down_move if down_move > up_move and down_move > 0 else 0)

            # Smooth the values
            atr = np.mean(tr[-periods:])
            di_plus = 100 * np.mean(dm_plus[-periods:]) / atr if atr > 0 else 0
            di_minus = 100 * np.mean(
                dm_minus[-periods:]) / atr if atr > 0 else 0

            # Calculate ADX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)

            return min(dx, 100)

        except Exception as e:
            logger.error(f"❌ ADX calculation failed: {e}")
            return 25

    def _get_multitimeframe_momentum(self, market_data, current_price):
        """Get momentum confirmation from multiple timeframes"""
        try:
            mtf_signals = []

            for tf in self.confirmation_timeframes:
                if tf not in market_data:
                    continue

                tf_data = market_data[tf]
                tf_closes = np.array([candle[4] for candle in tf_data])

                # Calculate momentum for this timeframe
                if len(tf_closes) >= 10:
                    tf_momentum = (current_price -
                                   tf_closes[-10]) / tf_closes[-10]
                    tf_trend = self._calculate_trend_strength(tf_closes)

                    mtf_signals.append({
                        'timeframe':
                        tf,
                        'momentum':
                        tf_momentum,
                        'trend_strength':
                        tf_trend,
                        'direction':
                        'bullish' if tf_momentum > 0.01 else
                        'bearish' if tf_momentum < -0.01 else 'neutral'
                    })

            # Calculate consensus
            bullish_signals = sum(1 for s in mtf_signals
                                  if s['direction'] == 'bullish')
            bearish_signals = sum(1 for s in mtf_signals
                                  if s['direction'] == 'bearish')
            total_signals = len(mtf_signals)

            if total_signals == 0:
                consensus = 'neutral'
                consensus_strength = 0
            else:
                if bullish_signals > bearish_signals:
                    consensus = 'bullish'
                    consensus_strength = bullish_signals / total_signals
                elif bearish_signals > bullish_signals:
                    consensus = 'bearish'
                    consensus_strength = bearish_signals / total_signals
                else:
                    consensus = 'neutral'
                    consensus_strength = 0.5

            return {
                'signals': mtf_signals,
                'consensus': consensus,
                'consensus_strength': consensus_strength,
                'total_confirmations': len(mtf_signals)
            }

        except Exception as e:
            logger.error(f"❌ Multi-timeframe momentum analysis failed: {e}")
            return {
                'signals': [],
                'consensus': 'neutral',
                'consensus_strength': 0,
                'total_confirmations': 0
            }

    def _calculate_trend_strength(self, prices):
        """Calculate trend strength (0-1)"""
        try:
            if len(prices) < 20:
                return 0.5

            # Linear regression slope
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)

            # R-squared for trend quality
            y_pred = slope * x + np.mean(prices)
            ss_res = np.sum((prices - y_pred)**2)
            ss_tot = np.sum((prices - np.mean(prices))**2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))

            # Combine slope direction and R-squared
            trend_strength = min(abs(slope) / np.mean(prices) * 100,
                                 1.0) * r_squared

            return max(0, min(1, trend_strength))

        except Exception as e:
            logger.error(f"❌ Trend strength calculation failed: {e}")
            return 0.5

    def _analyze_trend_quality(self, closes, highs, lows):
        """Analyze trend quality and consistency"""
        try:
            # Higher highs and higher lows for uptrend
            recent_highs = highs[-10:]
            recent_lows = lows[-10:]

            higher_highs = sum(1 for i in range(1, len(recent_highs))
                               if recent_highs[i] > recent_highs[i - 1])
            higher_lows = sum(1 for i in range(1, len(recent_lows))
                              if recent_lows[i] > recent_lows[i - 1])

            uptrend_quality = (higher_highs +
                               higher_lows) / (2 * (len(recent_highs) - 1))

            # Lower highs and lower lows for downtrend
            lower_highs = sum(1 for i in range(1, len(recent_highs))
                              if recent_highs[i] < recent_highs[i - 1])
            lower_lows = sum(1 for i in range(1, len(recent_lows))
                             if recent_lows[i] < recent_lows[i - 1])

            downtrend_quality = (lower_highs +
                                 lower_lows) / (2 * (len(recent_highs) - 1))

            # Overall trend strength
            trend_strength = self._calculate_trend_strength(closes)

            return {
                'uptrend_quality': uptrend_quality,
                'downtrend_quality': downtrend_quality,
                'trend_strength': trend_strength,
                'trend_consistency': max(uptrend_quality, downtrend_quality)
            }

        except Exception as e:
            logger.error(f"❌ Trend quality analysis failed: {e}")
            return {
                'uptrend_quality': 0,
                'downtrend_quality': 0,
                'trend_strength': 0,
                'trend_consistency': 0
            }

    def _analyze_momentum_persistence(self, closes):
        """Analyze how persistent momentum has been"""
        try:
            if len(closes) < 20:
                return {'persistence_score': 0, 'momentum_periods': 0}

            # Calculate rolling momentum
            momentum_periods = []
            for i in range(10, len(closes)):
                momentum = (closes[i] - closes[i - 5]) / closes[i - 5]
                momentum_periods.append(momentum)

            # Count consistent momentum periods
            positive_momentum = sum(1 for m in momentum_periods if m > 0.01)
            negative_momentum = sum(1 for m in momentum_periods if m < -0.01)

            total_periods = len(momentum_periods)

            # Persistence score
            if total_periods > 0:
                persistence_score = max(positive_momentum,
                                        negative_momentum) / total_periods
            else:
                persistence_score = 0

            return {
                'persistence_score': persistence_score,
                'momentum_periods': total_periods,
                'positive_periods': positive_momentum,
                'negative_periods': negative_momentum
            }

        except Exception as e:
            logger.error(f"❌ Momentum persistence analysis failed: {e}")
            return {
                'persistence_score': 0,
                'momentum_periods': 0,
                'positive_periods': 0,
                'negative_periods': 0
            }

    def _analyze_breakout_potential(self, highs, lows, closes, volumes):
        """Analyze breakout potential for momentum continuation"""
        try:
            if len(closes) < 20:
                return {'breakout_potential': 0, 'volume_confirmation': False}

            # Calculate resistance and support levels
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]

            resistance = np.max(recent_highs[:-1])  # Exclude current candle
            support = np.min(recent_lows[:-1])

            current_price = closes[-1]

            # Distance to breakout levels
            resistance_distance = (resistance - current_price) / current_price
            support_distance = (current_price - support) / current_price

            # Volume analysis for breakout confirmation
            recent_volumes = volumes[-5:]
            avg_volume = np.mean(volumes[-20:-5])
            current_volume_ratio = np.mean(recent_volumes) / avg_volume

            # Breakout potential scoring
            if resistance_distance < 0.02:  # Within 2% of resistance
                breakout_potential = 0.8 if current_volume_ratio > 1.5 else 0.4
                breakout_direction = 'bullish'
            elif support_distance < 0.02:  # Within 2% of support
                breakout_potential = 0.8 if current_volume_ratio > 1.5 else 0.4
                breakout_direction = 'bearish'
            else:
                breakout_potential = 0.2
                breakout_direction = 'neutral'

            return {
                'breakout_potential': breakout_potential,
                'breakout_direction': breakout_direction,
                'volume_confirmation': current_volume_ratio > 1.3,
                'resistance_level': resistance,
                'support_level': support,
                'volume_ratio': current_volume_ratio
            }

        except Exception as e:
            logger.error(f"❌ Breakout analysis failed: {e}")
            return {
                'breakout_potential': 0,
                'volume_confirmation': False,
                'breakout_direction': 'neutral'
            }

    def _analyze_market_structure(self, highs, lows, closes):
        """Analyze market structure for momentum context"""
        try:
            if len(closes) < 20:
                return {'structure_score': 0, 'structure_type': 'unknown'}

            # Identify swing highs and lows
            swing_highs = []
            swing_lows = []

            for i in range(2, len(highs) - 2):
                if highs[i] > highs[i - 1] and highs[i] > highs[
                        i + 1] and highs[i] > highs[i -
                                                    2] and highs[i] > highs[i +
                                                                            2]:
                    swing_highs.append((i, highs[i]))
                if lows[i] < lows[i - 1] and lows[i] < lows[
                        i + 1] and lows[i] < lows[i - 2] and lows[i] < lows[i +
                                                                            2]:
                    swing_lows.append((i, lows[i]))

            # Analyze structure trend
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                # Check for higher highs and higher lows (uptrend structure)
                hh_count = sum(1 for i in range(1, len(swing_highs))
                               if swing_highs[i][1] > swing_highs[i - 1][1])
                hl_count = sum(1 for i in range(1, len(swing_lows))
                               if swing_lows[i][1] > swing_lows[i - 1][1])

                # Check for lower highs and lower lows (downtrend structure)
                lh_count = sum(1 for i in range(1, len(swing_highs))
                               if swing_highs[i][1] < swing_highs[i - 1][1])
                ll_count = sum(1 for i in range(1, len(swing_lows))
                               if swing_lows[i][1] < swing_lows[i - 1][1])

                total_swings = max(
                    len(swing_highs) - 1,
                    len(swing_lows) - 1, 1)

                uptrend_score = (hh_count + hl_count) / (2 * total_swings)
                downtrend_score = (lh_count + ll_count) / (2 * total_swings)

                if uptrend_score > 0.6:
                    structure_type = 'uptrend'
                    structure_score = uptrend_score
                elif downtrend_score > 0.6:
                    structure_type = 'downtrend'
                    structure_score = downtrend_score
                else:
                    structure_type = 'sideways'
                    structure_score = 0.5
            else:
                structure_type = 'unknown'
                structure_score = 0.5

            return {
                'structure_score': structure_score,
                'structure_type': structure_type,
                'swing_highs': len(swing_highs),
                'swing_lows': len(swing_lows)
            }

        except Exception as e:
            logger.error(f"❌ Market structure analysis failed: {e}")
            return {
                'structure_score': 0.5,
                'structure_type': 'unknown',
                'swing_highs': 0,
                'swing_lows': 0
            }

    def _combine_momentum_analyses(self, pair, current_price,
                                   momentum_indicators, mtf_momentum,
                                   trend_analysis, persistence_analysis,
                                   breakout_analysis, structure_analysis,
                                   volume_analysis):
        """Combine all momentum analyses into a comprehensive assessment"""
        try:
            # Determine primary momentum direction
            rsi = momentum_indicators.get('rsi', 50)
            macd_histogram = momentum_indicators.get('macd',
                                                     {}).get('histogram', 0)
            roc = momentum_indicators.get('roc', 0)
            cci = momentum_indicators.get('cci', 0)
            adx = momentum_indicators.get('adx', 25)

            # Get pair-specific threshold
            pair_threshold = self.base_momentum_threshold * self.asset_multipliers.get(
                pair, 1.0)

            # Bullish momentum conditions
            bullish_conditions = [
                rsi > 55
                and rsi < 75,  # RSI in momentum zone but not overbought
                macd_histogram > 0,  # MACD histogram positive
                roc > pair_threshold,  # Rate of change above threshold
                cci > 0,  # CCI positive
                adx > 25,  # Strong trend
                mtf_momentum['consensus'] == 'bullish',
                trend_analysis['uptrend_quality'] > 0.6,
                volume_analysis['volume_momentum_score'] > 0.3
            ]

            # Bearish momentum conditions
            bearish_conditions = [
                rsi < 45 and rsi > 25,  # RSI in momentum zone but not oversold
                macd_histogram < 0,  # MACD histogram negative
                roc
                < -pair_threshold,  # Rate of change below negative threshold
                cci < 0,  # CCI negative
                adx > 25,  # Strong trend
                mtf_momentum['consensus'] == 'bearish',
                trend_analysis['downtrend_quality'] > 0.6,
                volume_analysis['volume_momentum_score'] > 0.3
            ]

            bullish_score = sum(bullish_conditions) / len(bullish_conditions)
            bearish_score = sum(bearish_conditions) / len(bearish_conditions)

            # Determine signal direction
            if bullish_score >= 0.6 and bullish_score > bearish_score:
                signal_direction = 'buy'
                momentum_strength = self._categorize_momentum_strength(
                    abs(roc))
                base_confidence = bullish_score
            elif bearish_score >= 0.6 and bearish_score > bullish_score:
                signal_direction = 'sell'
                momentum_strength = self._categorize_momentum_strength(
                    abs(roc))
                base_confidence = bearish_score
            else:
                return None  # No clear momentum signal

            # Enhance confidence with additional factors
            confidence_multipliers = []

            # Multi-timeframe confirmation
            confidence_multipliers.append(mtf_momentum['consensus_strength'])

            # Persistence bonus
            if persistence_analysis['persistence_score'] > 0.7:
                confidence_multipliers.append(1.1)

            # Breakout confirmation
            if breakout_analysis['breakout_potential'] > 0.6:
                confidence_multipliers.append(1.15)

            # Volume confirmation
            if volume_analysis['volume_momentum_score'] > 0.6:
                confidence_multipliers.append(1.1)

            # Structure alignment
            if ((signal_direction == 'buy'
                 and structure_analysis['structure_type'] == 'uptrend') or
                (signal_direction == 'sell'
                 and structure_analysis['structure_type'] == 'downtrend')):
                confidence_multipliers.append(1.1)

            # Calculate final confidence
            final_confidence = base_confidence
            for multiplier in confidence_multipliers:
                final_confidence *= multiplier

            final_confidence = max(0.3, min(0.95, final_confidence))

            return {
                'pair': pair,
                'signal_direction': signal_direction,
                'momentum_strength': momentum_strength,
                'confidence': final_confidence,
                'current_price': current_price,
                'indicators': momentum_indicators,
                'multitimeframe': mtf_momentum,
                'trend_analysis': trend_analysis,
                'persistence': persistence_analysis,
                'breakout': breakout_analysis,
                'structure': structure_analysis,
                'volume': volume_analysis,
                'bullish_score': bullish_score,
                'bearish_score': bearish_score
            }

        except Exception as e:
            logger.error(f"❌ Momentum analysis combination failed: {e}")
            return None

    def _categorize_momentum_strength(self, momentum_value):
        """Categorize momentum strength"""
        for strength, (min_val, max_val) in self.momentum_levels.items():
            if min_val <= momentum_value < max_val:
                return strength
        return 'weak'

    async def _create_momentum_signal(self, pair, momentum_analysis,
                                      market_conditions):
        """Create trading signal from momentum analysis"""
        try:
            if not momentum_analysis:
                return None

            current_price = momentum_analysis['current_price']
            direction = momentum_analysis['signal_direction']
            confidence = momentum_analysis['confidence']
            atr = momentum_analysis['indicators'].get('atr',
                                                      current_price * 0.02)

            # Calculate dynamic targets based on momentum strength and ATR
            momentum_strength = momentum_analysis['momentum_strength']

            # Target multipliers based on momentum strength
            target_multipliers = {
                'weak': 1.5,  # 1.5x ATR
                'moderate': 2.0,  # 2x ATR
                'strong': 3.0,  # 3x ATR
                'extreme': 4.0  # 4x ATR
            }

            target_multiplier = target_multipliers.get(momentum_strength, 2.0)
            stop_multiplier = 1.2  # 1.2x ATR for stop loss

            if direction == 'buy':
                target_price = current_price + (atr * target_multiplier)
                stop_loss = current_price - (atr * stop_multiplier)
            else:
                target_price = current_price - (atr * target_multiplier)
                stop_loss = current_price + (atr * stop_multiplier)

            # Calculate quality score for ranking
            quality_factors = [
                confidence,
                momentum_analysis['multitimeframe']['consensus_strength'],
                momentum_analysis['persistence']['persistence_score'],
                momentum_analysis['breakout']['breakout_potential'],
                momentum_analysis['volume']['volume_momentum_score']
            ]

            quality_score = np.mean(quality_factors)

            return {
                'pair':
                pair,
                'action':
                direction,
                'price':
                current_price,
                'target':
                target_price,
                'stop_loss':
                stop_loss,
                'confidence':
                confidence,
                'strategy':
                self.name,
                'momentum_strength':
                momentum_strength,
                'quality_score':
                quality_score,
                'analysis':
                momentum_analysis,
                'risk_reward_ratio':
                abs(target_price - current_price) /
                abs(current_price - stop_loss),
                'expected_holding_hours':
                self._estimate_holding_period(momentum_strength),
                'market_conditions':
                market_conditions,
                'timestamp':
                datetime.now()
            }

        except Exception as e:
            logger.error(f"❌ Momentum signal creation failed: {e}")
            return None

    def _estimate_holding_period(self, momentum_strength):
        """Estimate holding period based on momentum strength"""
        holding_periods = {
            'weak': 12,  # 12 hours
            'moderate': 24,  # 24 hours
            'strong': 48,  # 48 hours
            'extreme': 72  # 72 hours
        }
        return holding_periods.get(momentum_strength, 24)

    async def _validate_momentum_signal(self, signal, market_data):
        """Validate momentum signal with additional checks"""
        try:
            # Risk-reward ratio check
            if signal.get('risk_reward_ratio', 0) < 1.5:
                logger.debug(
                    f"❌ Poor risk-reward ratio for {signal['pair']}: {signal.get('risk_reward_ratio', 0):.2f}"
                )
                return False

            # Confidence threshold
            if signal['confidence'] < 0.5:
                return False

            # Quality score threshold
            if signal.get('quality_score', 0) < 0.4:
                return False

            # Check for recent signals (avoid overtrading)
            recent_signals = [
                s for s in self.signal_history
                if s['pair'] == signal['pair'] and (
                    datetime.now() - s['timestamp']).total_seconds() < 3600
            ]  # 1 hour

            if len(recent_signals) >= 2:  # Max 2 signals per hour per pair
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Momentum signal validation failed: {e}")
            return False

    def _rank_momentum_signals(self, signals):
        """Rank signals by quality and return top performers"""
        if not signals:
            return signals

        # Sort by quality score and confidence
        signals.sort(key=lambda s:
                     (s.get('quality_score', 0) * 0.6 + s['confidence'] * 0.4),
                     reverse=True)

        # Return top 3 signals maximum
        return signals[:3]

    def _track_signal_generation(self, signal):
        """Track signal generation for performance analysis"""
        self.momentum_performance['total_signals'] += 1

        # Add to signal history
        self.signal_history.append({
            'pair':
            signal['pair'],
            'action':
            signal['action'],
            'confidence':
            signal['confidence'],
            'momentum_strength':
            signal['momentum_strength'],
            'timestamp':
            signal['timestamp']
        })

        # Maintain history limit
        if len(self.signal_history) > self.max_signal_history:
            self.signal_history = self.signal_history[-self.
                                                      max_signal_history:]

        # Update strength performance tracking
        strength = signal['momentum_strength']
        if strength in self.momentum_performance[
                'momentum_strength_performance']:
            self.momentum_performance['momentum_strength_performance'][
                strength]['signals'] += 1

    def record_momentum_result(self, signal, success, actual_profit=0):
        """Record momentum trade result for learning"""
        try:
            if success:
                self.momentum_performance['successful_momentum_trades'] += 1

                # Update average profit
                current_avg = self.momentum_performance['avg_momentum_profit']
                success_count = self.momentum_performance[
                    'successful_momentum_trades']
                self.momentum_performance['avg_momentum_profit'] = (
                    (current_avg *
                     (success_count - 1)) + actual_profit) / success_count

                # Update strength-specific performance
                strength = signal.get('momentum_strength', 'moderate')
                if strength in self.momentum_performance[
                        'momentum_strength_performance']:
                    strength_perf = self.momentum_performance[
                        'momentum_strength_performance'][strength]
                    strength_perf['successful'] += 1

                    # Update strength average profit
                    current_strength_avg = strength_perf['avg_profit']
                    strength_success_count = strength_perf['successful']
                    strength_perf['avg_profit'] = (
                        (current_strength_avg * (strength_success_count - 1)) +
                        actual_profit) / strength_success_count

                logger.info(
                    f"✅ Momentum success: {signal['pair']} - {actual_profit:.2%} profit"
                )
            else:
                logger.warning(f"❌ Momentum failed: {signal['pair']}")

        except Exception as e:
            logger.error(f"❌ Failed to record momentum result: {e}")

    def get_strategy_performance(self):
        """Get comprehensive momentum strategy performance"""
        if self.momentum_performance['total_signals'] == 0:
            return {
                'total_signals': 0,
                'success_rate': 0,
                'avg_profit': 0,
                'best_momentum_strength': None,
                'strength_breakdown': {}
            }

        success_rate = (
            self.momentum_performance['successful_momentum_trades'] /
            self.momentum_performance['total_signals'])

        # Find best performing momentum strength
        best_strength = None
        best_performance = 0

        for strength, perf in self.momentum_performance[
                'momentum_strength_performance'].items():
            if perf['signals'] > 0:
                strength_success_rate = perf['successful'] / perf['signals']
                if strength_success_rate > best_performance:
                    best_performance = strength_success_rate
                    best_strength = strength

        return {
            'total_signals':
            self.momentum_performance['total_signals'],
            'success_rate':
            success_rate,
            'avg_profit':
            self.momentum_performance['avg_momentum_profit'],
            'best_momentum_strength':
            best_strength,
            'strength_breakdown':
            self.momentum_performance['momentum_strength_performance']
        }
