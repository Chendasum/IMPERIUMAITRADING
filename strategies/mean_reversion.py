import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ProfessionalMeanReversionStrategy:
    """Professional Mean Reversion Strategy with Advanced Statistical Analysis"""
    
    def __init__(self):
        self.name = "mean_reversion"
        self.exchange_manager = None
        self.risk_manager = None
        
        # Enhanced parameters for different market conditions
        self.base_periods = 20
        self.adaptive_periods = True  # Adjust periods based on volatility
        self.std_multipliers = {
            'conservative': 2.5,  # Wider bands for fewer signals
            'moderate': 2.0,      # Standard Bollinger Bands
            'aggressive': 1.5     # Tighter bands for more signals
        }
        
        # Multi-timeframe analysis
        self.primary_timeframe = '1h'
        self.confirmation_timeframes = ['4h', '1d']
        
        # Market regime detection
        self.trend_threshold = 0.02  # 2% trend threshold
        self.volatility_threshold = 0.03  # 3% volatility threshold
        
        # Enhanced mean reversion indicators
        self.rsi_oversold = 25
        self.rsi_overbought = 75
        self.stoch_oversold = 20
        self.stoch_overbought = 80
        
        # Risk management parameters
        self.max_holding_period = 72  # Max 72 hours for mean reversion
        self.profit_target_multiplier = 0.6  # Take profit at 60% to mean
        self.stop_loss_multiplier = 1.2  # Stop loss beyond band
        
        # Market condition tracking
        self.market_regime = 'neutral'  # trending, ranging, volatile
        self.volatility_regime = 'normal'  # low, normal, high
        
        # Performance tracking
        self.strategy_performance = {
            'total_signals': 0,
            'successful_trades': 0,
            'avg_holding_time': 0,
            'best_performing_pairs': {},
            'market_regime_performance': {}
        }
        
        # Asset-specific parameters
        self.asset_configs = {
            'BTC/USDT': {
                'volatility_adj': 1.2,
                'trend_sensitivity': 0.8,
                'min_volume': 10000000  # $10M daily volume
            },
            'ETH/USDT': {
                'volatility_adj': 1.3,
                'trend_sensitivity': 0.9,
                'min_volume': 5000000   # $5M daily volume
            },
            'BNB/USDT': {
                'volatility_adj': 1.4,
                'trend_sensitivity': 1.0,
                'min_volume': 1000000   # $1M daily volume
            },
            # Add more pairs as needed
        }
    
    def initialize(self, exchange_manager, risk_manager):
        """Initialize enhanced mean reversion strategy"""
        self.exchange_manager = exchange_manager
        self.risk_manager = risk_manager
        
        logger.info("✅ Professional Mean Reversion strategy initialized")
        logger.info(f"📊 Monitoring {len(self.asset_configs)} crypto pairs")
        logger.info(f"🎯 Market regime detection enabled")
    
    async def generate_signals(self):
        """Generate enhanced mean reversion trading signals"""
        signals = []
        
        try:
            # First, detect current market regime
            await self._detect_market_regime()
            
            # Get appropriate pairs based on market conditions
            active_pairs = self._get_suitable_pairs_for_regime()
            
            for pair in active_pairs:
                try:
                    # Get multi-timeframe data
                    market_data = await self._gather_market_data(pair)
                    if not market_data:
                        continue
                    
                    # Perform comprehensive mean reversion analysis
                    signal = await self._analyze_enhanced_mean_reversion(pair, market_data)
                    
                    if signal and await self._validate_mean_reversion_signal(signal):
                        signals.append(signal)
                        self.strategy_performance['total_signals'] += 1
                        logger.info(f"📊 Mean Reversion Signal: {pair} - {signal['action']} - Confidence: {signal['confidence']:.2%}")
                        
                except Exception as e:
                    logger.error(f"❌ Error analyzing {pair}: {e}")
                    continue
            
            # Sort signals by quality score
            signals.sort(key=lambda s: s.get('quality_score', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Enhanced mean reversion signal generation failed: {e}")
            
        return signals[:3]  # Return top 3 signals max
    
    async def _detect_market_regime(self):
        """Detect current market regime for strategy adaptation"""
        try:
            # Analyze major pairs to determine overall market condition
            btc_data = await self._get_timeframe_data('BTC/USDT', '1d', 30)
            if not btc_data:
                return
            
            closes = np.array([candle[4] for candle in btc_data])
            
            # Calculate trend strength
            recent_trend = (closes[-1] - closes[-10]) / closes[-10]  # 10-day trend
            medium_trend = (closes[-1] - closes[-20]) / closes[-20]  # 20-day trend
            
            # Calculate volatility
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns[-20:]) * np.sqrt(365)  # Annualized volatility
            
            # Determine regime
            if abs(recent_trend) > self.trend_threshold or abs(medium_trend) > self.trend_threshold:
                self.market_regime = 'trending'
            else:
                self.market_regime = 'ranging'
            
            if volatility > self.volatility_threshold:
                self.volatility_regime = 'high'
            elif volatility < self.volatility_threshold / 2:
                self.volatility_regime = 'low'
            else:
                self.volatility_regime = 'normal'
            
            logger.info(f"📊 Market Regime: {self.market_regime.title()}, Volatility: {self.volatility_regime.title()}")
            
        except Exception as e:
            logger.error(f"❌ Market regime detection failed: {e}")
    
    def _get_suitable_pairs_for_regime(self):
        """Get pairs suitable for current market regime"""
        all_pairs = list(self.asset_configs.keys())
        
        if self.market_regime == 'trending':
            # In trending markets, be more selective
            return all_pairs[:2]  # Focus on BTC and ETH
        elif self.volatility_regime == 'high':
            # In high volatility, avoid smaller pairs
            return ['BTC/USDT', 'ETH/USDT']
        else:
            # Normal conditions, use all pairs
            return all_pairs
    
    async def _gather_market_data(self, pair):
        """Gather comprehensive market data for analysis"""
        try:
            market_data = {}
            
            # Get primary timeframe data
            primary_data = await self._get_timeframe_data(
                pair, self.primary_timeframe, self.base_periods + 30
            )
            if not primary_data:
                return None
            
            market_data['primary'] = primary_data
            
            # Get confirmation timeframe data
            for tf in self.confirmation_timeframes:
                conf_data = await self._get_timeframe_data(pair, tf, 50)
                if conf_data:
                    market_data[tf] = conf_data
            
            # Get volume data for liquidity analysis
            market_data['volume_profile'] = self._analyze_volume_profile(primary_data)
            
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Failed to gather market data for {pair}: {e}")
            return None
    
    async def _get_timeframe_data(self, pair, timeframe, periods):
        """Get OHLCV data for specific timeframe"""
        try:
            exchange_name = list(self.exchange_manager.exchanges.keys())[0] if self.exchange_manager.exchanges else 'binance'
            
            ohlcv = await self.exchange_manager.get_ohlcv(
                exchange_name, pair, timeframe, periods
            )
            
            return ohlcv if len(ohlcv) >= 20 else None
            
        except Exception as e:
            logger.error(f"❌ Failed to get {timeframe} data for {pair}: {e}")
            return None
    
    def _analyze_volume_profile(self, ohlcv_data):
        """Analyze volume profile for liquidity assessment"""
        try:
            volumes = np.array([candle[5] for candle in ohlcv_data])
            prices = np.array([candle[4] for candle in ohlcv_data])
            
            avg_volume = np.mean(volumes[-20:])
            volume_trend = (volumes[-5:].mean() - volumes[-20:-5].mean()) / volumes[-20:-5].mean()
            
            return {
                'avg_volume': avg_volume,
                'volume_trend': volume_trend,
                'liquidity_score': min(avg_volume / 1000000, 1.0)  # Normalize to 0-1
            }
            
        except Exception as e:
            logger.error(f"❌ Volume profile analysis failed: {e}")
            return {'avg_volume': 0, 'volume_trend': 0, 'liquidity_score': 0}
    
    async def _analyze_enhanced_mean_reversion(self, pair, market_data):
        """Enhanced mean reversion analysis with multiple confirmations"""
        try:
            primary_data = market_data['primary']
            closes = np.array([candle[4] for candle in primary_data])
            highs = np.array([candle[2] for candle in primary_data])
            lows = np.array([candle[3] for candle in primary_data])
            volumes = np.array([candle[5] for candle in primary_data])
            
            current_price = closes[-1]
            
            # 1. Enhanced Bollinger Bands Analysis
            bb_analysis = self._calculate_enhanced_bollinger_bands(closes, pair)
            if not bb_analysis:
                return None
            
            # 2. Multiple Mean Reversion Indicators
            oscillator_analysis = self._calculate_oscillator_confluence(highs, lows, closes)
            
            # 3. Statistical Mean Reversion Tests
            statistical_analysis = self._perform_statistical_tests(closes)
            
            # 4. Multi-timeframe Confirmation
            mtf_confirmation = self._get_multitimeframe_confirmation(market_data, current_price)
            
            # 5. Volume Confirmation
            volume_confirmation = self._analyze_volume_confirmation(volumes, closes)
            
            # Combine all analyses
            signal = self._synthesize_mean_reversion_signal(
                pair, current_price, bb_analysis, oscillator_analysis,
                statistical_analysis, mtf_confirmation, volume_confirmation,
                market_data['volume_profile']
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Enhanced mean reversion analysis failed for {pair}: {e}")
            return None
    
    def _calculate_enhanced_bollinger_bands(self, closes, pair):
        """Calculate enhanced Bollinger Bands with regime adaptation"""
        try:
            # Adaptive period based on volatility
            if self.adaptive_periods:
                volatility = np.std(closes[-20:]) / np.mean(closes[-20:])
                if volatility > 0.05:  # High volatility
                    periods = max(10, self.base_periods - 5)
                elif volatility < 0.02:  # Low volatility
                    periods = min(30, self.base_periods + 5)
                else:
                    periods = self.base_periods
            else:
                periods = self.base_periods
            
            if len(closes) < periods:
                return None
            
            # Calculate multiple Bollinger Band sets
            std_mult = self._get_regime_adjusted_multiplier()
            
            sma = np.mean(closes[-periods:])
            std_dev = np.std(closes[-periods:])
            
            # Main bands
            upper_band = sma + (std_dev * std_mult)
            lower_band = sma - (std_dev * std_mult)
            
            # Inner bands for early signals
            inner_upper = sma + (std_dev * std_mult * 0.7)
            inner_lower = sma - (std_dev * std_mult * 0.7)
            
            # Calculate additional metrics
            current_price = closes[-1]
            bb_position = (current_price - lower_band) / (upper_band - lower_band)
            bb_squeeze = (upper_band - lower_band) / sma  # Band width
            
            # Price position analysis
            distance_from_mean = abs(current_price - sma) / sma
            z_score = (current_price - sma) / std_dev
            
            return {
                'sma': sma,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'inner_upper': inner_upper,
                'inner_lower': inner_lower,
                'bb_position': bb_position,
                'bb_squeeze': bb_squeeze,
                'distance_from_mean': distance_from_mean,
                'z_score': z_score,
                'periods': periods
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced Bollinger Bands calculation failed: {e}")
            return None
    
    def _get_regime_adjusted_multiplier(self):
        """Get standard deviation multiplier based on market regime"""
        if self.market_regime == 'trending':
            return self.std_multipliers['conservative']  # Wider bands in trends
        elif self.volatility_regime == 'high':
            return self.std_multipliers['conservative']  # Wider bands in high vol
        else:
            return self.std_multipliers['moderate']
    
    def _calculate_oscillator_confluence(self, highs, lows, closes):
        """Calculate multiple oscillators for confluence"""
        try:
            # RSI
            rsi = self._calculate_rsi(closes, 14)
            
            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(highs, lows, closes, 14)
            
            # Williams %R
            williams_r = self._calculate_williams_r(highs, lows, closes, 14)
            
            # CCI (Commodity Channel Index)
            cci = self._calculate_cci(highs, lows, closes, 20)
            
            # Determine oversold/overbought confluence
            oversold_signals = 0
            overbought_signals = 0
            
            if rsi < self.rsi_oversold:
                oversold_signals += 1
            elif rsi > self.rsi_overbought:
                overbought_signals += 1
            
            if stoch_k < self.stoch_oversold and stoch_d < self.stoch_oversold:
                oversold_signals += 1
            elif stoch_k > self.stoch_overbought and stoch_d > self.stoch_overbought:
                overbought_signals += 1
            
            if williams_r < -80:
                oversold_signals += 1
            elif williams_r > -20:
                overbought_signals += 1
            
            if cci < -100:
                oversold_signals += 1
            elif cci > 100:
                overbought_signals += 1
            
            return {
                'rsi': rsi,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d,
                'williams_r': williams_r,
                'cci': cci,
                'oversold_confluence': oversold_signals,
                'overbought_confluence': overbought_signals
            }
            
        except Exception as e:
            logger.error(f"❌ Oscillator confluence calculation failed: {e}")
            return None
    
    def _perform_statistical_tests(self, closes):
        """Perform statistical tests for mean reversion"""
        try:
            # Augmented Dickey-Fuller test for stationarity
            # (Simplified version - would use statsmodels in production)
            
            # Calculate returns
            returns = np.diff(closes) / closes[:-1]
            
            # Autocorrelation test
            lag1_corr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
            
            # Hurst Exponent (simplified calculation)
            hurst = self._calculate_hurst_exponent(closes)
            
            # Half-life of mean reversion
            half_life = self._calculate_half_life(closes)
            
            # Mean reversion strength
            reversion_strength = 1 - abs(lag1_corr)  # Higher when less autocorrelated
            
            return {
                'hurst_exponent': hurst,
                'half_life': half_life,
                'reversion_strength': reversion_strength,
                'autocorrelation': lag1_corr
            }
            
        except Exception as e:
            logger.error(f"❌ Statistical tests failed: {e}")
            return {'hurst_exponent': 0.5, 'half_life': 10, 'reversion_strength': 0.5, 'autocorrelation': 0}
    
    def _calculate_hurst_exponent(self, prices):
        """Calculate Hurst Exponent (simplified)"""
        try:
            lags = range(2, min(20, len(prices)//4))
            tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5  # Random walk
    
    def _calculate_half_life(self, prices):
        """Calculate half-life of mean reversion"""
        try:
            price_lag = prices[:-1]
            price_ret = prices[1:] - prices[:-1]
            price_lag_const = np.column_stack([price_lag, np.ones(len(price_lag))])
            
            beta = np.linalg.lstsq(price_lag_const, price_ret, rcond=None)[0][0]
            half_life = -np.log(2) / beta if beta < 0 else np.inf
            
            return min(half_life, 100) if half_life > 0 else 10
        except:
            return 10  # Default assumption
    
    def _get_multitimeframe_confirmation(self, market_data, current_price):
        """Get confirmation from higher timeframes"""
        try:
            confirmations = []
            
            for tf in self.confirmation_timeframes:
                if tf not in market_data:
                    continue
                
                tf_data = market_data[tf]
                tf_closes = np.array([candle[4] for candle in tf_data])
                
                # Check if higher timeframe shows mean reversion setup
                tf_sma = np.mean(tf_closes[-20:])
                tf_distance = abs(current_price - tf_sma) / tf_sma
                
                if tf_distance > 0.05:  # 5% away from higher TF mean
                    confirmations.append({
                        'timeframe': tf,
                        'mean': tf_sma,
                        'distance': tf_distance,
                        'confirming': True
                    })
            
            return {
                'total_confirmations': len(confirmations),
                'confirmations': confirmations,
                'mtf_strength': min(len(confirmations) / len(self.confirmation_timeframes), 1.0)
            }
            
        except Exception as e:
            logger.error(f"❌ Multi-timeframe confirmation failed: {e}")
            return {'total_confirmations': 0, 'confirmations': [], 'mtf_strength': 0}
    
    def _analyze_volume_confirmation(self, volumes, closes):
        """Analyze volume for mean reversion confirmation"""
        try:
            recent_volumes = volumes[-10:]
            recent_closes = closes[-10:]
            
            # Volume trend
            volume_trend = (recent_volumes[-3:].mean() - recent_volumes[:3].mean()) / recent_volumes[:3].mean()
            
            # Price-volume relationship
            price_changes = np.diff(recent_closes) / recent_closes[:-1]
            volume_changes = np.diff(recent_volumes) / recent_volumes[:-1]
            
            # Divergence detection (high volume on reversals is good)
            pv_correlation = np.corrcoef(abs(price_changes), volume_changes[:-1])[0, 1] if len(price_changes) > 1 else 0
            
            return {
                'volume_trend': volume_trend,
                'price_volume_correlation': pv_correlation,
                'volume_confirmation': volume_trend > 0.1  # Increasing volume
            }
            
        except Exception as e:
            logger.error(f"❌ Volume confirmation analysis failed: {e}")
            return {'volume_trend': 0, 'price_volume_correlation': 0, 'volume_confirmation': False}
    
    def _synthesize_mean_reversion_signal(self, pair, current_price, bb_analysis, 
                                        oscillator_analysis, statistical_analysis,
                                        mtf_confirmation, volume_confirmation, volume_profile):
        """Synthesize all analyses into a final signal"""
        try:
            # Check for mean reversion setup
            signal_type = None
            base_confidence = 0
            
            # Bollinger Band signals
            if current_price <= bb_analysis['lower_band']:
                if oscillator_analysis['oversold_confluence'] >= 2:  # At least 2 oscillators oversold
                    signal_type = 'buy'
                    base_confidence = 0.6
            elif current_price >= bb_analysis['upper_band']:
                if oscillator_analysis['overbought_confluence'] >= 2:  # At least 2 oscillators overbought
                    signal_type = 'sell'
                    base_confidence = 0.6
            
            if not signal_type:
                return None
            
            # Calculate enhanced confidence
            confidence_adjustments = []
            
            # Statistical strength
            if statistical_analysis['reversion_strength'] > 0.7:
                confidence_adjustments.append(0.1)
            
            # Multi-timeframe confirmation
            confidence_adjustments.append(mtf_confirmation['mtf_strength'] * 0.15)
            
            # Volume confirmation
            if volume_confirmation['volume_confirmation']:
                confidence_adjustments.append(0.1)
            
            # Market regime adjustment
            if self.market_regime == 'ranging':
                confidence_adjustments.append(0.1)  # Better for mean reversion
            elif self.market_regime == 'trending':
                confidence_adjustments.append(-0.1)  # Worse for mean reversion
            
            # Liquidity adjustment
            confidence_adjustments.append(volume_profile['liquidity_score'] * 0.05)
            
            final_confidence = base_confidence + sum(confidence_adjustments)
            final_confidence = max(0.3, min(0.9, final_confidence))
            
            # Calculate targets and stops
            if signal_type == 'buy':
                target = current_price + (bb_analysis['sma'] - current_price) * self.profit_target_multiplier
                stop_loss = bb_analysis['lower_band'] - (bb_analysis['upper_band'] - bb_analysis['lower_band']) * 0.1
            else:
                target = current_price - (current_price - bb_analysis['sma']) * self.profit_target_multiplier
                stop_loss = bb_analysis['upper_band'] + (bb_analysis['upper_band'] - bb_analysis['lower_band']) * 0.1
            
            # Quality score for ranking
            quality_score = (
                final_confidence * 0.4 +
                statistical_analysis['reversion_strength'] * 0.2 +
                mtf_confirmation['mtf_strength'] * 0.2 +
                volume_profile['liquidity_score'] * 0.2
            )
            
            return {
                'pair': pair,
                'action': signal_type,
                'price': current_price,
                'target': target,
                'stop_loss': stop_loss,
                'confidence': final_confidence,
                'strategy': self.name,
                'quality_score': quality_score,
                'analysis_details': {
                    'bollinger_bands': bb_analysis,
                    'oscillators': oscillator_analysis,
                    'statistics': statistical_analysis,
                    'multitimeframe': mtf_confirmation,
                    'volume': volume_confirmation
                },
                'expected_holding_hours': min(statistical_analysis['half_life'], self.max_holding_period),
                'risk_reward_ratio': abs(target - current_price) / abs(current_price - stop_loss),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Signal synthesis failed: {e}")
            return None
    
    async def _validate_mean_reversion_signal(self, signal):
        """Validate mean reversion signal with additional checks"""
        try:
            # Check risk-reward ratio
            if signal.get('risk_reward_ratio', 0) < 1.2:  # Minimum 1.2:1
                return False
            
            # Check confidence threshold
            if signal['confidence'] < 0.5:
                return False
            
            # Check market regime suitability
            if self.market_regime == 'trending' and signal['confidence'] < 0.7:
                return False  # Higher confidence needed in trending markets
            
            # Check if we have too many mean reversion positions
            # (This would integrate with position manager)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Signal validation failed: {e}")
            return False
    
    # Helper calculation methods
    def _calculate_rsi(self, prices, periods=14):
        """Calculate RSI"""
        try:
            if len(prices) < periods + 1:
                return 50
            
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
        except:
            return 50
    
    def _calculate_stochastic(self, highs, lows, closes, periods=14):
        """Calculate Stochastic Oscillator"""
        try:
            if len(closes) < periods:
                return 50, 50
            
            highest_high = np.max(highs[-periods:])
            lowest_low = np.min(lows[-periods:])
            
            if highest_high == lowest_low:
                return 50, 50
            
            k_percent = 100 * (closes[-1] - lowest_low) / (highest_high - lowest_low)
            d_percent = k_percent  # Simplified - would normally be 3-period SMA of %K
            
            return k_percent, d_percent
        except:
            return 50, 50
    
    def _calculate_williams_r(self, highs, lows, closes, periods=14):
        """Calculate Williams %R"""
        try:
            if len(closes) < periods:
                return -50
            
            highest_high = np.max(highs[-periods:])
            lowest_low = np.min(lows[-periods:])
            
            if highest_high == lowest_low:
                return -50
            
            williams_r = -100 * (highest_high - closes[-1]) / (highest_high - lowest_low)
            return williams_r
        except:
            return -50
    
    def _calculate_cci(self, highs, lows, closes, periods=20):
        """Calculate Commodity Channel Index"""
        try:
            if len(closes) < periods:
                return 0
            
            typical_prices = (highs + lows + closes) / 3
            sma_tp = np.mean(typical_prices[-periods:])
            mean_deviation = np.mean(np.abs(typical_prices[-periods:] - sma_tp))
            
            if mean_deviation == 0:
                return 0
            
            cci = (typical_prices[-1] - sma_tp) / (0.015 * mean_deviation)
            return cci
        except:
            return 0
    
    def get_strategy_performance(self):
        """Get comprehensive strategy performance metrics"""
        return self.strategy_performance
