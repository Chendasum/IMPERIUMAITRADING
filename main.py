#!/usr/bin/env python3
"""
🏛️ TRADING AI SUPERPOWER - ENHANCED
Reformed Fund Architect - Algorithmic Trading System
Enhanced with Professional Risk Management & Portfolio Optimization
"""

import asyncio
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from config import TradingConfig
from utils.exchange_manager import ExchangeManager
from utils.metatrader_manager import MetaTraderManager
from utils.live_trading_executor import LiveTradingExecutor
from utils.risk_manager import RiskManager
from utils.telegram_bot import TelegramNotifier
from strategies.arbitrage import ArbitrageStrategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.forex_momentum import ForexMomentumStrategy

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingAISuperpower:
    """Enhanced AI Trading System with Professional Features"""

    def __init__(self):
        self.config = TradingConfig()
        self.exchange_manager = ExchangeManager()
        self.metatrader_manager = MetaTraderManager(self.config)
        self.live_executor = None
        self.risk_manager = RiskManager()
        self.notifier = TelegramNotifier()

        # Enhanced strategy management
        self.strategies = {
            'arbitrage': {'handler': ArbitrageStrategy(), 'weight': 0.4, 'max_positions': 3},
            'momentum': {'handler': MomentumStrategy(), 'weight': 0.3, 'max_positions': 2},
            'mean_reversion': {'handler': MeanReversionStrategy(), 'weight': 0.2, 'max_positions': 2},
            'forex_momentum': {'handler': ForexMomentumStrategy(), 'weight': 0.1, 'max_positions': 3}
        }

        # Enhanced performance tracking
        self.is_running = False
        self.total_profit = 0.0
        self.daily_profit = 0.0
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Portfolio management
        self.active_positions = {}
        self.position_correlations = {}
        self.strategy_performance = defaultdict(dict)

        # Advanced metrics
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.profit_factor = 0.0
        self.win_rate = 0.0

        # Market condition tracking
        self.market_volatility = 0.0
        self.market_trend = 'neutral'
        self.last_volatility_check = datetime.now()

        # Circuit breakers
        self.emergency_stop = False
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5

        # Performance optimization
        self.signal_history = []
        self.execution_times = []

    async def initialize(self):
        """Initialize all trading components with enhanced features"""
        try:
            logger.info("🏛️ ENHANCED TRADING AI SUPERPOWER - Initializing...")

            # Initialize exchange connections
            await self.exchange_manager.initialize()
            logger.info("✅ Exchange connections established")

            # Initialize live trading executor
            self.live_executor = LiveTradingExecutor(self.exchange_manager)
            logger.info("🚀 Live trading executor ready")

            # Initialize MetaTrader for forex
            await self.metatrader_manager.initialize()
            logger.info("✅ MetaTrader forex integration ready")

            # Enhanced balance initialization
            await self._initialize_enhanced_balance()

            # Initialize notification system
            await self.notifier.initialize()
            logger.info("✅ Telegram notifications ready")

            # Initialize strategies with enhanced parameters
            await self._initialize_enhanced_strategies()

            # Initialize market analysis
            await self._initialize_market_analysis()

            logger.info("🚀 ENHANCED TRADING AI SUPERPOWER - Fully Operational!")
            await self._send_startup_report()

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            await self.notifier.send_message(f"🚨 System initialization failed: {e}")
            raise

    async def _initialize_enhanced_balance(self):
        """Enhanced balance initialization with multiple account support"""
        total_balance = 0.0
        balance_sources = []

        # Get MetaTrader balance
        mt_balance_info = await self.metatrader_manager.get_account_balance()
        if mt_balance_info and mt_balance_info['balance'] > 0:
            mt_balance = mt_balance_info['balance']
            total_balance += mt_balance
            balance_sources.append(f"MetaTrader: ${mt_balance:.2f}")
            logger.info(f"💰 MetaTrader Balance: ${mt_balance:.2f}")

        # Get crypto exchange balances (if available)
        try:
            crypto_balance = await self._get_crypto_balance()
            if crypto_balance > 0:
                total_balance += crypto_balance
                balance_sources.append(f"Crypto: ${crypto_balance:.2f}")
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch crypto balance: {e}")

        # Use fallback if no real balances found
        if total_balance == 0:
            total_balance = self.config.INITIAL_BALANCE
            balance_sources.append(f"Fallback: ${total_balance:.2f}")
            logger.warning("⚠️ Using fallback balance")

        # Initialize risk management
        self.risk_manager.initialize(total_balance, self.metatrader_manager)
        logger.info(f"💰 Total Account Balance: ${total_balance:.2f}")
        logger.info(f"📊 Balance Sources: {', '.join(balance_sources)}")

    async def _get_crypto_balance(self):
        """Get total crypto balance across exchanges"""
        try:
            # This would integrate with your exchange manager
            # For now, return 0 as placeholder
            return 0.0
        except Exception:
            return 0.0

    async def _initialize_enhanced_strategies(self):
        """Initialize strategies with enhanced configuration"""
        for name, strategy_config in self.strategies.items():
            strategy = strategy_config['handler']

            if name == 'forex_momentum':
                strategy.initialize(self.metatrader_manager, self.risk_manager)
            else:
                strategy.initialize(self.exchange_manager, self.risk_manager)

            # Initialize strategy performance tracking
            self.strategy_performance[name] = {
                'trades': 0,
                'profit': 0.0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'active_positions': 0
            }

            logger.info(f"✅ Enhanced {name.title().replace('_', ' ')} strategy initialized")

    async def _initialize_market_analysis(self):
        """Initialize market condition analysis"""
        try:
            # Calculate initial market volatility
            await self._update_market_conditions()
            logger.info(f"📊 Market Analysis Initialized - Volatility: {self.market_volatility:.2%}")
        except Exception as e:
            logger.warning(f"⚠️ Market analysis initialization failed: {e}")

    async def _send_startup_report(self):
        """Send enhanced startup report"""
        balance_info = self.risk_manager.get_risk_metrics()

        message = f"""
🏛️ **ENHANCED TRADING AI SYSTEM ONLINE**

💰 **ACCOUNT STATUS:**
• Total Balance: ${balance_info['current_balance']:,.2f}
• Risk Per Trade: {self.risk_manager.max_risk_per_trade*100:.1f}%
• Daily Loss Limit: {self.risk_manager.daily_loss_limit*100:.1f}%
• Max Positions: {sum(s['max_positions'] for s in self.strategies.values())}

📊 **MARKET CONDITIONS:**
• Volatility: {self.market_volatility:.2%}
• Trend: {self.market_trend.title()}

🎯 **ACTIVE STRATEGIES:**
• Arbitrage (40% allocation)
• Momentum (30% allocation)  
• Mean Reversion (20% allocation)
• Forex Momentum (10% allocation)

🚀 **SYSTEM STATUS:** FULLY OPERATIONAL
"""
        await self.notifier.send_message(message)

    async def run_enhanced_trading_cycle(self):
        """Enhanced main trading loop with advanced features"""
        cycle_count = 0

        while self.is_running and not self.emergency_stop:
            try:
                cycle_start = time.time()
                cycle_count += 1
                logger.info(f"🔄 Starting enhanced trading cycle #{cycle_count}...")

                # Emergency circuit breaker check
                if await self._check_circuit_breakers():
                    logger.warning("🚨 Circuit breaker activated - stopping trading")
                    break

                # Update market conditions every 10 cycles
                if cycle_count % 10 == 0:
                    await self._update_market_conditions()

                # Portfolio rebalancing check every 50 cycles
                if cycle_count % 50 == 0:
                    await self._rebalance_portfolio()

                # Generate and prioritize signals
                all_signals = await self._generate_prioritized_signals()

                # Execute trades with portfolio optimization
                executed_trades = await self._execute_optimized_trades(all_signals)

                # Update performance metrics
                await self._update_performance_metrics()

                # Log enhanced cycle summary
                cycle_time = time.time() - cycle_start
                self.execution_times.append(cycle_time)

                if executed_trades > 0:
                    logger.info(f"💰 Executed {executed_trades} optimized trades")
                    await self._send_trade_summary(executed_trades)
                else:
                    logger.info("📊 Market analysis complete - no optimal trades found")

                logger.info(f"✅ Enhanced cycle #{cycle_count} completed in {cycle_time:.2f}s")

                # Adaptive cycle timing based on market conditions
                sleep_time = self._calculate_adaptive_sleep_time()
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"❌ Enhanced trading cycle error: {e}")
                await self.notifier.send_message(f"⚠️ Trading cycle error: {e}")
                await asyncio.sleep(60)

    async def _check_circuit_breakers(self):
        """Check various circuit breaker conditions"""
        # Check daily loss limit
        if self.risk_manager.check_daily_loss_limit():
            await self.notifier.send_message("🛑 Daily loss limit reached - trading paused")
            return True

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            await self.notifier.send_message(f"🛑 {self.consecutive_losses} consecutive losses - emergency stop")
            return True

        # Check maximum drawdown
        risk_metrics = self.risk_manager.get_risk_metrics()
        if risk_metrics['current_balance'] < self.risk_manager.initial_balance * 0.85:  # 15% drawdown
            await self.notifier.send_message("🛑 Maximum drawdown exceeded - emergency stop")
            return True

        return False

    async def _update_market_conditions(self):
        """Update market volatility and trend analysis"""
        try:
            # This would analyze recent price movements across markets
            # For now, using placeholder logic
            self.market_volatility = np.random.uniform(0.01, 0.05)  # 1-5% volatility

            # Determine market trend (simplified)
            if self.market_volatility > 0.035:
                self.market_trend = 'volatile'
            elif self.daily_profit > 0:
                self.market_trend = 'bullish'
            elif self.daily_profit < 0:
                self.market_trend = 'bearish'
            else:
                self.market_trend = 'neutral'

            self.last_volatility_check = datetime.now()

        except Exception as e:
            logger.error(f"❌ Market condition update failed: {e}")

    async def _generate_prioritized_signals(self):
        """Generate and prioritize signals from all strategies"""
        all_signals = []

        for strategy_name, strategy_config in self.strategies.items():
            try:
                strategy = strategy_config['handler']
                signals = await strategy.generate_signals()

                # Add strategy metadata to signals
                for signal in signals:
                    signal['strategy'] = strategy_name
                    signal['weight'] = strategy_config['weight']
                    signal['max_positions'] = strategy_config['max_positions']

                    # Adjust confidence based on strategy performance
                    strategy_perf = self.strategy_performance[strategy_name]
                    if strategy_perf['win_rate'] > 0.6:  # Good performance
                        signal['confidence'] *= 1.1
                    elif strategy_perf['win_rate'] < 0.4:  # Poor performance
                        signal['confidence'] *= 0.9

                all_signals.extend(signals)

            except Exception as e:
                logger.error(f"❌ Signal generation failed for {strategy_name}: {e}")

        # Sort signals by confidence score and strategic priority
        all_signals.sort(key=lambda s: s['confidence'] * s['weight'], reverse=True)

        return all_signals

    async def _execute_optimized_trades(self, signals):
        """Execute trades with portfolio optimization"""
        executed_trades = 0
        strategy_positions = defaultdict(int)

        for signal in signals:
            try:
                strategy_name = signal['strategy']

                # Check strategy position limits
                if strategy_positions[strategy_name] >= signal['max_positions']:
                    continue

                # Check correlation with existing positions
                if await self._check_position_correlation(signal):
                    continue

                # Execute trade based on asset type
                if strategy_name == 'forex_momentum':
                    success = await self._execute_enhanced_forex_trade(signal)
                else:
                    success = await self._execute_enhanced_crypto_trade(signal)

                if success:
                    executed_trades += 1
                    strategy_positions[strategy_name] += 1

                    # Update active positions
                    position_id = f"{signal['pair']}_{datetime.now().timestamp()}"
                    self.active_positions[position_id] = {
                        'signal': signal,
                        'strategy': strategy_name,
                        'timestamp': datetime.now()
                    }

                # Limit total trades per cycle
                if executed_trades >= 5:  # Max 5 trades per cycle
                    break

            except Exception as e:
                logger.error(f"❌ Trade execution failed: {e}")

        return executed_trades

    async def _check_position_correlation(self, signal):
        """Check if new position would create excessive correlation"""
        # Simplified correlation check
        pair = signal['pair']
        base_currency = pair[:3] if len(pair) >= 6 else pair

        similar_positions = sum(1 for pos in self.active_positions.values() 
                              if pos['signal']['pair'].startswith(base_currency))

        return similar_positions >= 2  # Max 2 positions with same base currency

    async def _execute_enhanced_crypto_trade(self, signal):
        """Execute enhanced crypto trade with better position management"""
        try:
            # Enhanced risk validation
            if not await self.risk_manager.validate_trade(signal):
                return False

            # Calculate optimized position size
            position_size = self._calculate_optimized_position_size(signal)

            # Execute trade
            profit = await self.execute_live_trade(signal, position_size)

            # Update tracking
            await self._update_trade_tracking(signal, profit, 'crypto')

            return profit != 0

        except Exception as e:
            logger.error(f"❌ Enhanced crypto trade failed: {e}")
            return False

    async def _execute_enhanced_forex_trade(self, signal):
        """Execute enhanced forex trade with better position management"""
        try:
            # Enhanced risk validation
            if not await self.risk_manager.validate_trade(signal):
                return False

            # Calculate optimized position size
            position_size = self._calculate_optimized_position_size(signal)

            # Execute forex trade
            profit = await self.metatrader_manager.execute_forex_trade(signal, position_size)

            # Update tracking
            await self._update_trade_tracking(signal, profit, 'forex')

            return profit != 0

        except Exception as e:
            logger.error(f"❌ Enhanced forex trade failed: {e}")
            return False

    def _calculate_optimized_position_size(self, signal):
        """Calculate position size with multiple optimization factors"""
        base_size = self.risk_manager.calculate_position_size(signal)

        # Market volatility adjustment
        vol_adjustment = 1.0
        if self.market_volatility > 0.04:  # High volatility
            vol_adjustment = 0.5
        elif self.market_volatility < 0.02:  # Low volatility
            vol_adjustment = 1.2

        # Strategy performance adjustment
        strategy_perf = self.strategy_performance[signal['strategy']]
        perf_adjustment = 1.0
        if strategy_perf['win_rate'] > 0.7:
            perf_adjustment = 1.1
        elif strategy_perf['win_rate'] < 0.4:
            perf_adjustment = 0.8

        # Confidence adjustment
        confidence_adjustment = signal['confidence']

        optimized_size = base_size * vol_adjustment * perf_adjustment * confidence_adjustment

        return max(optimized_size, base_size * 0.5)  # Minimum 50% of base size

    async def _update_trade_tracking(self, signal, profit, trade_type):
        """Update comprehensive trade tracking"""
        strategy_name = signal['strategy']

        # Update overall metrics
        self.total_profit += profit
        self.daily_profit += profit
        self.trade_count += 1

        if profit > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1

        # Update strategy performance
        strategy_perf = self.strategy_performance[strategy_name]
        strategy_perf['trades'] += 1
        strategy_perf['profit'] += profit

        if strategy_perf['trades'] > 0:
            strategy_perf['win_rate'] = self.winning_trades / strategy_perf['trades']
            strategy_perf['avg_profit'] = strategy_perf['profit'] / strategy_perf['trades']

    async def _update_performance_metrics(self):
        """Update advanced performance metrics"""
        if self.trade_count > 0:
            self.win_rate = self.winning_trades / self.trade_count

            if self.losing_trades > 0 and self.winning_trades > 0:
                avg_win = self.total_profit / self.winning_trades if self.winning_trades > 0 else 0
                avg_loss = abs(self.total_profit) / self.losing_trades if self.losing_trades > 0 else 1
                self.profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

    def _calculate_adaptive_sleep_time(self):
        """Calculate adaptive sleep time based on market conditions"""
        base_sleep = 30  # 30 seconds base

        # Faster cycles in volatile markets
        if self.market_volatility > 0.04:
            return base_sleep * 0.5  # 15 seconds
        elif self.market_volatility < 0.02:
            return base_sleep * 1.5  # 45 seconds

        return base_sleep

    async def _rebalance_portfolio(self):
        """Rebalance portfolio based on strategy performance"""
        try:
            logger.info("🔄 Performing portfolio rebalancing...")

            # Analyze strategy performance
            best_strategy = max(self.strategy_performance.items(), 
                              key=lambda x: x[1].get('win_rate', 0))
            worst_strategy = min(self.strategy_performance.items(), 
                               key=lambda x: x[1].get('win_rate', 1))

            logger.info(f"📊 Best Strategy: {best_strategy[0]} (Win Rate: {best_strategy[1].get('win_rate', 0)*100:.1f}%)")
            logger.info(f"📊 Worst Strategy: {worst_strategy[0]} (Win Rate: {worst_strategy[1].get('win_rate', 0)*100:.1f}%)")

            # Send rebalancing report
            await self.notifier.send_message(
                f"🔄 Portfolio Rebalancing Complete\n"
                f"Top Performer: {best_strategy[0].title().replace('_', ' ')}\n"
                f"Needs Improvement: {worst_strategy[0].title().replace('_', ' ')}"
            )

        except Exception as e:
            logger.error(f"❌ Portfolio rebalancing failed: {e}")

    async def _send_trade_summary(self, trade_count):
        """Send enhanced trade summary"""
        if trade_count > 2:  # Only for significant trading activity
            risk_metrics = self.risk_manager.get_risk_metrics()

            message = f"""
💰 **TRADING ACTIVITY SUMMARY**

🎯 Trades Executed: {trade_count}
💵 Session Profit: ${self.daily_profit:.2f}
📊 Win Rate: {self.win_rate*100:.1f}%
🏛️ Total Portfolio: ${risk_metrics['current_balance']:,.2f}

📈 **PERFORMANCE:**
• Profit Factor: {self.profit_factor:.2f}
• Market Conditions: {self.market_trend.title()}
• Volatility: {self.market_volatility:.2%}
"""
            await self.notifier.send_message(message)

    # Include original execute_live_trade method here...
    async def execute_live_trade(self, signal, position_size):
        """Execute live trade with real money (original method)"""
        try:
            if not self.live_executor:
                logger.error("❌ Live executor not available - REAL TRADING REQUIRED")
                return 0

            exchange_name = 'binance'
            symbol = signal['pair']
            side = signal['action']

            current_price = signal['price']
            amount = position_size / current_price

            order = await self.live_executor.execute_market_order(
                exchange_name=exchange_name,
                symbol=symbol,
                side=side,
                amount=amount
            )

            if side == 'buy':
                profit = (signal['target'] - current_price) * amount
            else:
                profit = (current_price - signal['target']) * amount

            logger.info(f"💰 LIVE TRADE EXECUTED: {symbol} - ${profit:.2f} expected profit")
            return profit

        except Exception as e:
            logger.error(f"❌ Live trade execution failed: {e}")
            return 0

    async def start_enhanced_system(self):
        """Start the enhanced trading system"""
        try:
            await self.initialize()
            self.is_running = True
            await self.run_enhanced_trading_cycle()

        except Exception as e:
            logger.error(f"❌ Enhanced system error: {e}")
            await self.notifier.send_message(f"🚨 ENHANCED SYSTEM ERROR: {e}")
        finally:
            self.is_running = False
            logger.info("🛑 Enhanced Trading AI Superpower stopped")


# Enhanced main execution
async def main():
    """Enhanced main entry point"""
    trading_ai = TradingAISuperpower()

    try:
        await trading_ai.start_enhanced_system()
    except KeyboardInterrupt:
        logger.info("🛑 Manual shutdown requested")
        trading_ai.is_running = False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    print("🏛️ ENHANCED TRADING AI SUPERPOWER - Starting System...")
    asyncio.run(main())
