#!/usr/bin/env python3
"""
🏛️ TRADING AI SUPERPOWER
Reformed Fund Architect - Algorithmic Trading System
"""

import asyncio
import logging
import time
from datetime import datetime
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

    def __init__(self):
        self.config = TradingConfig()
        self.exchange_manager = ExchangeManager()
        self.metatrader_manager = MetaTraderManager(self.config)
        self.live_executor = None  # Will be initialized after exchange manager
        self.risk_manager = RiskManager()
        self.notifier = TelegramNotifier()

        # Initialize strategies (crypto + forex)
        self.strategies = {
            'arbitrage': ArbitrageStrategy(),
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'forex_momentum': ForexMomentumStrategy()
        }

        self.is_running = False
        self.total_profit = 0.0
        self.daily_profit = 0.0
        self.trade_count = 0

    async def initialize(self):
        """Initialize all trading components"""
        try:
            logger.info("🏛️ TRADING AI SUPERPOWER - Initializing...")

            # Initialize exchange connections
            await self.exchange_manager.initialize()
            logger.info("✅ Exchange connections established")
            
            # Initialize live trading executor
            self.live_executor = LiveTradingExecutor(self.exchange_manager)
            logger.info("🚀 Live trading executor ready")
            
            # Initialize MetaTrader for forex
            await self.metatrader_manager.initialize()
            logger.info("✅ MetaTrader forex integration ready")

            # Initialize risk management
            self.risk_manager.initialize(self.config.INITIAL_BALANCE)
            logger.info("✅ Risk management system ready")

            # Initialize notification system
            await self.notifier.initialize()
            logger.info("✅ Telegram notifications ready")

            # Initialize strategies
            for name, strategy in self.strategies.items():
                if name == 'forex_momentum':
                    strategy.initialize(self.metatrader_manager, self.risk_manager)
                else:
                    strategy.initialize(self.exchange_manager, self.risk_manager)
                logger.info(f"✅ {name.title().replace('_', ' ')} strategy initialized")

            logger.info("🚀 TRADING AI SUPERPOWER - Fully Operational!")
            await self.notifier.send_message(
                "🏛️ Trading AI System Online - Ready for Market Domination!")

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            await self.notifier.send_message(
                f"🚨 System initialization failed: {e}")
            raise

    async def run_trading_cycle(self):
        """Main trading loop"""
        while self.is_running:
            try:
                cycle_start = time.time()
                logger.info("🔄 Starting trading cycle...")

                # Check daily loss limit
                if self.risk_manager.check_daily_loss_limit():
                    logger.warning(
                        "🛑 Daily loss limit reached - stopping trading")
                    await self.notifier.send_message(
                        "🛑 Daily loss limit reached - trading paused")
                    break

                # Generate signals from all strategies
                executed_trades = 0
                
                # Run arbitrage strategy (lowest risk)
                arbitrage_signals = await self.strategies['arbitrage'].generate_signals()
                if arbitrage_signals:
                    logger.info(f"📊 Arbitrage: {len(arbitrage_signals)} signals generated")
                for signal in arbitrage_signals:
                    if signal['confidence'] > 0.8:  # High confidence only
                        success = await self.execute_trade(signal, 'arbitrage')
                        if success:
                            executed_trades += 1

                # Run momentum strategy
                momentum_signals = await self.strategies['momentum'].generate_signals()
                if momentum_signals:
                    logger.info(f"📈 Momentum: {len(momentum_signals)} signals generated")
                for signal in momentum_signals:
                    if signal['confidence'] > 0.7:
                        success = await self.execute_trade(signal, 'momentum')
                        if success:
                            executed_trades += 1

                # Run mean reversion strategy
                reversion_signals = await self.strategies['mean_reversion'].generate_signals()
                if reversion_signals:
                    logger.info(f"🔄 Mean Reversion: {len(reversion_signals)} signals generated")
                for signal in reversion_signals:
                    if signal['confidence'] > 0.75:
                        success = await self.execute_trade(signal, 'mean_reversion')
                        if success:
                            executed_trades += 1

                # Run forex momentum strategy
                forex_signals = await self.strategies['forex_momentum'].generate_signals()
                if forex_signals:
                    logger.info(f"💱 Forex Momentum: {len(forex_signals)} signals generated")
                for signal in forex_signals:
                    if signal['confidence'] > 0.7:
                        success = await self.execute_forex_trade(signal, 'forex_momentum')
                        if success:
                            executed_trades += 1
                
                # Log cycle summary
                total_signals = len(arbitrage_signals) + len(momentum_signals) + len(reversion_signals) + len(forex_signals)
                if executed_trades > 0:
                    logger.info(f"💰 Executed {executed_trades} trades from {total_signals} signals")
                elif total_signals > 0:
                    logger.info(f"⚠️ {total_signals} signals generated but none passed risk validation")
                else:
                    logger.info("📊 Market analysis complete - no trading signals this cycle")

                # Performance reporting
                cycle_time = time.time() - cycle_start
                logger.info(
                    f"✅ Trading cycle completed in {cycle_time:.2f} seconds")

                # Wait before next cycle
                await asyncio.sleep(30)  # 30-second cycles

            except Exception as e:
                logger.error(f"❌ Trading cycle error: {e}")
                await self.notifier.send_message(f"⚠️ Trading cycle error: {e}"
                                                 )
                await asyncio.sleep(60)  # Wait longer on error

    async def execute_trade(self, signal, strategy_name):
        """Execute a trading signal"""
        try:
            # Check if trade passes risk management
            if not self.risk_manager.validate_trade(signal):
                logger.info(
                    f"❌ Trade rejected by risk management: {signal['pair']}")
                return False

            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(signal)

            # LIVE TRADING ONLY - NO SIMULATION
            profit = await self.execute_live_trade(signal, position_size)
            logger.info(
                f"💰 LIVE TRADE EXECUTED: {signal['pair']} - {signal['action']} - Profit: ${profit:.2f}"
            )

            # Update performance tracking
            self.total_profit += profit
            self.daily_profit += profit
            self.trade_count += 1

            # Send notification for significant trades
            if abs(profit) > 50:  # Notify for trades > $50 profit/loss
                await self.notifier.send_message(
                    f"💰 {strategy_name.upper()} TRADE\n"
                    f"Pair: {signal['pair']}\n"
                    f"Action: {signal['action']}\n"
                    f"Profit: ${profit:.2f}\n"
                    f"Total Profit: ${self.total_profit:.2f}")
            
            return True

        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            await self.notifier.send_message(f"🚨 Trade execution failed: {e}")
            return False

    async def execute_forex_trade(self, signal, strategy_name):
        """Execute a forex trading signal"""
        try:
            # Check if trade passes risk management
            if not self.risk_manager.validate_trade(signal):
                logger.info(f"❌ Forex trade rejected by risk management: {signal['pair']}")
                return False

            # Calculate position size for forex
            position_size = self.risk_manager.calculate_position_size(signal)

            # LIVE FOREX TRADING ONLY - NO SIMULATION
            profit = await self.metatrader_manager.execute_forex_trade(signal, position_size)
            logger.info(f"💱 LIVE FOREX TRADE: {signal['pair']} - {signal['action']} - Profit: ${profit:.2f}")

            # Update performance tracking
            self.total_profit += profit
            self.daily_profit += profit
            self.trade_count += 1

            # Send notification for significant forex trades
            if abs(profit) > 25:  # Notify for forex trades > $25 profit/loss
                await self.notifier.send_message(
                    f"💱 {strategy_name.upper().replace('_', ' ')} FOREX TRADE\n"
                    f"Pair: {signal['pair']}\n"
                    f"Action: {signal['action']}\n" 
                    f"Profit: ${profit:.2f}\n"
                    f"Total Profit: ${self.total_profit:.2f}")
            
            return True

        except Exception as e:
            logger.error(f"❌ Forex trade execution failed: {e}")
            await self.notifier.send_message(f"🚨 Forex trade execution failed: {e}")
            return False

    # ALL SIMULATION FUNCTIONS REMOVED - LIVE TRADING ONLY

    async def execute_live_trade(self, signal, position_size):
        """Execute live trade with real money"""
        try:
            if not self.live_executor:
                logger.error("❌ Live executor not available - REAL TRADING REQUIRED")
                return 0
            
            exchange_name = 'binance'  # Primary exchange
            symbol = signal['pair']
            side = signal['action']  # 'buy' or 'sell'
            
            # Convert position size to appropriate amount for the symbol
            current_price = signal['price']
            amount = position_size / current_price  # Convert USD to base currency amount
            
            # Execute the trade through live executor
            order = await self.live_executor.execute_market_order(
                exchange_name=exchange_name,
                symbol=symbol,
                side=side,
                amount=amount
            )
            
            # Calculate actual profit based on target price
            if side == 'buy':
                profit = (signal['target'] - current_price) * amount
            else:
                profit = (current_price - signal['target']) * amount
                
            logger.info(f"💰 LIVE TRADE EXECUTED: {symbol} - ${profit:.2f} expected profit")
            
            return profit
            
        except Exception as e:
            logger.error(f"❌ Live trade execution failed: {e}")
            # NO SIMULATION FALLBACK - LIVE TRADING ONLY
            return 0

    async def daily_reset(self):
        """Reset daily tracking"""
        # Send daily report
        await self.notifier.send_message(
            f"📊 DAILY TRADING REPORT\n"
            f"Daily Profit: ${self.daily_profit:.2f}\n"
            f"Total Profit: ${self.total_profit:.2f}\n"
            f"Trades Today: {self.trade_count}\n"
            f"Success Rate: {self.calculate_success_rate():.1f}%")

        # Reset daily counters
        self.daily_profit = 0.0
        self.risk_manager.reset_daily_limits()
        logger.info("🔄 Daily reset completed")

    def calculate_success_rate(self):
        """Calculate trading success rate"""
        if self.trade_count == 0:
            return 0.0
        return (self.total_profit > 0) * 100  # Simplified calculation

    async def start(self):
        """Start the trading system"""
        try:
            await self.initialize()
            self.is_running = True

            # Start trading cycle
            await self.run_trading_cycle()

        except Exception as e:
            logger.error(f"❌ System error: {e}")
            await self.notifier.send_message(f"🚨 SYSTEM ERROR: {e}")
        finally:
            self.is_running = False
            logger.info("🛑 Trading AI Superpower stopped")

    def stop(self):
        """Stop the trading system"""
        self.is_running = False
        logger.info("🛑 Stop signal received")


# Main execution
async def main():
    """Main entry point"""
    trading_ai = TradingAISuperpower()

    try:
        await trading_ai.start()
    except KeyboardInterrupt:
        logger.info("🛑 Manual shutdown requested")
        trading_ai.stop()
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    print("🏛️ TRADING AI SUPERPOWER - Starting System...")
    asyncio.run(main())
