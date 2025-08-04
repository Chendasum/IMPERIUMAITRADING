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
from utils.risk_manager import RiskManager
from utils.telegram_bot import TelegramNotifier
from strategies.arbitrage import ArbitrageStrategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingAISuperpower:

    def __init__(self):
        self.config = TradingConfig()
        self.exchange_manager = ExchangeManager()
        self.risk_manager = RiskManager()
        self.notifier = TelegramNotifier()

        # Initialize strategies
        self.strategies = {
            'arbitrage': ArbitrageStrategy(),
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy()
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

            # Initialize risk management
            self.risk_manager.initialize(self.config.INITIAL_BALANCE)
            logger.info("✅ Risk management system ready")

            # Initialize notification system
            await self.notifier.initialize()
            logger.info("✅ Telegram notifications ready")

            # Initialize strategies
            for name, strategy in self.strategies.items():
                strategy.initialize(self.exchange_manager, self.risk_manager)
                logger.info(f"✅ {name.title()} strategy initialized")

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

                # Run arbitrage strategy (lowest risk)
                arbitrage_signals = await self.strategies['arbitrage'
                                                          ].generate_signals()
                for signal in arbitrage_signals:
                    if signal['confidence'] > 0.8:  # High confidence only
                        await self.execute_trade(signal, 'arbitrage')

                # Run momentum strategy
                momentum_signals = await self.strategies['momentum'
                                                         ].generate_signals()
                for signal in momentum_signals:
                    if signal['confidence'] > 0.7:
                        await self.execute_trade(signal, 'momentum')

                # Run mean reversion strategy
                reversion_signals = await self.strategies['mean_reversion'
                                                          ].generate_signals()
                for signal in reversion_signals:
                    if signal['confidence'] > 0.75:
                        await self.execute_trade(signal, 'mean_reversion')

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
                return

            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(signal)

            if self.config.TRADING_MODE == 'paper':
                # Paper trading
                profit = await self.simulate_trade(signal, position_size)
                logger.info(
                    f"📊 PAPER TRADE: {signal['pair']} - {signal['action']} - Profit: ${profit:.2f}"
                )
            else:
                # Live trading
                profit = await self.execute_live_trade(signal, position_size)
                logger.info(
                    f"💰 LIVE TRADE: {signal['pair']} - {signal['action']} - Profit: ${profit:.2f}"
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

        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            await self.notifier.send_message(f"🚨 Trade execution failed: {e}")

    async def simulate_trade(self, signal, position_size):
        """Simulate trade for paper trading"""
        # Simulate trade execution with realistic slippage
        slippage = 0.001  # 0.1% slippage simulation

        if signal['action'] == 'buy':
            entry_price = signal['price'] * (1 + slippage)
            exit_price = signal['target'] * (1 - slippage)
        else:
            entry_price = signal['price'] * (1 - slippage)
            exit_price = signal['target'] * (1 + slippage)

        profit_percentage = (exit_price - entry_price) / entry_price
        profit_amount = position_size * profit_percentage

        return profit_amount

    async def execute_live_trade(self, signal, position_size):
        """Execute actual trade (implement when ready for live trading)"""
        # This would contain actual exchange API calls
        # For now, return simulated profit
        return await self.simulate_trade(signal, position_size)

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
