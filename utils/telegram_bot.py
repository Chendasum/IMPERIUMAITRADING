import asyncio
import logging
from telegram import Bot
from config import TradingConfig

logger = logging.getLogger(__name__)


class TelegramNotifier:

    def __init__(self):
        self.config = TradingConfig()
        self.bot = None

    async def initialize(self):
        """Initialize Telegram bot"""
        try:
            if self.config.TELEGRAM_BOT_TOKEN:
                self.bot = Bot(token=self.config.TELEGRAM_BOT_TOKEN)

                # Test connection
                await self.bot.get_me()
                logger.info("✅ Telegram bot initialized")

        except Exception as e:
            logger.error(f"❌ Telegram initialization failed: {e}")
            self.bot = None

    async def send_message(self, message):
        """Send notification message"""
        try:
            if self.bot and self.config.TELEGRAM_CHAT_ID:
                await self.bot.send_message(
                    chat_id=self.config.TELEGRAM_CHAT_ID,
                    text=message,
                    parse_mode='Markdown')
                logger.info(f"📱 Notification sent: {message[:50]}...")

        except Exception as e:
            logger.error(f"❌ Failed to send notification: {e}")

    async def send_daily_report(self, profit, trades, success_rate):
        """Send daily performance report"""
        message = f"""
🏛️ *DAILY TRADING REPORT*

💰 *Daily Profit:* ${profit:.2f}
📊 *Trades Executed:* {trades}
📈 *Success Rate:* {success_rate:.1f}%
⚡ *System Status:* Operational

🚀 Trading AI Superpower continuing market domination!
        """

        await self.send_message(message)
