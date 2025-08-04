import os
import requests
import logging
from config import TradingConfig

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """Minimal, reliable Telegram notifier for real trading systems"""

    def __init__(self):
        self.config = TradingConfig()
        self.bot_token = self.config.TELEGRAM_BOT_TOKEN
        self.chat_id = self.config.TELEGRAM_CHAT_ID
        self.initialized = False

    async def initialize(self):
        """Test Telegram bot connection and enable notifications"""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials missing — notifications disabled.")
            return

        if await self._test_connection():
            self.initialized = True
            logger.info("Telegram bot connected and ready.")
            await self.send_message("IMPERIUM TRADING SYSTEM ACTIVE")
        else:
            logger.error("Telegram connection failed.")
            self.initialized = False

    async def _test_connection(self):
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and response.json().get("ok"):
                return True
        except Exception as e:
            logger.error(f"Telegram test connection error: {e}")
        return False

    async def send_message(self, message: str):
        """Send a plain-text Telegram message"""
        if not self.initialized:
            logger.debug(f"[DEBUG] Telegram not initialized. Message skipped:\n{message}")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message[:4096],
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200 and response.json().get("ok"):
                logger.info("Telegram message sent.")
                return True
            else:
                logger.warning(f"Telegram API failed: {response.text}")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

        return False

    async def send_trade_alert(self, trade):
        """Send a trade execution alert"""
        if not trade.get("executed", True):
            return False

        msg = (
            f"TRADE EXECUTED:\n"
            f"Pair: {trade.get('pair')}\n"
            f"Action: {trade.get('action').upper()}\n"
            f"Strategy: {trade.get('strategy')}\n"
            f"Price: ${trade.get('price', 0):.4f}\n"
            f"Size: ${trade.get('size', 0):.2f}\n"
            f"Profit Target: ${trade.get('profit', 0):.2f}\n"
            f"Confidence: {trade.get('confidence', 0) * 100:.1f}%\n"
            f"Time: {trade.get('timestamp')}"
        )
        return await self.send_message(msg)

    async def send_daily_report(self, report):
        """Send daily performance summary"""
        msg = (
            f"DAILY REPORT:\n"
            f"Total Profit: ${report.get('total_profit', 0):.2f}\n"
            f"Daily Profit: ${report.get('daily_profit', 0):.2f}\n"
            f"Trades Today: {report.get('trade_count', 0)}\n"
            f"Success Rate: {report.get('success_rate', 0):.1f}%\n\n"
            f"Current Balance: ${report.get('current_balance', 0):.2f}\n"
            f"Balance Change: {report.get('balance_change', 0):+.2f}%\n\n"
            f"Arbitrage: {report.get('arbitrage_trades', 0)} | "
            f"Momentum: {report.get('momentum_trades', 0)} | "
            f"Reversion: {report.get('reversion_trades', 0)}"
        )
        return await self.send_message(msg)

    async def send_error_alert(self, error_text):
        """Send a system error alert"""
        return await self.send_message(f"SYSTEM ERROR:\n{error_text}")

    async def send_system_status(self, status):
        """Send basic system status update"""
        msg = (
            f"SYSTEM STATUS:\n"
            f"Trading Engine: {'Active' if status.get('active') else 'Inactive'}\n"
            f"Strategies: {', '.join(status.get('strategies', []))}\n"
            f"Balance: ${status.get('balance', 0):.2f}\n"
            f"Daily PnL: ${status.get('daily_pnl', 0):+.2f}\n"
            f"Health: {'Healthy' if status.get('healthy') else 'Warning'}"
        )
        return await self.send_message(msg)
