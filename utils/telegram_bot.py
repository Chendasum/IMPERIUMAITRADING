import asyncio
import logging
import requests
from config import TradingConfig

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self):
        self.config = TradingConfig()
        self.bot_token = self.config.TELEGRAM_BOT_TOKEN
        self.chat_id = self.config.TELEGRAM_CHAT_ID
        self.initialized = False
        
    async def initialize(self):
        """Initialize Telegram bot for notifications"""
        try:
            if self.bot_token and self.chat_id:
                self.initialized = True
                logger.info("✅ Telegram notifications initialized")
                await self.send_message("🏛️ IMPERIUM AI TRADING SYSTEM - Notifications Active")
            else:
                logger.warning("⚠️ Telegram credentials not configured - notifications disabled")
                self.initialized = False
                
        except Exception as e:
            logger.error(f"❌ Telegram initialization failed: {e}")
            self.initialized = False
    
    async def send_message(self, message):
        """Send message via Telegram"""
        if not self.initialized:
            logger.info(f"📢 NOTIFICATION: {message}")
            return
            
        try:
            # Real Telegram API integration
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': str(self.chat_id),
                'text': message[:4096],  # Telegram message limit
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"📱 TELEGRAM: Message sent successfully")
            else:
                logger.error(f"❌ Telegram API error: {response.status_code}")
                logger.info(f"📢 FALLBACK NOTIFICATION: {message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram message: {e}")
            logger.info(f"📢 FALLBACK NOTIFICATION: {message}")
    
    async def send_trade_alert(self, trade_data):
        """Send trade execution alert"""
        message = f"""
🏛️ IMPERIUM TRADING ALERT

💰 TRADE EXECUTED:
• Pair: {trade_data.get('pair', 'N/A')}
• Action: {trade_data.get('action', 'N/A')}
• Strategy: {trade_data.get('strategy', 'N/A')}
• Price: ${trade_data.get('price', 0):.4f}
• Size: ${trade_data.get('size', 0):.2f}
• Expected Profit: ${trade_data.get('profit', 0):.2f}

📊 Confidence: {trade_data.get('confidence', 0)*100:.1f}%
⏰ Time: {trade_data.get('timestamp', 'N/A')}
"""
        await self.send_message(message)
    
    async def send_daily_report(self, report_data):
        """Send daily trading report"""
        message = f"""
📊 DAILY TRADING REPORT

💰 PERFORMANCE:
• Total Profit: ${report_data.get('total_profit', 0):.2f}
• Daily Profit: ${report_data.get('daily_profit', 0):.2f}
• Trades Today: {report_data.get('trade_count', 0)}
• Success Rate: {report_data.get('success_rate', 0):.1f}%

📈 BALANCE:
• Current: ${report_data.get('current_balance', 0):.2f}
• Change: {report_data.get('balance_change', 0):+.2f}%

🎯 STRATEGIES:
• Arbitrage: {report_data.get('arbitrage_trades', 0)} trades
• Momentum: {report_data.get('momentum_trades', 0)} trades
• Mean Reversion: {report_data.get('reversion_trades', 0)} trades
"""
        await self.send_message(message)
    
    async def send_error_alert(self, error_message):
        """Send error alert"""
        message = f"🚨 IMPERIUM TRADING ERROR:\n{error_message}"
        await self.send_message(message)
    
    async def send_system_status(self, status_data):
        """Send system status update"""
        message = f"""
🏛️ SYSTEM STATUS UPDATE

⚡ TRADING ENGINE: {'ACTIVE' if status_data.get('active', False) else 'INACTIVE'}
📊 Strategies Running: {', '.join(status_data.get('strategies', []))}
💰 Current Balance: ${status_data.get('balance', 0):.2f}
🎯 Daily P&L: ${status_data.get('daily_pnl', 0):+.2f}

🔧 SYSTEM HEALTH: {'GOOD' if status_data.get('healthy', True) else 'WARNING'}
"""
        await self.send_message(message)
