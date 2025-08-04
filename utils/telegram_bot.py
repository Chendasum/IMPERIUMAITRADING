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
            # Debug: Print config values (remove in production)
            logger.info(f"🔍 Debug - Bot token exists: {bool(self.bot_token)}")
            logger.info(f"🔍 Debug - Chat ID exists: {bool(self.chat_id)}")
            
            if self.bot_token and self.chat_id:
                # Test the connection first
                test_result = await self._test_connection()
                if test_result:
                    self.initialized = True
                    logger.info("✅ Telegram notifications initialized")
                    await self.send_message("🏛️ IMPERIUM AI TRADING SYSTEM - Notifications Active")
                else:
                    logger.error("❌ Telegram connection test failed")
                    self.initialized = False
            else:
                logger.warning("⚠️ Telegram credentials not configured - notifications disabled")
                self.initialized = False
                
        except Exception as e:
            logger.error(f"❌ Telegram initialization failed: {e}")
            self.initialized = False
    
    async def _test_connection(self):
        """Test Telegram bot connection"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                bot_info = response.json()
                if bot_info.get('ok'):
                    logger.info(f"✅ Bot connected: {bot_info['result']['first_name']}")
                    return True
                else:
                    logger.error(f"❌ Bot API error: {bot_info.get('description', 'Unknown error')}")
                    return False
            else:
                logger.error(f"❌ Bot connection failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Bot connection test failed: {e}")
            return False
    
    async def send_message(self, message):
        """Send message via Telegram"""
        if not self.initialized:
            logger.info(f"📢 NOTIFICATION: {message}")
            return False
            
        try:
            # Real Telegram API integration
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            # Handle different chat_id formats
            chat_id = self.chat_id
            if isinstance(chat_id, str) and chat_id.startswith('@'):
                # Channel username format
                chat_id_param = chat_id
            else:
                # Numeric chat ID
                try:
                    chat_id_param = int(chat_id)
                except:
                    chat_id_param = str(chat_id)
            
            payload = {
                'chat_id': chat_id_param,
                'text': message[:4096],  # Telegram message limit
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            logger.info(f"🔍 Sending to chat_id: {chat_id_param}")
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    logger.info(f"📱 TELEGRAM: Message sent successfully")
                    return True
                else:
                    logger.error(f"❌ Telegram API error: {result.get('description', 'Unknown error')}")
                    logger.info(f"📢 FALLBACK NOTIFICATION: {message}")
                    return False
            else:
                logger.error(f"❌ Telegram HTTP error: {response.status_code}")
                logger.error(f"❌ Response: {response.text}")
                logger.info(f"📢 FALLBACK NOTIFICATION: {message}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram message: {e}")
            logger.info(f"📢 FALLBACK NOTIFICATION: {message}")
            return False
    
    async def send_trade_alert(self, trade_data):
        """Send trade execution alert"""
        message = f"""
🏛️ *IMPERIUM TRADING ALERT*

💰 *TRADE EXECUTED:*
• Pair: `{trade_data.get('pair', 'N/A')}`
• Action: `{trade_data.get('action', 'N/A').upper()}`
• Strategy: `{trade_data.get('strategy', 'N/A')}`
• Price: `${trade_data.get('price', 0):.4f}`
• Size: `${trade_data.get('size', 0):.2f}`
• Expected Profit: `${trade_data.get('profit', 0):.2f}`

📊 Confidence: `{trade_data.get('confidence', 0)*100:.1f}%`
⏰ Time: `{trade_data.get('timestamp', 'N/A')}`
"""
        return await self.send_message(message)
    
    async def send_daily_report(self, report_data):
        """Send daily trading report"""
        message = f"""
📊 *DAILY TRADING REPORT*

💰 *PERFORMANCE:*
• Total Profit: `${report_data.get('total_profit', 0):.2f}`
• Daily Profit: `${report_data.get('daily_profit', 0):.2f}`
• Trades Today: `{report_data.get('trade_count', 0)}`
• Success Rate: `{report_data.get('success_rate', 0):.1f}%`

📈 *BALANCE:*
• Current: `${report_data.get('current_balance', 0):.2f}`
• Change: `{report_data.get('balance_change', 0):+.2f}%`

🎯 *STRATEGIES:*
• Arbitrage: `{report_data.get('arbitrage_trades', 0)} trades`
• Momentum: `{report_data.get('momentum_trades', 0)} trades`
• Mean Reversion: `{report_data.get('reversion_trades', 0)} trades`
"""
        return await self.send_message(message)
    
    async def send_error_alert(self, error_message):
        """Send error alert"""
        message = f"🚨 *IMPERIUM TRADING ERROR:*\n`{error_message}`"
        return await self.send_message(message)
    
    async def send_system_status(self, status_data):
        """Send system status update"""
        status = "✅ ACTIVE" if status_data.get('active', False) else "❌ INACTIVE"
        health = "✅ GOOD" if status_data.get('healthy', True) else "⚠️ WARNING"
        
        message = f"""
🏛️ *SYSTEM STATUS UPDATE*

⚡ *TRADING ENGINE:* `{status}`
📊 *Strategies Running:* `{', '.join(status_data.get('strategies', []))}`
💰 *Current Balance:* `${status_data.get('balance', 0):.2f}`
🎯 *Daily P&L:* `${status_data.get('daily_pnl', 0):+.2f}`

🔧 *SYSTEM HEALTH:* `{health}`
"""
        return await self.send_message(message)
