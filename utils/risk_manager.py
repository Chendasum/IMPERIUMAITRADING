import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self):
        self.initial_balance = None  # Will be fetched from real account
        self.current_balance = None  # Will be fetched from real account
        self.max_risk_per_trade = 0.02  # 2%
        self.daily_loss_limit = 0.05    # 5%
        self.daily_losses = 0.0
        self.daily_reset_time = datetime.now().date()
        self.metatrader_manager = None
        
    def initialize(self, initial_balance, metatrader_manager=None):
        """Initialize risk management with real account balance"""
        self.metatrader_manager = metatrader_manager
        if metatrader_manager:
            # Use real account balance from MetaTrader
            self.initial_balance = initial_balance
            self.current_balance = initial_balance
            logger.info(f"✅ Risk management initialized with REAL MetaTrader balance: ${initial_balance:,.2f}")
        else:
            # Fallback to provided balance
            self.initial_balance = initial_balance
            self.current_balance = initial_balance
            logger.info(f"⚠️ Risk management initialized with provided balance: ${initial_balance:,.2f}")
        
    async def validate_trade(self, signal):
        """Validate if trade passes risk management checks"""
        try:
            # Refresh real balance if MetaTrader connected
            if self.metatrader_manager:
                balance_info = await self.metatrader_manager.get_account_balance()
                if balance_info and balance_info['balance'] > 0:
                    self.current_balance = balance_info['balance']
                    logger.info(f"💰 Updated real balance: ${self.current_balance:.2f}")
            
            # Check daily loss limit
            if self.daily_losses >= (self.initial_balance * self.daily_loss_limit):
                logger.warning("🛑 Daily loss limit reached - trade rejected")
                return False
            
            # Check if we have sufficient balance
            position_size = self.calculate_position_size(signal)
            if position_size <= 0:
                logger.warning("💸 Insufficient balance for trade")
                return False
                
            # Check risk per trade
            max_loss = position_size * 0.05  # 5% max loss per trade
            if max_loss > (self.current_balance * self.max_risk_per_trade):
                logger.warning("⚠️ Trade exceeds maximum risk per trade")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Risk validation error: {e}")
            return False
    
    def calculate_position_size(self, signal):
        """Calculate appropriate position size based on risk management"""
        try:
            # Ensure we have valid balance values
            if self.current_balance is None or self.current_balance <= 0:
                logger.warning("❌ Invalid current balance for position sizing")
                return 0
                
            # Calculate maximum risk amount
            max_risk_amount = self.current_balance * self.max_risk_per_trade
            
            # Estimate potential loss (5% stop loss)
            estimated_loss_per_unit = signal['price'] * 0.05
            
            # Calculate position size
            if estimated_loss_per_unit > 0:
                position_size = max_risk_amount / estimated_loss_per_unit
                # Limit to available balance
                max_position = self.current_balance * 0.95  # Keep 5% as buffer
                position_size = min(position_size, max_position)
            else:
                position_size = 0
                
            logger.info(f"📊 Position size calculated: ${position_size:.2f}")
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"❌ Position size calculation error: {e}")
            return 0
    
    def check_daily_loss_limit(self):
        """Check if daily loss limit has been reached"""
        # Reset daily counters if new day
        current_date = datetime.now().date()
        if current_date > self.daily_reset_time:
            self.reset_daily_limits()
            
        if self.initial_balance:
            return self.daily_losses >= (self.initial_balance * self.daily_loss_limit)
        return False
    
    def update_balance(self, profit_loss):
        """Update balance and daily loss tracking"""
        self.current_balance += profit_loss
        
        if profit_loss < 0:
            self.daily_losses += abs(profit_loss)
            
        logger.info(f"💰 Balance updated: ${self.current_balance:.2f} (Change: ${profit_loss:+.2f})")
    
    def reset_daily_limits(self):
        """Reset daily loss tracking"""
        self.daily_losses = 0.0
        self.daily_reset_time = datetime.now().date()
        logger.info("🔄 Daily risk limits reset")
    
    def get_risk_metrics(self):
        """Get current risk management metrics"""
        return {
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'daily_losses': self.daily_losses,
            'daily_loss_limit': (self.initial_balance * self.daily_loss_limit) if self.initial_balance else 0,
            'max_risk_per_trade': self.max_risk_per_trade,
            'balance_change_pct': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100 if self.initial_balance and self.current_balance else 0
        }
