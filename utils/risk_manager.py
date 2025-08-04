
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self):
        self.initial_balance = 10000
        self.current_balance = 10000
        self.max_risk_per_trade = 0.02  # 2%
        self.daily_loss_limit = 0.05    # 5%
        self.daily_loss = 0.0
        self.max_position_size = 0.1    # 10% max position
        
    def initialize(self, initial_balance):
        """Initialize risk management with starting balance"""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        logger.info(f"💰 Risk Manager initialized with ${initial_balance:,.2f}")
        
    def validate_trade(self, signal):
        """Validate if trade meets risk criteria"""
        try:
            # Check if we have enough balance
            min_trade_size = self.current_balance * 0.01  # 1% minimum
            if self.current_balance < min_trade_size:
                logger.warning("⚠️ Insufficient balance for trading")
                return False
            
            # Check daily loss limit
            if self.daily_loss >= (self.initial_balance * self.daily_loss_limit):
                logger.warning("⚠️ Daily loss limit reached")
                return False
            
            # Additional validation checks can be added here
            return True
            
        except Exception as e:
            logger.error(f"❌ Risk validation error: {e}")
            return False
    
    def calculate_position_size(self, signal):
        """Calculate appropriate position size based on risk"""
        try:
            # Calculate position size based on risk per trade
            risk_amount = self.current_balance * self.max_risk_per_trade
            
            # Use confidence to adjust position size
            confidence_multiplier = signal.get('confidence', 0.5)
            adjusted_risk = risk_amount * confidence_multiplier
            
            # Ensure position doesn't exceed maximum
            max_position = self.current_balance * self.max_position_size
            position_size = min(adjusted_risk, max_position)
            
            logger.info(f"📊 Position size calculated: ${position_size:.2f}")
            return position_size
            
        except Exception as e:
            logger.error(f"❌ Position size calculation error: {e}")
            return self.current_balance * 0.01  # Default to 1%
    
    def update_balance(self, profit_loss):
        """Update balance after trade"""
        self.current_balance += profit_loss
        if profit_loss < 0:
            self.daily_loss += abs(profit_loss)
        
        logger.info(f"💰 Balance updated: ${self.current_balance:.2f}")
    
    def check_daily_loss_limit(self):
        """Check if daily loss limit is reached"""
        return self.daily_loss >= (self.initial_balance * self.daily_loss_limit)
    
    def reset_daily_limits(self):
        """Reset daily tracking (call at start of new day)"""
        self.daily_loss = 0.0
        logger.info("🔄 Daily risk limits reset")
    
    def get_risk_metrics(self):
        """Get current risk metrics"""
        return {
            'current_balance': self.current_balance,
            'daily_loss': self.daily_loss,
            'daily_loss_pct': (self.daily_loss / self.initial_balance) * 100,
            'remaining_risk': (self.initial_balance * self.daily_loss_limit) - self.daily_loss
        }
