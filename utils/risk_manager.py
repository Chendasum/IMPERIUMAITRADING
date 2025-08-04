import logging
from datetime import datetime, timedelta
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class RiskManager:

    def __init__(self):
        # Core balance tracking
        self.initial_balance = None
        self.current_balance = None
        self.peak_balance = None  # For drawdown calculation

        # Risk limits (industry standard)
        self.max_risk_per_trade = 0.01  # 1% per trade (institutional standard)
        self.max_portfolio_risk = 0.06  # 6% max portfolio risk
        self.daily_loss_limit = 0.03  # 3% daily loss limit
        self.max_drawdown_limit = 0.15  # 15% max drawdown

        # Position limits
        self.max_positions = 10  # Max concurrent positions
        self.max_correlation = 0.7  # Max correlation between positions
        self.max_sector_exposure = 0.25  # 25% max in one sector/asset class

        # Dynamic risk adjustment
        self.volatility_multiplier = 1.0  # Adjust risk based on market volatility
        self.consecutive_losses = 0  # Track consecutive losses
        self.max_consecutive_losses = 5  # Reduce size after 5 losses

        # Performance tracking
        self.daily_losses = 0.0
        self.daily_profits = 0.0
        self.trade_history = deque(maxlen=100)  # Last 100 trades
        self.daily_reset_time = datetime.now().date()

        # Advanced metrics
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.var_95 = 0.0  # Value at Risk 95%

        # Current positions tracking
        self.open_positions = {}
        self.position_correlations = {}

        self.metatrader_manager = None

    def initialize(self, initial_balance, metatrader_manager=None):
        """Initialize professional risk management system"""
        self.metatrader_manager = metatrader_manager
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance

        logger.info(f"🏛️ Professional Risk Management System Initialized")
        logger.info(f"💰 Initial Balance: ${initial_balance:,.2f}")
        logger.info(
            f"⚡ Max Risk Per Trade: {self.max_risk_per_trade*100:.1f}%")
        logger.info(f"🛡️ Max Daily Loss: {self.daily_loss_limit*100:.1f}%")
        logger.info(f"📉 Max Drawdown: {self.max_drawdown_limit*100:.1f}%")

    async def validate_trade(self, signal):
        """Advanced trade validation with multiple risk checks"""
        try:
            # Update real balance
            await self._update_real_balance()

            # 1. Basic balance and loss limit checks
            if not self._check_basic_limits():
                return False, "Basic risk limits exceeded"

            # 2. Drawdown protection
            if not self._check_drawdown_limits():
                return False, "Maximum drawdown exceeded"

            # 3. Position limits
            if not self._check_position_limits():
                return False, "Position limits exceeded"

            # 4. Correlation limits
            if not self._check_correlation_limits(signal):
                return False, "Correlation limits exceeded"

            # 5. Volatility-adjusted sizing
            if not self._check_volatility_limits(signal):
                return False, "Market volatility too high"

            # 6. Consecutive loss protection
            if not self._check_consecutive_losses():
                return False, "Too many consecutive losses"

            # 7. Kelly Criterion check
            if not self._kelly_criterion_check(signal):
                return False, "Kelly Criterion suggests no trade"

            logger.info("✅ Trade passed all professional risk checks")
            return True, "Trade approved"

        except Exception as e:
            logger.error(f"❌ Risk validation error: {e}")
            return False, "Risk validation failed"

    def calculate_position_size(self, signal):
        """Advanced position sizing using multiple methods"""
        try:
            # Method 1: Fixed fractional (base size)
            base_size = self._fixed_fractional_sizing(signal)

            # Method 2: Kelly Criterion adjustment
            kelly_adjustment = self._kelly_sizing_factor()

            # Method 3: Volatility adjustment
            vol_adjustment = self._volatility_adjustment(signal)

            # Method 4: Consecutive loss adjustment
            loss_adjustment = self._consecutive_loss_adjustment()

            # Method 5: Drawdown adjustment
            drawdown_adjustment = self._drawdown_adjustment()

            # Combine all methods
            final_size = base_size * kelly_adjustment * vol_adjustment * loss_adjustment * drawdown_adjustment

            # Apply hard limits
            max_position = self.current_balance * 0.1  # Never risk more than 10% on one trade
            final_size = min(final_size, max_position)

            logger.info(f"📊 Advanced Position Sizing:")
            logger.info(f"   Base: ${base_size:.2f}")
            logger.info(f"   Kelly: {kelly_adjustment:.2f}x")
            logger.info(f"   Volatility: {vol_adjustment:.2f}x")
            logger.info(f"   Loss Adj: {loss_adjustment:.2f}x")
            logger.info(f"   Final: ${final_size:.2f}")

            return max(0, final_size)

        except Exception as e:
            logger.error(f"❌ Position sizing error: {e}")
            return 0

    def _fixed_fractional_sizing(self, signal):
        """Basic fixed fractional position sizing"""
        risk_amount = self.current_balance * self.max_risk_per_trade
        stop_loss_pct = signal.get('stop_loss', 0.05)  # 5% default
        return risk_amount / stop_loss_pct

    def _kelly_sizing_factor(self):
        """Kelly Criterion for optimal position sizing"""
        if len(self.trade_history) < 10:
            return 0.5  # Conservative until we have data

        wins = [t for t in self.trade_history if t['profit'] > 0]
        losses = [t for t in self.trade_history if t['profit'] < 0]

        if not wins or not losses:
            return 0.5

        win_rate = len(wins) / len(self.trade_history)
        avg_win = np.mean([t['profit'] for t in wins])
        avg_loss = abs(np.mean([t['profit'] for t in losses]))

        if avg_loss == 0:
            return 0.5

        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss
        kelly_fraction = (b * win_rate - (1 - win_rate)) / b

        # Cap Kelly at 25% and make it positive
        return max(0.1, min(0.25, kelly_fraction))

    def _volatility_adjustment(self, signal):
        """Adjust position size based on market volatility"""
        volatility = signal.get('volatility', 0.02)  # Default 2%

        if volatility > 0.05:  # High volatility
            return 0.5
        elif volatility > 0.03:  # Medium volatility
            return 0.7
        else:  # Low volatility
            return 1.0

    def _consecutive_loss_adjustment(self):
        """Reduce size after consecutive losses"""
        if self.consecutive_losses >= 5:
            return 0.25  # 75% reduction
        elif self.consecutive_losses >= 3:
            return 0.5  # 50% reduction
        elif self.consecutive_losses >= 2:
            return 0.75  # 25% reduction
        return 1.0

    def _drawdown_adjustment(self):
        """Reduce size based on current drawdown"""
        current_drawdown = (self.peak_balance -
                            self.current_balance) / self.peak_balance

        if current_drawdown > 0.10:  # 10%+ drawdown
            return 0.5
        elif current_drawdown > 0.05:  # 5%+ drawdown
            return 0.75
        return 1.0

    def _check_basic_limits(self):
        """Check basic risk limits"""
        if self.daily_losses >= (self.initial_balance * self.daily_loss_limit):
            logger.warning("🛑 Daily loss limit reached")
            return False
        return True

    def _check_drawdown_limits(self):
        """Check maximum drawdown"""
        current_drawdown = (self.peak_balance -
                            self.current_balance) / self.peak_balance
        if current_drawdown >= self.max_drawdown_limit:
            logger.warning(
                f"🛑 Max drawdown exceeded: {current_drawdown*100:.1f}%")
            return False
        return True

    def _check_position_limits(self):
        """Check position count limits"""
        if len(self.open_positions) >= self.max_positions:
            logger.warning("🛑 Maximum positions reached")
            return False
        return True

    def _check_correlation_limits(self, signal):
        """Check correlation with existing positions"""
        # Simplified correlation check - in reality would use price correlations
        symbol = signal.get('pair', '')
        base_currency = symbol[:3] if len(symbol) >= 6 else symbol

        similar_positions = sum(
            1 for pos in self.open_positions.values()
            if pos.get('symbol', '').startswith(base_currency))

        if similar_positions >= 3:  # Max 3 positions in same base currency
            logger.warning("🛑 Too many correlated positions")
            return False
        return True

    def _check_volatility_limits(self, signal):
        """Check if market volatility is acceptable"""
        volatility = signal.get('volatility', 0.02)
        if volatility > 0.08:  # 8% volatility threshold
            logger.warning("🛑 Market volatility too high")
            return False
        return True

    def _check_consecutive_losses(self):
        """Check consecutive loss limits"""
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning("🛑 Too many consecutive losses - trading paused")
            return False
        return True

    def _kelly_criterion_check(self, signal):
        """Check if Kelly Criterion suggests trading"""
        kelly_factor = self._kelly_sizing_factor()
        return kelly_factor > 0.1  # Only trade if Kelly suggests >10%

    async def _update_real_balance(self):
        """Update balance from MetaTrader"""
        if self.metatrader_manager:
            balance_info = await self.metatrader_manager.get_account_balance()
            if balance_info and balance_info['balance'] > 0:
                self.current_balance = balance_info['balance']
                self.peak_balance = max(self.peak_balance,
                                        self.current_balance)

    def record_trade(self, trade_result):
        """Record trade for performance analysis"""
        trade_data = {
            'timestamp': datetime.now(),
            'symbol': trade_result.get('symbol'),
            'profit': trade_result.get('profit', 0),
            'size': trade_result.get('size', 0),
            'strategy': trade_result.get('strategy')
        }

        self.trade_history.append(trade_data)

        # Update consecutive losses
        if trade_data['profit'] < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Update daily tracking
        if trade_data['profit'] < 0:
            self.daily_losses += abs(trade_data['profit'])
        else:
            self.daily_profits += trade_data['profit']

        # Update balance
        self.current_balance += trade_data['profit']
        self.peak_balance = max(self.peak_balance, self.current_balance)

        # Calculate advanced metrics
        self._calculate_performance_metrics()

    def _calculate_performance_metrics(self):
        """Calculate advanced performance metrics"""
        if len(self.trade_history) < 10:
            return

        profits = [t['profit'] for t in self.trade_history]

        # Win rate
        wins = [p for p in profits if p > 0]
        self.win_rate = len(wins) / len(profits)

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum([p for p in profits if p < 0]))
        self.profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Sharpe ratio (simplified)
        if len(profits) > 1:
            returns_std = np.std(profits)
            avg_return = np.mean(profits)
            self.sharpe_ratio = avg_return / returns_std if returns_std > 0 else 0

        # Value at Risk 95%
        self.var_95 = np.percentile(profits, 5) if profits else 0

    def get_risk_report(self):
        """Get comprehensive risk report"""
        current_drawdown = (self.peak_balance -
                            self.current_balance) / self.peak_balance * 100

        return {
            'balance_metrics': {
                'current_balance':
                self.current_balance,
                'initial_balance':
                self.initial_balance,
                'peak_balance':
                self.peak_balance,
                'total_return_pct':
                ((self.current_balance - self.initial_balance) /
                 self.initial_balance) * 100,
                'current_drawdown_pct':
                current_drawdown
            },
            'risk_metrics': {
                'daily_losses': self.daily_losses,
                'daily_profits': self.daily_profits,
                'consecutive_losses': self.consecutive_losses,
                'open_positions': len(self.open_positions),
                'var_95': self.var_95
            },
            'performance_metrics': {
                'win_rate': self.win_rate * 100,
                'profit_factor': self.profit_factor,
                'sharpe_ratio': self.sharpe_ratio,
                'total_trades': len(self.trade_history)
            },
            'risk_limits': {
                'max_risk_per_trade_pct': self.max_risk_per_trade * 100,
                'daily_loss_limit_pct': self.daily_loss_limit * 100,
                'max_drawdown_limit_pct': self.max_drawdown_limit * 100,
                'max_positions': self.max_positions
            }
        }

    def get_risk_metrics(self):
        """Get current risk metrics for reporting"""
        return {
            'current_balance': self.current_balance or 10000,
            'peak_balance': self.peak_balance or 10000,
            'daily_profit': self.daily_profits,
            'daily_loss': self.daily_losses,
            'total_positions': len(self.open_positions),
            'max_risk_per_trade': self.max_risk_per_trade,
            'daily_loss_limit': self.daily_loss_limit,
            'max_drawdown_limit': self.max_drawdown_limit,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'var_95': self.var_95
        }
