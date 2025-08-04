import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


@dataclass
class TradeExecution:
    """Trade execution result with comprehensive data"""
    execution_id: str
    timestamp: datetime
    exchange: str
    symbol: str
    side: str
    amount: float
    price: float
    fee: float
    order_id: str
    status: OrderStatus
    execution_time_ms: float
    slippage: float
    market_impact: float


@dataclass
class Position:
    """Trading position with risk management"""
    position_id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    entry_timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_drawdown: float = 0.0
    max_profit: float = 0.0


class ProfessionalLiveTradingExecutor:
    """Professional Live Trading Executor with Advanced Risk Management"""

    def __init__(self, exchange_manager):
        self.exchange_manager = exchange_manager

        # Position and order management
        self.open_positions: Dict[str, Position] = {}
        self.open_orders: Dict[str, Dict] = {}
        self.trade_history: List[TradeExecution] = []
        self.execution_queue = asyncio.Queue()

        # Risk management parameters
        self.max_position_size = 10000  # Max $10,000 per position
        self.max_total_exposure = 50000  # Max $50,000 total exposure
        self.max_daily_loss = 2500  # Max $2,500 daily loss
        self.position_size_limit_pct = 0.1  # Max 10% of balance per position

        # Execution parameters
        self.slippage_tolerance = 0.005  # 0.5% max acceptable slippage
        self.execution_timeout = 30  # 30 seconds max execution time
        self.partial_fill_timeout = 60  # 60 seconds for partial fills
        self.max_retries = 3

        # Smart order routing
        self.exchange_preferences = {
            'binance': {
                'priority': 1,
                'fee_tier': 0.001
            },  # 0.1% fees
            'coinbase': {
                'priority': 2,
                'fee_tier': 0.005
            },  # 0.5% fees
            'bybit': {
                'priority': 3,
                'fee_tier': 0.001
            }  # 0.1% fees
        }

        # Performance tracking
        self.execution_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_execution_time': 0.0,
            'avg_slippage': 0.0,
            'total_fees_paid': 0.0,
            'daily_pnl': 0.0,
            'max_drawdown': 0.0
        }

        # Risk monitoring
        self.daily_losses = 0.0
        self.daily_reset_time = datetime.now().date()
        self.emergency_stop = False

        # Execution monitoring
        self.execution_monitor_task = None
        self.position_monitor_task = None

    async def initialize(self):
        """Initialize the professional trading executor"""
        try:
            logger.info("🚀 Initializing Professional Trading Executor...")

            # Start background monitoring tasks
            self.execution_monitor_task = asyncio.create_task(
                self._monitor_executions())
            self.position_monitor_task = asyncio.create_task(
                self._monitor_positions())

            # Load existing positions if any
            await self._load_existing_positions()

            logger.info("✅ Professional Trading Executor initialized")

        except Exception as e:
            logger.error(f"❌ Trading executor initialization failed: {e}")
            raise

    async def execute_smart_order(
            self,
            symbol: str,
            side: str,
            amount: float,
            order_type: OrderType = OrderType.MARKET,
            price: Optional[float] = None,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None,
            strategy: str = "manual") -> Optional[TradeExecution]:
        """Execute order with smart routing and risk management"""
        try:
            # Generate unique execution ID
            execution_id = str(uuid.uuid4())
            start_time = time.time()

            logger.info(f"🎯 Smart Order Execution Started: {execution_id}")
            logger.info(f"   Symbol: {symbol}, Side: {side}, Amount: {amount}")

            # Pre-execution risk checks
            if not await self._pre_execution_risk_check(
                    symbol, side, amount, price):
                logger.error("❌ Pre-execution risk check failed")
                return None

            # Smart exchange selection
            best_exchange = await self._select_best_exchange(
                symbol, side, amount, order_type)
            if not best_exchange:
                logger.error("❌ No suitable exchange found")
                return None

            # Execute with enhanced error handling
            execution_result = await self._execute_with_monitoring(
                best_exchange, symbol, side, amount, order_type, price,
                execution_id)

            if execution_result:
                # Post-execution processing
                await self._post_execution_processing(execution_result,
                                                      stop_loss, take_profit,
                                                      strategy)

                # Update metrics
                execution_time = (time.time() - start_time) * 1000
                self._update_execution_metrics(execution_result,
                                               execution_time)

                logger.info(
                    f"✅ Smart Order Executed: {execution_id} in {execution_time:.0f}ms"
                )
                return execution_result
            else:
                logger.error(f"❌ Smart Order Failed: {execution_id}")
                return None

        except Exception as e:
            logger.error(f"❌ Smart order execution failed: {e}")
            return None

    async def _pre_execution_risk_check(self, symbol: str, side: str,
                                        amount: float,
                                        price: Optional[float]) -> bool:
        """Comprehensive pre-execution risk validation"""
        try:
            # Check emergency stop
            if self.emergency_stop:
                logger.error("🛑 Emergency stop active - blocking all trades")
                return False

            # Check daily loss limits
            if self._check_daily_loss_limit():
                logger.error("🛑 Daily loss limit exceeded")
                return False

            # Estimate position value
            estimated_price = price if price else await self._get_market_price(
                symbol)
            if not estimated_price:
                logger.error("❌ Could not determine market price")
                return False

            position_value = amount * estimated_price

            # Check position size limits
            if position_value > self.max_position_size:
                logger.error(
                    f"❌ Position size ${position_value:.2f} exceeds limit ${self.max_position_size}"
                )
                return False

            # Check total exposure
            current_exposure = self._calculate_total_exposure()
            if current_exposure + position_value > self.max_total_exposure:
                logger.error(
                    f"❌ Total exposure would exceed limit: ${current_exposure + position_value:.2f}"
                )
                return False

            # Check account balance
            available_balance = await self._get_available_balance()
            required_margin = position_value * 1.1  # 10% buffer

            if side == 'buy' and available_balance < required_margin:
                logger.error(
                    f"❌ Insufficient balance: ${available_balance:.2f} < ${required_margin:.2f}"
                )
                return False

            # Check for position concentration
            if not self._check_position_concentration(symbol, position_value):
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Pre-execution risk check failed: {e}")
            return False

    async def _select_best_exchange(self, symbol: str, side: str,
                                    amount: float,
                                    order_type: OrderType) -> Optional[str]:
        """Select best exchange using smart routing algorithm"""
        try:
            available_exchanges = self.exchange_manager.get_connected_exchanges(
            )
            if not available_exchanges:
                return None

            exchange_scores = {}

            for exchange_name in available_exchanges:
                try:
                    # Get current market data
                    prices = await self.exchange_manager.get_prices(
                        exchange_name, [symbol])
                    if not prices or symbol not in prices:
                        continue

                    price_data = prices[symbol]

                    # Calculate exchange score based on multiple factors
                    score = await self._calculate_exchange_score(
                        exchange_name, symbol, side, amount, price_data,
                        order_type)

                    if score > 0:
                        exchange_scores[exchange_name] = score

                except Exception as e:
                    logger.warning(f"⚠️ Error evaluating {exchange_name}: {e}")
                    continue

            if not exchange_scores:
                return None

            # Select exchange with highest score
            best_exchange = max(exchange_scores.keys(),
                                key=lambda k: exchange_scores[k])
            logger.info(
                f"🎯 Selected exchange: {best_exchange} (score: {exchange_scores[best_exchange]:.2f})"
            )

            return best_exchange

        except Exception as e:
            logger.error(f"❌ Exchange selection failed: {e}")
            return None

    async def _calculate_exchange_score(self, exchange_name: str, symbol: str,
                                        side: str, amount: float,
                                        price_data: Dict,
                                        order_type: OrderType) -> float:
        """Calculate exchange score for smart routing"""
        try:
            score = 100.0  # Base score

            # Factor 1: Liquidity (bid-ask spread)
            bid = price_data.get('bid', 0)
            ask = price_data.get('ask', 0)
            if bid and ask:
                spread = (ask - bid) / bid
                spread_penalty = spread * 1000  # Penalize wide spreads
                score -= spread_penalty

            # Factor 2: Exchange priority and fees
            exchange_config = self.exchange_preferences.get(exchange_name, {})
            priority_bonus = (4 - exchange_config.get('priority', 4)) * 10
            fee_penalty = exchange_config.get('fee_tier', 0.001) * 1000
            score += priority_bonus - fee_penalty

            # Factor 3: Volume (for market impact)
            volume = price_data.get('volume', 0)
            if volume > 0:
                market_impact = (amount *
                                 (ask if side == 'buy' else bid)) / volume
                impact_penalty = min(market_impact * 100,
                                     50)  # Cap at 50 points
                score -= impact_penalty

            # Factor 4: Recent execution performance
            recent_performance = self._get_recent_exchange_performance(
                exchange_name)
            score += recent_performance * 20

            # Factor 5: Order type suitability
            if order_type == OrderType.MARKET:
                # Prefer exchanges with tight spreads for market orders
                score += (1 - min(spread, 0.01)) * 10

            return max(0, score)

        except Exception as e:
            logger.error(f"❌ Exchange score calculation failed: {e}")
            return 0

    async def _execute_with_monitoring(
            self, exchange_name: str, symbol: str, side: str, amount: float,
            order_type: OrderType, price: Optional[float],
            execution_id: str) -> Optional[TradeExecution]:
        """Execute order with comprehensive monitoring"""
        try:
            exchange = self.exchange_manager.exchanges.get(exchange_name)
            if not exchange:
                raise Exception(f"Exchange {exchange_name} not available")

            # Get pre-execution price for slippage calculation
            pre_execution_price = await self._get_market_price(
                symbol, exchange_name)

            # Execute the order based on type
            if order_type == OrderType.MARKET:
                order_result = await self._execute_market_order_with_monitoring(
                    exchange, symbol, side, amount, execution_id)
            elif order_type == OrderType.LIMIT:
                order_result = await self._execute_limit_order_with_monitoring(
                    exchange, symbol, side, amount, price, execution_id)
            else:
                raise Exception(f"Unsupported order type: {order_type}")

            if not order_result:
                return None

            # Calculate execution metrics
            execution_price = order_result.get('average',
                                               order_result.get('price', 0))
            slippage = self._calculate_slippage(pre_execution_price,
                                                execution_price, side)
            market_impact = self._calculate_market_impact(
                amount, execution_price, symbol)

            # Create execution record
            execution = TradeExecution(
                execution_id=execution_id,
                timestamp=datetime.now(),
                exchange=exchange_name,
                symbol=symbol,
                side=side,
                amount=order_result.get('filled', amount),
                price=execution_price,
                fee=order_result.get('fee', {}).get('cost', 0),
                order_id=order_result.get('id', ''),
                status=OrderStatus.FILLED,
                execution_time_ms=0,  # Will be set by caller
                slippage=slippage,
                market_impact=market_impact)

            # Add to trade history
            self.trade_history.append(execution)

            return execution

        except Exception as e:
            logger.error(f"❌ Order execution with monitoring failed: {e}")
            self.execution_metrics['failed_executions'] += 1
            return None

    async def _execute_market_order_with_monitoring(
            self, exchange, symbol: str, side: str, amount: float,
            execution_id: str) -> Optional[Dict]:
        """Execute market order with enhanced monitoring"""
        try:
            # Add to monitoring queue
            await self.execution_queue.put({
                'execution_id': execution_id,
                'type': 'market_order',
                'start_time': time.time()
            })

            # Execute with timeout
            order_task = asyncio.create_task(
                exchange.create_market_order(symbol, side, amount))

            try:
                order_result = await asyncio.wait_for(
                    order_task, timeout=self.execution_timeout)
                logger.info(
                    f"✅ Market order executed: {order_result.get('id', 'N/A')}"
                )
                return order_result

            except asyncio.TimeoutError:
                logger.error(
                    f"⏰ Market order timeout for execution {execution_id}")
                order_task.cancel()
                return None

        except Exception as e:
            logger.error(f"❌ Market order execution failed: {e}")
            return None

    async def _execute_limit_order_with_monitoring(
            self, exchange, symbol: str, side: str, amount: float,
            price: float, execution_id: str) -> Optional[Dict]:
        """Execute limit order with monitoring and partial fill handling"""
        try:
            # Place initial limit order
            order_result = await exchange.create_limit_order(
                symbol, side, amount, price)
            order_id = order_result.get('id')

            if not order_id:
                return None

            # Monitor for fills
            start_time = time.time()
            check_interval = 2.0  # Check every 2 seconds

            while time.time() - start_time < self.partial_fill_timeout:
                # Check order status
                order_status = await exchange.fetch_order(order_id, symbol)

                if order_status['status'] == 'closed':
                    logger.info(f"✅ Limit order filled: {order_id}")
                    return order_status
                elif order_status['status'] == 'canceled':
                    logger.warning(f"⚠️ Limit order canceled: {order_id}")
                    return None

                # Check for partial fills
                filled_amount = order_status.get('filled', 0)
                if filled_amount > 0:
                    logger.info(
                        f"📊 Limit order partially filled: {filled_amount}/{amount}"
                    )

                await asyncio.sleep(check_interval)

            # Timeout reached - cancel remaining order
            logger.warning(f"⏰ Limit order timeout, canceling: {order_id}")
            await exchange.cancel_order(order_id, symbol)

            # Return partial fill if any
            final_status = await exchange.fetch_order(order_id, symbol)
            if final_status.get('filled', 0) > 0:
                return final_status

            return None

        except Exception as e:
            logger.error(f"❌ Limit order execution failed: {e}")
            return None

    async def _post_execution_processing(self, execution: TradeExecution,
                                         stop_loss: Optional[float],
                                         take_profit: Optional[float],
                                         strategy: str):
        """Process execution and update positions"""
        try:
            # Update or create position
            await self._update_position(execution)

            # Set stop loss and take profit if specified
            if stop_loss or take_profit:
                await self._set_position_protection(execution.symbol,
                                                    stop_loss, take_profit)

            # Update daily P&L
            self._update_daily_pnl(execution)

            # Risk monitoring
            await self._check_risk_limits()

            logger.info(
                f"✅ Post-execution processing completed for {execution.execution_id}"
            )

        except Exception as e:
            logger.error(f"❌ Post-execution processing failed: {e}")

    async def _update_position(self, execution: TradeExecution):
        """Update position based on execution"""
        try:
            symbol = execution.symbol

            if symbol in self.open_positions:
                # Update existing position
                position = self.open_positions[symbol]

                if execution.side == position.side:
                    # Adding to position
                    total_size = position.size + execution.amount
                    weighted_price = (
                        (position.size * position.entry_price) +
                        (execution.amount * execution.price)) / total_size

                    position.size = total_size
                    position.entry_price = weighted_price
                else:
                    # Reducing or closing position
                    if execution.amount >= position.size:
                        # Position closed or reversed
                        realized_pnl = self._calculate_realized_pnl(
                            position, execution)
                        position.realized_pnl += realized_pnl

                        if execution.amount > position.size:
                            # Position reversed
                            position.side = execution.side
                            position.size = execution.amount - position.size
                            position.entry_price = execution.price
                        else:
                            # Position closed
                            del self.open_positions[symbol]
                            return
                    else:
                        # Partial position reduction
                        realized_pnl = self._calculate_realized_pnl(
                            position, execution)
                        position.realized_pnl += realized_pnl
                        position.size -= execution.amount
            else:
                # New position
                position = Position(position_id=str(uuid.uuid4()),
                                    symbol=symbol,
                                    side=execution.side,
                                    size=execution.amount,
                                    entry_price=execution.price,
                                    current_price=execution.price,
                                    unrealized_pnl=0.0,
                                    realized_pnl=0.0,
                                    entry_timestamp=execution.timestamp)

                self.open_positions[symbol] = position

            logger.debug(f"📊 Position updated for {symbol}")

        except Exception as e:
            logger.error(f"❌ Position update failed: {e}")

    async def _monitor_positions(self):
        """Continuously monitor open positions"""
        try:
            while True:
                if self.open_positions:
                    for symbol, position in self.open_positions.items():
                        try:
                            # Update current price
                            current_price = await self._get_market_price(symbol
                                                                         )
                            if current_price:
                                position.current_price = current_price

                                # Calculate unrealized P&L
                                position.unrealized_pnl = self._calculate_unrealized_pnl(
                                    position)

                                # Update max profit/drawdown
                                if position.unrealized_pnl > position.max_profit:
                                    position.max_profit = position.unrealized_pnl
                                elif position.unrealized_pnl < position.max_drawdown:
                                    position.max_drawdown = position.unrealized_pnl

                                # Check stop loss/take profit
                                await self._check_position_exits(position)

                        except Exception as e:
                            logger.error(
                                f"❌ Error monitoring position {symbol}: {e}")

                await asyncio.sleep(10)  # Check every 10 seconds

        except Exception as e:
            logger.error(f"❌ Position monitoring failed: {e}")

    async def _check_position_exits(self, position: Position):
        """Check if position should be closed due to stop loss or take profit"""
        try:
            if not position.stop_loss and not position.take_profit:
                return

            current_price = position.current_price
            should_close = False
            exit_reason = ""

            if position.side == 'buy':
                if position.stop_loss and current_price <= position.stop_loss:
                    should_close = True
                    exit_reason = "Stop Loss"
                elif position.take_profit and current_price >= position.take_profit:
                    should_close = True
                    exit_reason = "Take Profit"
            else:  # sell position
                if position.stop_loss and current_price >= position.stop_loss:
                    should_close = True
                    exit_reason = "Stop Loss"
                elif position.take_profit and current_price <= position.take_profit:
                    should_close = True
                    exit_reason = "Take Profit"

            if should_close:
                logger.info(
                    f"🎯 {exit_reason} triggered for {position.symbol} at {current_price}"
                )
                await self._close_position(position, exit_reason)

        except Exception as e:
            logger.error(f"❌ Position exit check failed: {e}")

    async def _close_position(self, position: Position, reason: str):
        """Close a position"""
        try:
            # Determine close side (opposite of position side)
            close_side = 'sell' if position.side == 'buy' else 'buy'

            # Execute closing order
            execution = await self.execute_smart_order(
                symbol=position.symbol,
                side=close_side,
                amount=position.size,
                order_type=OrderType.MARKET,
                strategy=f"position_close_{reason.lower()}")

            if execution:
                logger.info(f"✅ Position closed: {position.symbol} - {reason}")
            else:
                logger.error(f"❌ Failed to close position: {position.symbol}")

        except Exception as e:
            logger.error(f"❌ Position close failed: {e}")

    # Additional helper methods
    def _calculate_slippage(self, expected_price: float,
                            execution_price: float, side: str) -> float:
        """Calculate execution slippage"""
        if not expected_price or not execution_price:
            return 0.0

        if side == 'buy':
            return (execution_price - expected_price) / expected_price
        else:
            return (expected_price - execution_price) / expected_price

    def _calculate_market_impact(self, amount: float, price: float,
                                 symbol: str) -> float:
        """Calculate market impact of the trade"""
        # Simplified market impact calculation
        trade_value = amount * price
        return min(trade_value / 1000000, 0.01)  # Max 1% impact

    def _calculate_realized_pnl(self, position: Position,
                                execution: TradeExecution) -> float:
        """Calculate realized P&L from execution"""
        if position.side == 'buy':
            return (execution.price - position.entry_price) * execution.amount
        else:
            return (position.entry_price - execution.price) * execution.amount

    def _calculate_unrealized_pnl(self, position: Position) -> float:
        """Calculate unrealized P&L for position"""
        if position.side == 'buy':
            return (position.current_price -
                    position.entry_price) * position.size
        else:
            return (position.entry_price -
                    position.current_price) * position.size

    async def _get_market_price(
            self,
            symbol: str,
            exchange_name: Optional[str] = None) -> Optional[float]:
        """Get current market price for symbol"""
        try:
            if exchange_name:
                exchanges = [exchange_name]
            else:
                exchanges = self.exchange_manager.get_connected_exchanges()

            for exchange in exchanges:
                prices = await self.exchange_manager.get_prices(
                    exchange, [symbol])
                if prices and symbol in prices:
                    price_data = prices[symbol]
                    return (price_data.get('bid', 0) +
                            price_data.get('ask', 0)) / 2

            return None

        except Exception as e:
            logger.error(f"❌ Failed to get market price for {symbol}: {e}")
            return None

    def _calculate_total_exposure(self) -> float:
        """Calculate total position exposure"""
        total = 0.0
        for position in self.open_positions.values():
            total += position.size * position.current_price
        return total

    async def _get_available_balance(self) -> float:
        """Get available trading balance"""
        try:
            # Get balance from primary exchange
            primary_exchange = self.exchange_manager.get_connected_exchanges(
            )[0]
            balance = await self.exchange_manager.get_account_balance(
                primary_exchange)

            if balance and 'USDT' in balance:
                return balance['USDT'].get('free', 0)

            return 0.0

        except Exception as e:
            logger.error(f"❌ Failed to get available balance: {e}")
            return 0.0

    def _check_position_concentration(self, symbol: str,
                                      position_value: float) -> bool:
        """Check position concentration limits"""
        # Don't allow more than 30% in single symbol
        total_exposure = self._calculate_total_exposure()
        if total_exposure > 0:
            concentration = position_value / (total_exposure + position_value)
            if concentration > 0.3:
                logger.error(
                    f"❌ Position concentration too high: {concentration:.1%}")
                return False
        return True

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded"""
        # Reset daily counters if new day
        current_date = datetime.now().date()
        if current_date > self.daily_reset_time:
            self.daily_losses = 0.0
            self.daily_reset_time = current_date

        return self.daily_losses >= self.max_daily_loss

    def _update_daily_pnl(self, execution: TradeExecution):
        """Update daily P&L tracking"""
        # This would integrate with position P&L calculation
        pass

    async def _check_risk_limits(self):
        """Check all risk limits and take action if needed"""
        try:
            # Check total exposure
            total_exposure = self._calculate_total_exposure()
            if total_exposure > self.max_total_exposure:
                logger.warning(
                    f"⚠️ Total exposure limit exceeded: ${total_exposure:.2f}")
                # Could implement automatic position reduction here

            # Check daily losses
            if self._check_daily_loss_limit():
                logger.error(
                    "🛑 Daily loss limit reached - activating emergency stop")
                self.emergency_stop = True

        except Exception as e:
            logger.error(f"❌ Risk limit check failed: {e}")

    def _update_execution_metrics(self, execution: TradeExecution,
                                  execution_time_ms: float):
        """Update execution performance metrics"""
        self.execution_metrics['total_executions'] += 1
        self.execution_metrics['successful_executions'] += 1

        # Update averages
        total = self.execution_metrics['total_executions']

        # Execution time
        current_avg_time = self.execution_metrics['avg_execution_time']
        self.execution_metrics['avg_execution_time'] = (
            (current_avg_time * (total - 1)) + execution_time_ms) / total

        # Slippage
        current_avg_slippage = self.execution_metrics['avg_slippage']
        self.execution_metrics['avg_slippage'] = (
            (current_avg_slippage * (total - 1)) + execution.slippage) / total

        # Fees
        self.execution_metrics['total_fees_paid'] += execution.fee

    def _get_recent_exchange_performance(self, exchange_name: str) -> float:
        """Get recent performance score for exchange (0-1)"""
        # Analyze recent executions for this exchange
        recent_executions = [
            e for e in self.trade_history[-20:]  # Last 20 executions
            if e.exchange == exchange_name
        ]

        if not recent_executions:
            return 0.5  # Neutral score

        # Calculate success rate and average execution quality
        success_rate = len([
            e for e in recent_executions if e.status == OrderStatus.FILLED
        ]) / len(recent_executions)
        avg_slippage = np.mean([e.slippage for e in recent_executions])

        # Combine metrics into performance score
        slippage_penalty = min(abs(avg_slippage) * 10,
                               0.5)  # Cap penalty at 0.5
        performance_score = success_rate - slippage_penalty

        return max(0, min(1, performance_score))

    async def _load_existing_positions(self):
        """Load existing positions from exchange"""
        try:
            # This would typically load from a database or exchange API
            # For now, we'll check open orders and reconstruct positions
            logger.info("📊 Loading existing positions...")

            for exchange_name in self.exchange_manager.get_connected_exchanges(
            ):
                try:
                    # Get open orders
                    open_orders = await self.exchange_manager.exchanges[
                        exchange_name].fetch_open_orders()

                    # Get current positions (for margin trading)
                    # This would depend on exchange capabilities

                    logger.debug(
                        f"📊 Loaded {len(open_orders)} open orders from {exchange_name}"
                    )

                except Exception as e:
                    logger.warning(
                        f"⚠️ Could not load positions from {exchange_name}: {e}"
                    )

        except Exception as e:
            logger.error(f"❌ Failed to load existing positions: {e}")

    async def _monitor_executions(self):
        """Monitor execution queue for timeouts and issues"""
        try:
            while True:
                try:
                    # Process execution queue
                    execution_item = await asyncio.wait_for(
                        self.execution_queue.get(), timeout=1.0)

                    # Check for timeouts
                    if time.time(
                    ) - execution_item['start_time'] > self.execution_timeout:
                        logger.warning(
                            f"⏰ Execution timeout detected: {execution_item['execution_id']}"
                        )

                except asyncio.TimeoutError:
                    # No items in queue, continue
                    continue
                except Exception as e:
                    logger.error(f"❌ Execution monitoring error: {e}")

                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"❌ Execution monitor failed: {e}")

    async def _set_position_protection(self, symbol: str,
                                       stop_loss: Optional[float],
                                       take_profit: Optional[float]):
        """Set stop loss and take profit for position"""
        try:
            if symbol not in self.open_positions:
                return

            position = self.open_positions[symbol]

            if stop_loss:
                position.stop_loss = stop_loss
                logger.info(f"🛡️ Stop loss set for {symbol}: {stop_loss}")

            if take_profit:
                position.take_profit = take_profit
                logger.info(f"🎯 Take profit set for {symbol}: {take_profit}")

        except Exception as e:
            logger.error(f"❌ Failed to set position protection: {e}")

    # Public interface methods
    async def execute_market_order(self, exchange_name: str, symbol: str,
                                   side: str, amount: float) -> Optional[Dict]:
        """Execute market order (legacy interface for compatibility)"""
        execution = await self.execute_smart_order(
            symbol=symbol,
            side=side,
            amount=amount,
            order_type=OrderType.MARKET,
            strategy="legacy_market_order")

        if execution:
            return {
                'id': execution.order_id,
                'symbol': execution.symbol,
                'side': execution.side,
                'amount': execution.amount,
                'price': execution.price,
                'timestamp': execution.timestamp
            }

        return None

    async def place_limit_order(self, exchange_name: str, symbol: str,
                                side: str, amount: float,
                                price: float) -> Optional[Dict]:
        """Place limit order (legacy interface for compatibility)"""
        execution = await self.execute_smart_order(
            symbol=symbol,
            side=side,
            amount=amount,
            order_type=OrderType.LIMIT,
            price=price,
            strategy="legacy_limit_order")

        if execution:
            return {
                'id': execution.order_id,
                'symbol': execution.symbol,
                'side': execution.side,
                'amount': execution.amount,
                'price': execution.price,
                'timestamp': execution.timestamp
            }

        return None

    async def get_account_balance(self) -> Dict[str, Any]:
        """Get account balance from all exchanges"""
        balances = {}

        for exchange_name in self.exchange_manager.get_connected_exchanges():
            try:
                balance = await self.exchange_manager.get_account_balance(
                    exchange_name)
                if balance:
                    balances[exchange_name] = balance
            except Exception as e:
                logger.error(f"❌ Failed to get {exchange_name} balance: {e}")

        return balances

    async def get_open_orders(self) -> Dict[str, List]:
        """Get all open orders"""
        open_orders = {}

        for exchange_name in self.exchange_manager.get_connected_exchanges():
            try:
                exchange = self.exchange_manager.exchanges[exchange_name]
                orders = await exchange.fetch_open_orders()
                open_orders[exchange_name] = orders
            except Exception as e:
                logger.error(
                    f"❌ Failed to get {exchange_name} open orders: {e}")
                open_orders[exchange_name] = []

        return open_orders

    async def cancel_order(self, exchange_name: str, order_id: str,
                           symbol: str) -> bool:
        """Cancel a specific order"""
        try:
            exchange = self.exchange_manager.exchanges.get(exchange_name)
            if not exchange:
                return False

            await exchange.cancel_order(order_id, symbol)
            logger.info(f"✅ Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Order cancellation failed: {e}")
            return False

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders for symbol or all symbols"""
        cancelled_count = 0

        try:
            open_orders = await self.get_open_orders()

            for exchange_name, orders in open_orders.items():
                for order in orders:
                    if symbol is None or order.get('symbol') == symbol:
                        try:
                            success = await self.cancel_order(
                                exchange_name, order['id'], order['symbol'])
                            if success:
                                cancelled_count += 1
                        except Exception as e:
                            logger.error(
                                f"❌ Failed to cancel order {order['id']}: {e}")

            logger.info(f"✅ Cancelled {cancelled_count} orders")

        except Exception as e:
            logger.error(f"❌ Cancel all orders failed: {e}")

        return cancelled_count

    async def close_all_positions(self, reason: str = "manual_close") -> int:
        """Close all open positions"""
        closed_count = 0

        try:
            positions_to_close = list(self.open_positions.values())

            for position in positions_to_close:
                try:
                    await self._close_position(position, reason)
                    closed_count += 1
                except Exception as e:
                    logger.error(
                        f"❌ Failed to close position {position.symbol}: {e}")

            logger.info(f"✅ Closed {closed_count} positions")

        except Exception as e:
            logger.error(f"❌ Close all positions failed: {e}")

        return closed_count

    def activate_emergency_stop(self, reason: str = "manual_activation"):
        """Activate emergency stop"""
        self.emergency_stop = True
        logger.error(f"🛑 EMERGENCY STOP ACTIVATED: {reason}")

    def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        self.emergency_stop = False
        logger.info("✅ Emergency stop deactivated")

    def get_positions_summary(self) -> Dict[str, Any]:
        """Get comprehensive positions summary"""
        total_unrealized_pnl = sum(pos.unrealized_pnl
                                   for pos in self.open_positions.values())
        total_realized_pnl = sum(pos.realized_pnl
                                 for pos in self.open_positions.values())
        total_exposure = self._calculate_total_exposure()

        position_details = []
        for pos in self.open_positions.values():
            position_details.append({
                'symbol':
                pos.symbol,
                'side':
                pos.side,
                'size':
                pos.size,
                'entry_price':
                pos.entry_price,
                'current_price':
                pos.current_price,
                'unrealized_pnl':
                pos.unrealized_pnl,
                'unrealized_pnl_pct':
                (pos.unrealized_pnl / (pos.size * pos.entry_price)) * 100,
                'max_profit':
                pos.max_profit,
                'max_drawdown':
                pos.max_drawdown,
                'entry_time':
                pos.entry_timestamp.isoformat(),
                'holding_period':
                str(datetime.now() - pos.entry_timestamp)
            })

        return {
            'total_positions': len(self.open_positions),
            'total_exposure': total_exposure,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'positions': position_details
        }

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary"""
        recent_executions = self.trade_history[-50:] if len(
            self.trade_history) > 50 else self.trade_history

        if not recent_executions:
            return {
                'total_executions': 0,
                'success_rate': 0,
                'avg_execution_time': 0,
                'avg_slippage': 0,
                'total_fees': 0
            }

        successful_executions = [
            e for e in recent_executions if e.status == OrderStatus.FILLED
        ]
        success_rate = len(successful_executions) / len(recent_executions)

        return {
            'total_executions':
            len(recent_executions),
            'successful_executions':
            len(successful_executions),
            'success_rate':
            success_rate,
            'avg_execution_time':
            self.execution_metrics['avg_execution_time'],
            'avg_slippage':
            self.execution_metrics['avg_slippage'],
            'total_fees_paid':
            self.execution_metrics['total_fees_paid'],
            'recent_executions': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'symbol': e.symbol,
                    'side': e.side,
                    'amount': e.amount,
                    'price': e.price,
                    'slippage': e.slippage,
                    'fee': e.fee
                } for e in recent_executions[-10:]  # Last 10 executions
            ]
        }

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk management metrics"""
        return {
            'emergency_stop_active':
            self.emergency_stop,
            'daily_losses':
            self.daily_losses,
            'daily_loss_limit':
            self.max_daily_loss,
            'total_exposure':
            self._calculate_total_exposure(),
            'max_total_exposure':
            self.max_total_exposure,
            'exposure_utilization':
            (self._calculate_total_exposure() / self.max_total_exposure) * 100,
            'position_count':
            len(self.open_positions),
            'max_positions':
            10,  # Could be configurable
            'largest_position':
            max([
                pos.size * pos.current_price
                for pos in self.open_positions.values()
            ]) if self.open_positions else 0
        }

    async def shutdown(self):
        """Shutdown the trading executor"""
        try:
            logger.info("🛑 Shutting down Trading Executor...")

            # Cancel monitoring tasks
            if self.execution_monitor_task:
                self.execution_monitor_task.cancel()
            if self.position_monitor_task:
                self.position_monitor_task.cancel()

            # Wait for tasks to complete
            await asyncio.sleep(1)

            logger.info("✅ Trading Executor shutdown complete")

        except Exception as e:
            logger.error(f"❌ Shutdown failed: {e}")

    def get_trade_summary(self) -> Dict[str, Any]:
        """Get trading summary statistics (legacy compatibility)"""
        execution_summary = self.get_execution_summary()
        positions_summary = self.get_positions_summary()

        return {
            'total_trades':
            execution_summary['total_executions'],
            'executed_trades':
            execution_summary['successful_executions'],
            'pending_trades':
            len(self.open_orders),
            'open_positions':
            positions_summary['total_positions'],
            'total_pnl':
            positions_summary['total_unrealized_pnl'] +
            positions_summary['total_realized_pnl'],
            'latest_trades':
            execution_summary['recent_executions'][-5:]
        }


# Alias for backward compatibility
LiveTradingExecutor = ProfessionalLiveTradingExecutor
