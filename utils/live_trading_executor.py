import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class LiveTradingExecutor:
    def __init__(self, exchange_manager):
        self.exchange_manager = exchange_manager
        self.open_positions = {}
        self.trade_history = []
        
    async def execute_market_order(self, exchange_name, symbol, side, amount):
        """Execute a live market order"""
        try:
            exchange = self.exchange_manager.exchanges.get(exchange_name)
            if not exchange:
                raise Exception(f"Exchange {exchange_name} not available")
            
            # Execute market order
            order = await exchange.create_market_order(
                symbol=symbol,
                side=side,  # 'buy' or 'sell'
                amount=amount
            )
            
            logger.info(f"✅ LIVE ORDER EXECUTED: {side.upper()} {amount} {symbol} - Order ID: {order['id']}")
            
            # Track the order
            self.trade_history.append({
                'timestamp': datetime.now(),
                'exchange': exchange_name,
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'order_id': order['id'],
                'status': 'executed'
            })
            
            return order
            
        except Exception as e:
            logger.error(f"❌ LIVE ORDER FAILED: {e}")
            raise
    
    async def get_account_balance(self):
        """Get current account balance"""
        try:
            balances = {}
            for name, exchange in self.exchange_manager.exchanges.items():
                try:
                    balance = await exchange.fetch_balance()
                    balances[name] = balance
                except Exception as e:
                    logger.error(f"❌ Failed to get {name} balance: {e}")
            
            return balances
        except Exception as e:
            logger.error(f"❌ Balance fetch failed: {e}")
            return {}
    
    async def get_open_orders(self):
        """Get all open orders"""
        try:
            open_orders = {}
            for name, exchange in self.exchange_manager.exchanges.items():
                try:
                    orders = await exchange.fetch_open_orders()
                    open_orders[name] = orders
                except Exception as e:
                    logger.error(f"❌ Failed to get {name} open orders: {e}")
            
            return open_orders
        except Exception as e:
            logger.error(f"❌ Open orders fetch failed: {e}")
            return {}
    
    async def cancel_order(self, exchange_name, order_id, symbol):
        """Cancel a specific order"""
        try:
            exchange = self.exchange_manager.exchanges.get(exchange_name)
            if not exchange:
                raise Exception(f"Exchange {exchange_name} not available")
            
            result = await exchange.cancel_order(order_id, symbol)
            logger.info(f"✅ Order cancelled: {order_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Order cancellation failed: {e}")
            raise
    
    async def place_limit_order(self, exchange_name, symbol, side, amount, price):
        """Place a limit order"""
        try:
            exchange = self.exchange_manager.exchanges.get(exchange_name)
            if not exchange:
                raise Exception(f"Exchange {exchange_name} not available")
            
            order = await exchange.create_limit_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price
            )
            
            logger.info(f"✅ LIMIT ORDER PLACED: {side.upper()} {amount} {symbol} @ ${price}")
            
            # Track the order
            self.trade_history.append({
                'timestamp': datetime.now(),
                'exchange': exchange_name,
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'order_id': order['id'],
                'type': 'limit',
                'status': 'pending'
            })
            
            return order
            
        except Exception as e:
            logger.error(f"❌ LIMIT ORDER FAILED: {e}")
            raise
    
    def get_trade_summary(self):
        """Get trading summary statistics"""
        total_trades = len(self.trade_history)
        executed_trades = len([t for t in self.trade_history if t['status'] == 'executed'])
        
        return {
            'total_trades': total_trades,
            'executed_trades': executed_trades,
            'pending_trades': total_trades - executed_trades,
            'latest_trades': self.trade_history[-5:] if self.trade_history else []
        }