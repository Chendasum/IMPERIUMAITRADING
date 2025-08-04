import asyncio
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class ProfessionalArbitrageStrategy:
    """Professional arbitrage strategy with real market implementation"""
    
    def __init__(self):
        self.name = "arbitrage"
        self.exchange_manager = None
        self.risk_manager = None
        
        # Professional arbitrage parameters
        self.min_profit_threshold = 0.005  # 0.5% minimum (more realistic)
        self.max_position_size = 1000  # Max $1000 per arbitrage
        self.execution_time_limit = 3.0  # 3 seconds max execution time
        self.min_volume_threshold = 1000  # Min $1000 volume on both sides
        
        # Exchange fees (critical for real arbitrage)
        self.exchange_fees = {
            'binance': {'maker': 0.001, 'taker': 0.001},  # 0.1%
            'coinbase': {'maker': 0.005, 'taker': 0.005},  # 0.5%
            'bybit': {'maker': 0.001, 'taker': 0.001}     # 0.1%
        }
        
        # Track execution history to avoid repeated failed opportunities
        self.failed_opportunities = {}
        self.successful_arbitrages = []
        self.last_price_update = {}
        
        # Real-time tracking
        self.price_staleness_limit = 5  # seconds
        
    def initialize(self, exchange_manager, risk_manager):
        """Initialize strategy with enhanced validation"""
        self.exchange_manager = exchange_manager
        self.risk_manager = risk_manager
        
        # Validate we have multiple exchanges for arbitrage
        if len(self.exchange_manager.exchanges) < 2:
            logger.warning("⚠️ Arbitrage requires multiple exchanges - limited opportunities")
        else:
            logger.info(f"✅ Arbitrage initialized with {len(self.exchange_manager.exchanges)} exchanges")
        
        logger.info("✅ Professional Arbitrage strategy initialized")
    
    async def generate_signals(self):
        """Generate real arbitrage trading signals with proper validation"""
        signals = []
        
        try:
            # Get all available exchanges
            available_exchanges = list(self.exchange_manager.exchanges.keys())
            if len(available_exchanges) < 2:
                logger.debug("📊 Need at least 2 exchanges for arbitrage")
                return signals
            
            # Get real-time prices from all exchanges simultaneously
            exchange_prices = await self._get_all_exchange_prices(available_exchanges)
            
            # Find arbitrage opportunities across real exchanges
            opportunities = await self._find_real_arbitrage_opportunities(exchange_prices)
            
            # Validate and create signals for profitable opportunities
            for opportunity in opportunities:
                if await self._validate_arbitrage_opportunity(opportunity):
                    signal = await self._create_arbitrage_signal(opportunity)
                    if signal:
                        signals.append(signal)
            
            if signals:
                logger.info(f"💰 Found {len(signals)} validated arbitrage opportunities")
            else:
                logger.debug("📊 No profitable arbitrage opportunities found")
                
        except Exception as e:
            logger.error(f"❌ Arbitrage signal generation failed: {e}")
        
        return signals
    
    async def _get_all_exchange_prices(self, exchanges):
        """Get prices from all exchanges simultaneously"""
        exchange_prices = {}
        price_tasks = []
        
        # Create tasks for parallel price fetching
        for exchange in exchanges:
            task = asyncio.create_task(self._get_exchange_prices_with_metadata(exchange))
            price_tasks.append((exchange, task))
        
        # Execute all price fetches simultaneously
        for exchange, task in price_tasks:
            try:
                prices = await asyncio.wait_for(task, timeout=2.0)  # 2 second timeout
                if prices:
                    exchange_prices[exchange] = prices
                    self.last_price_update[exchange] = time.time()
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Price fetch timeout for {exchange}")
            except Exception as e:
                logger.warning(f"❌ Failed to get prices from {exchange}: {e}")
        
        return exchange_prices
    
    async def _get_exchange_prices_with_metadata(self, exchange):
        """Get prices with additional metadata for arbitrage analysis"""
        try:
            prices = await self.exchange_manager.get_prices(exchange)
            
            # Add metadata for each price
            enhanced_prices = {}
            for pair, price_data in prices.items():
                if price_data and 'bid' in price_data and 'ask' in price_data:
                    enhanced_prices[pair] = {
                        **price_data,
                        'spread': price_data['ask'] - price_data['bid'],
                        'spread_pct': (price_data['ask'] - price_data['bid']) / price_data['bid'],
                        'mid_price': (price_data['bid'] + price_data['ask']) / 2,
                        'timestamp': time.time(),
                        'exchange': exchange
                    }
            
            return enhanced_prices
            
        except Exception as e:
            logger.error(f"❌ Failed to get enhanced prices from {exchange}: {e}")
            return {}
    
    async def _find_real_arbitrage_opportunities(self, exchange_prices):
        """Find real arbitrage opportunities between exchanges"""
        opportunities = []
        
        # Get common pairs across exchanges
        common_pairs = self._get_common_pairs(exchange_prices)
        
        for pair in common_pairs:
            try:
                # Get all prices for this pair across exchanges
                pair_prices = {}
                for exchange, prices in exchange_prices.items():
                    if pair in prices:
                        pair_prices[exchange] = prices[pair]
                
                if len(pair_prices) < 2:
                    continue
                
                # Find best buy and sell opportunities
                best_buy = min(pair_prices.items(), key=lambda x: x[1]['ask'])
                best_sell = max(pair_prices.items(), key=lambda x: x[1]['bid'])
                
                buy_exchange, buy_data = best_buy
                sell_exchange, sell_data = best_sell
                
                # Skip if same exchange
                if buy_exchange == sell_exchange:
                    continue
                
                # Calculate gross profit
                buy_price = buy_data['ask']  # We buy at ask price
                sell_price = sell_data['bid']  # We sell at bid price
                gross_profit_pct = (sell_price - buy_price) / buy_price
                
                # Calculate net profit after fees
                buy_fee = self.exchange_fees.get(buy_exchange, {}).get('taker', 0.001)
                sell_fee = self.exchange_fees.get(sell_exchange, {}).get('taker', 0.001)
                total_fees = buy_fee + sell_fee
                net_profit_pct = gross_profit_pct - total_fees
                
                # Check if profitable after fees
                if net_profit_pct > self.min_profit_threshold:
                    opportunity = {
                        'pair': pair,
                        'buy_exchange': buy_exchange,
                        'sell_exchange': sell_exchange,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'gross_profit_pct': gross_profit_pct,
                        'net_profit_pct': net_profit_pct,
                        'total_fees': total_fees,
                        'buy_data': buy_data,
                        'sell_data': sell_data,
                        'timestamp': time.time()
                    }
                    opportunities.append(opportunity)
                    
                    logger.info(f"🔍 Arbitrage found: {pair} - Buy {buy_exchange} @ {buy_price:.6f}, Sell {sell_exchange} @ {sell_price:.6f} = {net_profit_pct:.2%} net profit")
            
            except Exception as e:
                logger.error(f"❌ Error analyzing arbitrage for {pair}: {e}")
        
        # Sort by net profit percentage (highest first)
        opportunities.sort(key=lambda x: x['net_profit_pct'], reverse=True)
        
        return opportunities
    
    def _get_common_pairs(self, exchange_prices):
        """Get pairs that are available on multiple exchanges"""
        if not exchange_prices:
            return []
        
        # Start with pairs from first exchange
        common_pairs = set(next(iter(exchange_prices.values())).keys())
        
        # Find intersection across all exchanges
        for prices in exchange_prices.values():
            common_pairs &= set(prices.keys())
        
        # Focus on major pairs for better liquidity
        priority_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
        
        # Return priority pairs first, then others
        result = [pair for pair in priority_pairs if pair in common_pairs]
        result.extend([pair for pair in common_pairs if pair not in priority_pairs])
        
        return result
    
    async def _validate_arbitrage_opportunity(self, opportunity):
        """Validate arbitrage opportunity with real-world constraints"""
        try:
            # Check price staleness (prices must be recent)
            current_time = time.time()
            buy_age = current_time - opportunity['buy_data']['timestamp']
            sell_age = current_time - opportunity['sell_data']['timestamp']
            
            if buy_age > self.price_staleness_limit or sell_age > self.price_staleness_limit:
                logger.debug(f"❌ Stale prices for {opportunity['pair']} - ages: {buy_age:.1f}s, {sell_age:.1f}s")
                return False
            
            # Check minimum profit threshold
            if opportunity['net_profit_pct'] < self.min_profit_threshold:
                return False
            
            # Check spread reasonableness (avoid artificially wide spreads)
            buy_spread_pct = opportunity['buy_data']['spread_pct']
            sell_spread_pct = opportunity['sell_data']['spread_pct']
            
            if buy_spread_pct > 0.01 or sell_spread_pct > 0.01:  # 1% spread limit
                logger.debug(f"❌ Excessive spread for {opportunity['pair']} - {buy_spread_pct:.2%}, {sell_spread_pct:.2%}")
                return False
            
            # Check if we've failed this opportunity recently
            opportunity_key = f"{opportunity['pair']}_{opportunity['buy_exchange']}_{opportunity['sell_exchange']}"
            if opportunity_key in self.failed_opportunities:
                last_failure = self.failed_opportunities[opportunity_key]
                if current_time - last_failure < 300:  # Don't retry for 5 minutes
                    return False
            
            # Estimate execution time and check feasibility
            estimated_execution_time = self._estimate_execution_time(opportunity)
            if estimated_execution_time > self.execution_time_limit:
                logger.debug(f"❌ {opportunity['pair']} execution time too long: {estimated_execution_time:.1f}s")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Opportunity validation failed: {e}")
            return False
    
    def _estimate_execution_time(self, opportunity):
        """Estimate total execution time for arbitrage"""
        # Base execution time per exchange
        base_time_per_exchange = 1.0  # 1 second per trade
        
        # Add network latency estimates
        network_latency = 0.5  # 500ms total network time
        
        # Add processing time
        processing_time = 0.3  # 300ms processing
        
        total_time = (base_time_per_exchange * 2) + network_latency + processing_time
        return total_time
    
    async def _create_arbitrage_signal(self, opportunity):
        """Create trading signal from validated arbitrage opportunity"""
        try:
            # Calculate optimal position size
            position_size = min(
                self.max_position_size,
                self.risk_manager.current_balance * 0.1  # Max 10% of balance per arbitrage
            )
            
            # Create the trading signal
            signal = {
                'pair': opportunity['pair'],
                'action': 'arbitrage',  # Special action type for arbitrage
                'price': opportunity['buy_price'],
                'target': opportunity['sell_price'],
                'confidence': min(0.95, opportunity['net_profit_pct'] * 100),  # Higher profit = higher confidence
                'strategy': self.name,
                'profit_expected': opportunity['net_profit_pct'],
                'profit_potential': opportunity['gross_profit_pct'],
                'timestamp': datetime.now(),
                
                # Arbitrage-specific data
                'buy_exchange': opportunity['buy_exchange'],
                'sell_exchange': opportunity['sell_exchange'],
                'total_fees': opportunity['total_fees'],
                'position_size': position_size,
                'execution_time_estimate': self._estimate_execution_time(opportunity),
                
                # Risk data
                'max_slippage': 0.001,  # 0.1% max acceptable slippage
                'timeout': 5.0,  # 5 second timeout for execution
            }
            
            logger.info(f"💰 ARBITRAGE SIGNAL: {opportunity['pair']} - {opportunity['net_profit_pct']:.2%} net profit")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Signal creation failed: {e}")
            return None
    
    def record_arbitrage_result(self, signal, success, actual_profit=0):
        """Record arbitrage execution result for learning"""
        try:
            if success:
                self.successful_arbitrages.append({
                    'pair': signal['pair'],
                    'expected_profit': signal['profit_expected'],
                    'actual_profit': actual_profit,
                    'timestamp': datetime.now()
                })
                logger.info(f"✅ Arbitrage success: {signal['pair']} - {actual_profit:.2%} actual profit")
            else:
                # Record failed opportunity to avoid repeating
                opportunity_key = f"{signal['pair']}_{signal['buy_exchange']}_{signal['sell_exchange']}"
                self.failed_opportunities[opportunity_key] = time.time()
                logger.warning(f"❌ Arbitrage failed: {signal['pair']}")
            
            # Cleanup old failed opportunities (older than 1 hour)
            current_time = time.time()
            self.failed_opportunities = {
                k: v for k, v in self.failed_opportunities.items() 
                if current_time - v < 3600
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to record arbitrage result: {e}")
    
    def get_performance_metrics(self):
        """Get arbitrage strategy performance metrics"""
        if not self.successful_arbitrages:
            return {
                'total_arbitrages': 0,
                'avg_profit': 0,
                'success_rate': 0,
                'total_profit': 0
            }
        
        total_arbitrages = len(self.successful_arbitrages)
        avg_profit = sum(arb['actual_profit'] for arb in self.successful_arbitrages) / total_arbitrages
        total_profit = sum(arb['actual_profit'] for arb in self.successful_arbitrages)
        
        return {
            'total_arbitrages': total_arbitrages,
            'avg_profit': avg_profit,
            'success_rate': 1.0,  # Only successful ones are recorded
            'total_profit': total_profit,
            'failed_opportunities': len(self.failed_opportunities)
        }
