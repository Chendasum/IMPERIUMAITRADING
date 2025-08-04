import ccxt
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import json
from config import TradingConfig

logger = logging.getLogger(__name__)


class ProfessionalExchangeManager:
    """Professional Exchange Manager with Enhanced Reliability and Multi-Exchange Support"""

    def __init__(self):
        self.exchanges = {}
        self.config = TradingConfig()
        
        # Connection status tracking
        self.connection_status = {}
        self.last_health_check = {}
        self.failed_requests = {}
        
        # Rate limiting and retry logic
        self.rate_limits = {}
        self.request_timestamps = {}
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        
        # Exchange-specific configurations
        self.exchange_configs = {
            'binance': {
                'name': 'Binance',
                'rate_limit': 1200,  # requests per minute
                'weight_limit': 1200,
                'timeout': 30000,
                'required_credentials': ['apiKey', 'secret'],
                'testnet_available': True
            },
            'coinbase': {
                'name': 'Coinbase Pro',
                'rate_limit': 10,  # requests per second
                'timeout': 30000,
                'required_credentials': ['apiKey', 'secret', 'passphrase'],
                'testnet_available': True
            },
            'bybit': {
                'name': 'ByBit',
                'rate_limit': 120,  # requests per minute
                'timeout': 30000,
                'required_credentials': ['apiKey', 'secret'],
                'testnet_available': True
            }
        }
        
        # Backup data sources for price feeds
        self.backup_sources = ['coingecko', 'alpha_vantage', 'finnhub']
        self.backup_clients = {}
        
        # Performance metrics
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0,
            'exchanges_online': 0
        }

    async def initialize(self):
        """Initialize exchange connections with enhanced error handling"""
        try:
            logger.info("🏛️ Initializing Professional Exchange Manager...")
            
            # Initialize primary exchanges
            await self._initialize_primary_exchanges()
            
            # Initialize backup data sources
            await self._initialize_backup_sources()
            
            # Perform initial health checks
            await self._perform_health_checks()
            
            # Setup monitoring
            self._setup_monitoring()
            
            # Log initialization summary
            self._log_initialization_summary()

        except Exception as e:
            logger.error(f"❌ Exchange Manager initialization failed: {e}")
            raise

    async def _initialize_primary_exchanges(self):
        """Initialize primary exchange connections"""
        
        # Initialize Binance
        await self._initialize_binance()
        
        # Initialize Coinbase Pro (if configured)
        if self.config.COINBASE_API_KEY and self.config.COINBASE_SECRET_KEY:
            await self._initialize_coinbase()
        
        # Initialize ByBit (if configured)
        if hasattr(self.config, 'BYBIT_API_KEY') and self.config.BYBIT_API_KEY:
            await self._initialize_bybit()

    async def _initialize_binance(self):
        """Initialize Binance exchange with enhanced configuration"""
        try:
            exchange_config = {
                'enableRateLimit': True,
                'rateLimit': 100,  # milliseconds between requests
                'timeout': 30000,  # 30 seconds
                'sandbox': False,  # LIVE TRADING ONLY
                'options': {
                    'defaultType': 'spot',  # spot trading
                    'adjustForTimeDifference': True,
                },
                'headers': {
                    'User-Agent': 'Professional-Trading-Bot/1.0'
                }
            }
            
            # Add API credentials if available
            if self.config.BINANCE_API_KEY and self.config.BINANCE_SECRET_KEY:
                exchange_config.update({
                    'apiKey': self.config.BINANCE_API_KEY,
                    'secret': self.config.BINANCE_SECRET_KEY,
                })
                
                # Add testnet support if configured
                if hasattr(self.config, 'BINANCE_TESTNET') and self.config.BINANCE_TESTNET:
                    exchange_config['sandbox'] = True
                    logger.info("⚠️ Binance testnet mode enabled")
                
                logger.info("✅ Binance initialized with API credentials")
            else:
                logger.info("📊 Binance initialized for market data only")
            
            self.exchanges['binance'] = ccxt.binance(exchange_config)
            self.connection_status['binance'] = 'initializing'
            
        except Exception as e:
            logger.error(f"❌ Binance initialization failed: {e}")
            self.connection_status['binance'] = 'failed'

    async def _initialize_coinbase(self):
        """Initialize Coinbase Pro exchange"""
        try:
            exchange_config = {
                'apiKey': self.config.COINBASE_API_KEY,
                'secret': self.config.COINBASE_SECRET_KEY,
                'enableRateLimit': True,
                'rateLimit': 100,
                'timeout': 30000,
                'sandbox': False,
            }
            
            # Add passphrase if available
            if hasattr(self.config, 'COINBASE_PASSPHRASE') and self.config.COINBASE_PASSPHRASE:
                exchange_config['passphrase'] = self.config.COINBASE_PASSPHRASE
            
            self.exchanges['coinbase'] = ccxt.coinbasepro(exchange_config)
            self.connection_status['coinbase'] = 'initializing'
            logger.info("✅ Coinbase Pro initialized")
            
        except Exception as e:
            logger.error(f"❌ Coinbase Pro initialization failed: {e}")
            self.connection_status['coinbase'] = 'failed'

    async def _initialize_bybit(self):
        """Initialize ByBit exchange"""
        try:
            exchange_config = {
                'apiKey': self.config.BYBIT_API_KEY,
                'secret': self.config.BYBIT_API_SECRET,
                'enableRateLimit': True,
                'rateLimit': 100,
                'timeout': 30000,
                'sandbox': False,
            }
            
            self.exchanges['bybit'] = ccxt.bybit(exchange_config)
            self.connection_status['bybit'] = 'initializing'
            logger.info("✅ ByBit initialized")
            
        except Exception as e:
            logger.error(f"❌ ByBit initialization failed: {e}")
            self.connection_status['bybit'] = 'failed'

    async def _initialize_backup_sources(self):
        """Initialize backup data sources"""
        try:
            # Initialize CoinGecko client (free API)
            self.backup_clients['coingecko'] = CoinGeckoClient()
            
            # Initialize Alpha Vantage (if API key available)
            if hasattr(self.config, 'ALPHA_VANTAGE_API_KEY') and self.config.ALPHA_VANTAGE_API_KEY:
                self.backup_clients['alpha_vantage'] = AlphaVantageClient(self.config.ALPHA_VANTAGE_API_KEY)
            
            # Initialize Finnhub (if API key available)
            if hasattr(self.config, 'FINNHUB_API_KEY') and self.config.FINNHUB_API_KEY:
                self.backup_clients['finnhub'] = FinnhubClient(self.config.FINNHUB_API_KEY)
            
            logger.info(f"✅ Initialized {len(self.backup_clients)} backup data sources")
            
        except Exception as e:
            logger.error(f"❌ Backup sources initialization failed: {e}")

    async def _perform_health_checks(self):
        """Perform initial health checks on all exchanges"""
        logger.info("🔍 Performing exchange health checks...")
        
        for exchange_name, exchange in self.exchanges.items():
            await self._check_exchange_health(exchange_name, exchange)
        
        # Update performance metrics
        self.performance_metrics['exchanges_online'] = len([
            status for status in self.connection_status.values() 
            if status == 'connected'
        ])

    async def _check_exchange_health(self, exchange_name, exchange):
        """Check health of a specific exchange"""
        try:
            start_time = time.time()
            
            # Test basic connectivity
            markets = await self._execute_with_retry(
                exchange.load_markets,
                exchange_name,
                "load_markets"
            )
            
            if markets:
                self.connection_status[exchange_name] = 'connected'
                response_time = time.time() - start_time
                
                logger.info(f"✅ {exchange_name.title()} health check passed ({response_time:.2f}s)")
                
                # Test API credentials if available
                if exchange.apiKey:
                    await self._test_api_credentials(exchange_name, exchange)
            else:
                self.connection_status[exchange_name] = 'failed'
                logger.error(f"❌ {exchange_name.title()} health check failed")
                
        except Exception as e:
            self.connection_status[exchange_name] = 'failed'
            logger.error(f"❌ {exchange_name.title()} health check failed: {e}")

    async def _test_api_credentials(self, exchange_name, exchange):
        """Test API credentials"""
        try:
            # Test account access
            balance = await self._execute_with_retry(
                exchange.fetch_balance,
                exchange_name,
                "fetch_balance"
            )
            
            if balance:
                logger.info(f"✅ {exchange_name.title()} API credentials verified")
            else:
                logger.warning(f"⚠️ {exchange_name.title()} API credentials may be limited")
                
        except Exception as e:
            logger.warning(f"⚠️ {exchange_name.title()} API credential test failed: {e}")

    def _setup_monitoring(self):
        """Setup exchange monitoring and alerts"""
        try:
            # Initialize monitoring metrics
            for exchange_name in self.exchanges.keys():
                self.failed_requests[exchange_name] = 0
                self.last_health_check[exchange_name] = datetime.now()
            
            logger.info("✅ Exchange monitoring configured")
            
        except Exception as e:
            logger.error(f"❌ Monitoring setup failed: {e}")

    def _log_initialization_summary(self):
        """Log initialization summary"""
        total_exchanges = len(self.exchange_configs)
        connected_exchanges = len([s for s in self.connection_status.values() if s == 'connected'])
        backup_sources = len(self.backup_clients)
        
        logger.info("📊 EXCHANGE MANAGER INITIALIZATION SUMMARY:")
        logger.info(f"   • Primary Exchanges: {connected_exchanges}/{total_exchanges} connected")
        logger.info(f"   • Backup Data Sources: {backup_sources} available")
        logger.info(f"   • Monitoring: {'✅ Active' if connected_exchanges > 0 else '❌ Inactive'}")
        
        if connected_exchanges == 0:
            logger.error("❌ No exchanges connected - trading functionality limited")
        elif connected_exchanges < total_exchanges:
            logger.warning(f"⚠️ Only {connected_exchanges} exchanges connected - some functionality may be limited")
        else:
            logger.info("🚀 All exchanges connected - full functionality available")

    async def get_prices(self, exchange_name: str, pairs: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get current prices with enhanced error handling and failover"""
        try:
            if not pairs:
                pairs = self.config.CRYPTO_PAIRS
            
            # Check if exchange is available
            if exchange_name not in self.exchanges:
                logger.warning(f"⚠️ Exchange {exchange_name} not available, trying backup sources")
                return await self._get_backup_prices(pairs)
            
            # Check connection status
            if self.connection_status.get(exchange_name) != 'connected':
                logger.warning(f"⚠️ Exchange {exchange_name} not connected, trying failover")
                return await self._handle_exchange_failover(exchange_name, pairs)
            
            exchange = self.exchanges[exchange_name]
            
            # Get ticker data with retry logic
            prices = await self._fetch_prices_with_retry(exchange, exchange_name, pairs)
            
            if prices:
                self.performance_metrics['successful_requests'] += 1
                return prices
            else:
                # Fallback to backup sources
                return await self._get_backup_prices(pairs)
                
        except Exception as e:
            logger.error(f"❌ Failed to get prices from {exchange_name}: {e}")
            self.performance_metrics['failed_requests'] += 1
            
            # Try backup sources
            return await self._get_backup_prices(pairs)

    async def _fetch_prices_with_retry(self, exchange, exchange_name: str, pairs: List[str]) -> Dict[str, Any]:
        """Fetch prices with retry logic"""
        try:
            # Handle rate limiting
            await self._handle_rate_limiting(exchange_name)
            
            # Fetch tickers
            tickers = await self._execute_with_retry(
                exchange.fetch_tickers,
                exchange_name,
                "fetch_tickers"
            )
            
            if not tickers:
                return {}
            
            # Format price data
            prices = {}
            for symbol, ticker in tickers.items():
                if symbol in pairs:
                    prices[symbol] = {
                        'bid': ticker.get('bid', 0),
                        'ask': ticker.get('ask', 0),
                        'last': ticker.get('last', 0),
                        'volume': ticker.get('baseVolume', 0),
                        'timestamp': ticker.get('timestamp', int(time.time() * 1000)),
                        'source': exchange_name
                    }
            
            logger.debug(f"📊 Retrieved {len(prices)} prices from {exchange_name}")
            return prices
            
        except Exception as e:
            logger.error(f"❌ Price fetch failed for {exchange_name}: {e}")
            return {}

    async def _handle_rate_limiting(self, exchange_name: str):
        """Handle exchange rate limiting"""
        try:
            config = self.exchange_configs.get(exchange_name, {})
            rate_limit = config.get('rate_limit', 100)  # Default 100 requests per minute
            
            # Track request timestamps
            now = time.time()
            if exchange_name not in self.request_timestamps:
                self.request_timestamps[exchange_name] = []
            
            # Clean old timestamps
            self.request_timestamps[exchange_name] = [
                ts for ts in self.request_timestamps[exchange_name]
                if now - ts < 60  # Keep last minute
            ]
            
            # Check if we're hitting rate limits
            if len(self.request_timestamps[exchange_name]) >= rate_limit:
                sleep_time = 60 - (now - self.request_timestamps[exchange_name][0])
                if sleep_time > 0:
                    logger.warning(f"⏰ Rate limit reached for {exchange_name}, waiting {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
            
            # Add current request timestamp
            self.request_timestamps[exchange_name].append(now)
            
        except Exception as e:
            logger.error(f"❌ Rate limiting handler failed: {e}")

    async def _execute_with_retry(self, func, exchange_name: str, operation: str, *args, **kwargs):
        """Execute exchange function with retry logic"""
        for attempt in range(self.max_retries):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Reset failed request counter on success
                self.failed_requests[exchange_name] = 0
                return result
                
            except Exception as e:
                self.failed_requests[exchange_name] += 1
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"⚠️ {exchange_name} {operation} attempt {attempt + 1} failed: {e}")
                    logger.info(f"🔄 Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ {exchange_name} {operation} failed after {self.max_retries} attempts: {e}")
                    raise e

    async def _handle_exchange_failover(self, failed_exchange: str, pairs: List[str]) -> Dict[str, Any]:
        """Handle exchange failover to other connected exchanges"""
        try:
            # Find alternative connected exchanges
            alternative_exchanges = [
                name for name, status in self.connection_status.items()
                if status == 'connected' and name != failed_exchange
            ]
            
            if alternative_exchanges:
                alternative = alternative_exchanges[0]
                logger.info(f"🔄 Failing over from {failed_exchange} to {alternative}")
                return await self.get_prices(alternative, pairs)
            else:
                # No alternative exchanges available
                logger.warning("⚠️ No alternative exchanges available, using backup sources")
                return await self._get_backup_prices(pairs)
                
        except Exception as e:
            logger.error(f"❌ Exchange failover failed: {e}")
            return await self._get_backup_prices(pairs)

    async def _get_backup_prices(self, pairs: List[str]) -> Dict[str, Any]:
        """Get prices from backup data sources"""
        try:
            logger.info("🔄 Fetching prices from backup sources...")
            
            # Try CoinGecko first (most reliable free API)
            if 'coingecko' in self.backup_clients:
                prices = await self.backup_clients['coingecko'].get_crypto_prices(pairs)
                if prices:
                    logger.info(f"✅ Retrieved {len(prices)} prices from CoinGecko")
                    return prices
            
            # Try Alpha Vantage
            if 'alpha_vantage' in self.backup_clients:
                prices = await self.backup_clients['alpha_vantage'].get_crypto_prices(pairs)
                if prices:
                    logger.info(f"✅ Retrieved {len(prices)} prices from Alpha Vantage")
                    return prices
            
            # Try Finnhub
            if 'finnhub' in self.backup_clients:
                prices = await self.backup_clients['finnhub'].get_crypto_prices(pairs)
                if prices:
                    logger.info(f"✅ Retrieved {len(prices)} prices from Finnhub")
                    return prices
            
            logger.error("❌ All backup price sources failed")
            return {}
            
        except Exception as e:
            logger.error(f"❌ Backup price fetch failed: {e}")
            return {}

    async def get_ohlcv(self, exchange_name: str, symbol: str, timeframe: str, limit: int = 100) -> List[List]:
        """Get OHLCV data with enhanced error handling"""
        try:
            if exchange_name not in self.exchanges:
                logger.error(f"❌ Exchange {exchange_name} not available")
                return []
            
            if self.connection_status.get(exchange_name) != 'connected':
                logger.warning(f"⚠️ Exchange {exchange_name} not connected")
                return []
            
            exchange = self.exchanges[exchange_name]
            
            # Handle rate limiting  
            await self._handle_rate_limiting(exchange_name)
            
            # Fetch OHLCV data with retry
            ohlcv = await self._execute_with_retry(
                exchange.fetch_ohlcv,
                exchange_name,
                "fetch_ohlcv",
                symbol, timeframe, limit=limit
            )
            
            if ohlcv:
                logger.debug(f"📊 Retrieved {len(ohlcv)} OHLCV candles for {symbol} from {exchange_name}")
                return ohlcv
            else:
                return []
                
        except Exception as e:
            logger.error(f"❌ Failed to get OHLCV from {exchange_name}: {e}")
            return []

    async def get_account_balance(self, exchange_name: str) -> Optional[Dict[str, Any]]:
        """Get account balance with enhanced error handling"""
        try:
            if exchange_name not in self.exchanges:
                logger.error(f"❌ Exchange {exchange_name} not available")
                return None
            
            exchange = self.exchanges[exchange_name]
            
            if not exchange.apiKey:
                logger.error(f"❌ No API credentials for {exchange_name}")
                return None
            
            # Handle rate limiting
            await self._handle_rate_limiting(exchange_name)
            
            # Fetch balance with retry
            balance = await self._execute_with_retry(
                exchange.fetch_balance,
                exchange_name,
                "fetch_balance"
            )
            
            if balance:
                logger.debug(f"💰 Retrieved account balance from {exchange_name}")
                return balance
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get balance from {exchange_name}: {e}")
            return None

    async def place_order(self, exchange_name: str, symbol: str, order_type: str, 
                         side: str, amount: float, price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Place order with enhanced validation and error handling"""
        try:
            if exchange_name not in self.exchanges:
                logger.error(f"❌ Exchange {exchange_name} not available")
                return None
            
            exchange = self.exchanges[exchange_name]
            
            if not exchange.apiKey:
                logger.error(f"❌ No API credentials for {exchange_name}")
                return None
            
            # Validate order parameters
            if not self._validate_order_params(symbol, order_type, side, amount, price):
                return None
            
            # Handle rate limiting
            await self._handle_rate_limiting(exchange_name)
            
            # Place order with retry
            if order_type == 'market':
                order = await self._execute_with_retry(
                    exchange.create_market_order,
                    exchange_name,
                    "create_market_order",
                    symbol, side, amount
                )
            else:
                order = await self._execute_with_retry(
                    exchange.create_limit_order,
                    exchange_name,
                    "create_limit_order",
                    symbol, side, amount, price
                )
            
            if order:
                logger.info(f"✅ Order placed on {exchange_name}: {order.get('id', 'N/A')}")
                return order
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to place order on {exchange_name}: {e}")
            return None

    def _validate_order_params(self, symbol: str, order_type: str, side: str, 
                              amount: float, price: Optional[float]) -> bool:
        """Validate order parameters"""
        try:
            # Basic validation
            if not symbol or not order_type or not side:
                logger.error("❌ Missing required order parameters")
                return False
            
            if amount <= 0:
                logger.error("❌ Order amount must be positive")
                return False
            
            if order_type not in ['market', 'limit']:
                logger.error(f"❌ Unsupported order type: {order_type}")
                return False
            
            if side not in ['buy', 'sell']:
                logger.error(f"❌ Invalid order side: {side}")
                return False
            
            if order_type == 'limit' and (not price or price <= 0):
                logger.error("❌ Limit orders require a positive price")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Order validation failed: {e}")
            return False

    async def monitor_exchanges(self):
        """Monitor exchange health and performance"""
        try:
            while True:
                for exchange_name in self.exchanges.keys():
                    # Check if health check is needed
                    last_check = self.last_health_check.get(exchange_name, datetime.min)
                    if (datetime.now() - last_check).total_seconds() > 300:  # 5 minutes
                        await self._check_exchange_health(exchange_name, self.exchanges[exchange_name])
                        self.last_health_check[exchange_name] = datetime.now()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Check every minute
                
        except Exception as e:
            logger.error(f"❌ Exchange monitoring failed: {e}")

    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            self.performance_metrics['exchanges_online'] = len([
                status for status in self.connection_status.values() 
                if status == 'connected'
            ])
            
            total_requests = self.performance_metrics['successful_requests'] + self.performance_metrics['failed_requests']
            self.performance_metrics['total_requests'] = total_requests
            
            if total_requests > 0:
                success_rate = self.performance_metrics['successful_requests'] / total_requests
                logger.debug(f"📊 Exchange success rate: {success_rate:.2%}")
                
        except Exception as e:
            logger.error(f"❌ Performance metrics update failed: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.performance_metrics,
            'connection_status': self.connection_status,
            'failed_requests_by_exchange': self.failed_requests
        }

    def get_connected_exchanges(self) -> List[str]:
        """Get list of connected exchanges"""
        return [
            name for name, status in self.connection_status.items()
            if status == 'connected'
        ]


# Backup data source clients
class CoinGeckoClient:
    """CoinGecko API client for backup price data"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = None
    
    async def get_crypto_prices(self, pairs: List[str]) -> Dict[str, Any]:
        """Get crypto prices from Finnhub"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            prices = {}
            
            for pair in pairs:
                # Convert pair format (BTC/USDT -> BINANCE:BTCUSDT)
                symbol = f"BINANCE:{pair.replace('/', '')}"
                
                url = f"{self.base_url}/quote"
                params = {
                    'symbol': symbol,
                    'token': self.api_key
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'c' in data:  # Current price
                            current_price = data['c']
                            prices[pair] = {
                                'bid': current_price * 0.999,
                                'ask': current_price * 1.001,
                                'last': current_price,
                                'volume': data.get('v', 0),
                                'timestamp': int(time.time() * 1000),
                                'source': 'finnhub'
                            }
                
                # Rate limiting
                await asyncio.sleep(0.1)
            
            return prices
            
        except Exception as e:
            logger.error(f"❌ Finnhub price fetch failed: {e}")
            return {}
        """Get crypto prices from CoinGecko"""
        try:
            # Convert pairs to CoinGecko format
            coin_ids = self._convert_pairs_to_coin_ids(pairs)
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': ','.join(coin_ids),
                'vs_currencies': 'usd',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._format_coingecko_prices(data, pairs)
                else:
                    logger.error(f"❌ CoinGecko API error: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"❌ CoinGecko price fetch failed: {e}")
            return {}
    
    def _convert_pairs_to_coin_ids(self, pairs: List[str]) -> List[str]:
        """Convert trading pairs to CoinGecko coin IDs"""
        mapping = {
            'BTC/USDT': 'bitcoin',
            'ETH/USDT': 'ethereum',
            'BNB/USDT': 'binancecoin',
            'ADA/USDT': 'cardano',
            'DOT/USDT': 'polkadot',
            # Add more mappings as needed
        }
        
        return [mapping.get(pair, pair.split('/')[0].lower()) for pair in pairs]
    
    def _format_coingecko_prices(self, data: Dict, pairs: List[str]) -> Dict[str, Any]:
        """Format CoinGecko data to standard format"""
        formatted = {}
        
        for pair in pairs:
            coin_id = self._convert_pairs_to_coin_ids([pair])[0]
            if coin_id in data:
                coin_data = data[coin_id]
                price = coin_data.get('usd', 0)
                
                formatted[pair] = {
                    'bid': price * 0.999,  # Approximate bid
                    'ask': price * 1.001,  # Approximate ask
                    'last': price,
                    'volume': coin_data.get('usd_24h_vol', 0),
                    'timestamp': int(time.time() * 1000),
                    'source': 'coingecko'
                }
        
        return formatted


class AlphaVantageClient:
    """Alpha Vantage API client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = None
    
    async def get_crypto_prices(self, pairs: List[str]) -> Dict[str, Any]:
        """Get crypto prices from Alpha Vantage"""
        # Implementation would go here
        return {}


class FinnhubClient:
    """Finnhub API client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.session = None
    
    async def get_crypto_prices(self, pairs: List[str]) -> Dict[str, Any]:
