import asyncio
import logging
from datetime import datetime
import requests
import json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class MetaTraderManager:
    def __init__(self, config):
        self.config = config
        self.mt_token = config.METAAPI_TOKEN
        self.account_id = None
        self.connection = None
        self.forex_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 
            'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY'
        ]
        
    async def initialize(self):
        """Initialize MetaTrader connection"""
        try:
            if not self.mt_token:
                logger.error("❌ MetaAPI token required for live forex trading")
                return False
                
            # Initialize real MetaAPI connection
            logger.info("🚀 Initializing LIVE MetaAPI connection...")
            
            # Real MetaAPI initialization
            import requests
            
            headers = {
                'auth-token': self.mt_token,
                'Content-Type': 'application/json'
            }
            
            # Get accounts from MetaAPI
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                
                response = requests.get(
                    'https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai/users/current/accounts',
                    headers=headers,
                    timeout=10,
                    verify=False  # Skip SSL verification for MetaAPI
                )
                
                if response.status_code == 200:
                    accounts = response.json()
                    if accounts:
                        self.account_id = accounts[0]['_id']
                        logger.info(f"✅ MetaAPI connected - Account: {self.account_id}")
                        logger.info("💰 LIVE FOREX TRADING ACTIVE")
                        return True
                    else:
                        logger.error("❌ No MetaTrader accounts found")
                        return False
                else:
                    logger.error(f"❌ MetaAPI authentication failed: {response.status_code}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ MetaAPI connection failed: {e}")
                return False
            
        except Exception as e:
            logger.error(f"❌ MetaTrader initialization failed: {e}")
            return False
    
    async def get_forex_prices(self):
        """Get current forex prices from MetaAPI"""
        try:
            if not self.mt_token or not self.account_id:
                logger.error("❌ MetaAPI not initialized for price fetching")
                return {}
            
            import requests
            
            headers = {
                'auth-token': self.mt_token,
                'Content-Type': 'application/json'
            }
            
            # Get live prices from MetaAPI
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                
                prices = {}
                for symbol in self.forex_pairs:
                    response = requests.get(
                        f'https://mt-client-api-v1.agiliumtrade.agiliumtrade.ai/users/current/accounts/{self.account_id}/symbols/{symbol}/current-price',
                        headers=headers,
                        timeout=5,
                        verify=False  # Skip SSL verification for MetaAPI
                    )
                    
                    if response.status_code == 200:
                        price_data = response.json()
                        prices[symbol] = {
                            'bid': price_data.get('bid', 0),
                            'ask': price_data.get('ask', 0),
                            'last': (price_data.get('bid', 0) + price_data.get('ask', 0)) / 2,
                            'spread': price_data.get('ask', 0) - price_data.get('bid', 0),
                            'timestamp': price_data.get('time', datetime.now())
                        }
                
                logger.info(f"💱 Retrieved {len(prices)} LIVE forex prices from MetaAPI")
                return prices
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch live forex prices: {e}")
                return {}
            
        except Exception as e:
            logger.error(f"❌ Failed to get forex prices: {e}")
            return {}  # NO SIMULATION - REAL PRICES ONLY
    
    # ALL FOREX SIMULATION FUNCTIONS REMOVED - LIVE METAAPI ONLY
    
    async def get_forex_ohlcv(self, symbol, timeframe='1h', limit=50):
        """Get forex OHLCV data from MetaAPI"""
        try:
            if not self.mt_token or not self.account_id:
                logger.error("❌ MetaAPI not initialized for OHLCV data")
                return []
            
            import requests
            from datetime import datetime, timedelta
            
            headers = {
                'auth-token': self.mt_token,
                'Content-Type': 'application/json'
            }
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=limit)
            
            # Get historical data from MetaAPI
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                
                response = requests.get(
                    f'https://mt-client-api-v1.agiliumtrade.agiliumtrade.ai/users/current/accounts/{self.account_id}/historical-market-data/symbols/{symbol}/timeframes/1h/candles',
                    headers=headers,
                    params={
                        'startTime': start_time.isoformat(),
                        'endTime': end_time.isoformat(),
                        'limit': limit
                    },
                    timeout=10,
                    verify=False  # Skip SSL verification for MetaAPI
                )
                
                if response.status_code == 200:
                    candles = response.json()
                    ohlcv = []
                    
                    for candle in candles:
                        ohlcv.append([
                            candle.get('time', 0),
                            candle.get('open', 0),
                            candle.get('high', 0),
                            candle.get('low', 0),
                            candle.get('close', 0),
                            candle.get('volume', 1000)  # Tick volume
                        ])
                    
                    logger.info(f"💱 Retrieved {len(ohlcv)} LIVE candles for {symbol}")
                    return ohlcv
                else:
                    logger.error(f"❌ OHLCV request failed: {response.status_code}")
                    return []
                    
            except Exception as e:
                logger.error(f"❌ Failed to fetch OHLCV data: {e}")
                return []
            
        except Exception as e:
            logger.error(f"❌ Failed to get forex OHLCV: {e}")
            return []  # NO SIMULATION - REAL METAAPI DATA ONLY
    
    # ALL SIMULATION FUNCTIONS COMPLETELY REMOVED - 100% LIVE TRADING ONLY
    
    async def execute_forex_trade(self, signal, position_size):
        """Execute live forex trade through MetaAPI"""
        try:
            if not self.mt_token or not self.account_id:
                logger.error("❌ MetaAPI not initialized for trade execution")
                return 0
            
            import requests
            
            headers = {
                'auth-token': self.mt_token,
                'Content-Type': 'application/json'
            }
            
            # Calculate trade parameters
            symbol = signal['pair']
            action_type = 'ORDER_TYPE_BUY' if signal['action'] == 'buy' else 'ORDER_TYPE_SELL'
            volume = round(position_size / 100000, 2)  # Convert to lots
            
            # Prepare trade request
            trade_request = {
                'actionType': 'TRADE_ACTION_DEAL',
                'symbol': symbol,
                'volume': volume,
                'type': action_type,
                'comment': 'IMPERIUM AI Trading',
                'magic': 12345
            }
            
            logger.info(f"💰 EXECUTING LIVE FOREX TRADE: {symbol} {action_type} {volume} lots")
            
            # Execute trade through MetaAPI
            try:
                response = requests.post(
                    f'https://mt-client-api-v1.agiliumtrade.agiliumtrade.ai/users/current/accounts/{self.account_id}/trade',
                    headers=headers,
                    json=trade_request,
                    timeout=10,
                    verify=False  # Skip SSL verification for MetaAPI
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('stringCode') == 'TRADE_RETCODE_DONE':
                        # Trade successful
                        logger.info(f"✅ LIVE FOREX TRADE EXECUTED: Order {result.get('order')}")
                        
                        # Calculate expected profit
                        price_diff = signal['target'] - signal['price']
                        if signal['action'] == 'sell':
                            price_diff = -price_diff
                            
                        profit = price_diff * volume * 100000  # Convert to USD
                        logger.info(f"💰 Expected profit: ${profit:.2f}")
                        return profit
                    else:
                        logger.error(f"❌ Trade failed: {result.get('message', 'Unknown error')}")
                        return 0
                else:
                    logger.error(f"❌ Trade request failed: {response.status_code}")
                    return 0
                    
            except Exception as e:
                logger.error(f"❌ Live forex trade execution failed: {e}")
                return 0
            
        except Exception as e:
            logger.error(f"❌ Forex trade execution failed: {e}")
            return 0
    
    # SIMULATION FUNCTIONS REMOVED - LIVE FOREX TRADING ONLY
    
    # ALL ENHANCED SIMULATION FUNCTIONS REMOVED - REAL METAAPI ONLY
    
    # ALL SIMULATION HELPER FUNCTIONS REMOVED - 100% REAL METAAPI TRADING
    
    def get_supported_pairs(self):
        """Get list of supported forex pairs"""
        return self.forex_pairs
