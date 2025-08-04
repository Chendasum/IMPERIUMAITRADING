
import asyncio
import logging
from datetime import datetime, timedelta
import requests
import json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

class MetaTraderManager:
    def __init__(self, config):
        self.config = config
        self.mt_token = config.METAAPI_TOKEN
        self.account_id = getattr(config, 'METAAPI_ACCOUNT_ID', None)
        self.connection = None
        self.forex_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
            'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY'
        ]
        self.base_urls = {
            'provisioning': 'https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai',
            'client': 'https://mt-client-api-48ec6j1qlzb1cpk1.new-york.agiliumtrade.ai',
            'streaming': 'https://mt-market-data-client-api-48ec6j1qlzb1cpk1.new-york.agiliumtrade.ai'
        }

    async def initialize(self):
        try:
            if not self.mt_token:
                logger.error("❌ MetaAPI token missing.")
                return False

            logger.info("🚀 Connecting to MetaAPI...")
            headers = {'auth-token': self.mt_token, 'Content-Type': 'application/json'}
            response = requests.get(f'{self.base_urls["provisioning"]}/users/current/accounts',
                                    headers=headers, timeout=10, verify=False)

            if response.status_code == 200:
                accounts = response.json()
                if not accounts:
                    logger.error("❌ No accounts found.")
                    return False

                if self.account_id and not any(acc['_id'] == self.account_id for acc in accounts):
                    logger.error(f"❌ Account ID {self.account_id} not found.")
                    return False
                self.account_id = self.account_id or accounts[0]['_id']

                logger.info(f"✅ Connected to MetaAPI account: {self.account_id}")
                return True
            else:
                logger.error(f"❌ Authentication failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            return False

    async def get_account_balance(self):
        if not self.account_id:
            logger.error("❌ No account connected.")
            return None
        try:
            headers = {'auth-token': self.mt_token}
            url = f'{self.base_urls["client"]}/users/current/accounts/{self.account_id}/account-information'
            response = requests.get(url, headers=headers, timeout=10, verify=False)

            if response.status_code == 200:
                info = response.json()
                return {
                    'balance': info.get('balance', 0),
                    'equity': info.get('equity', 0),
                    'currency': info.get('currency', 'USD'),
                    'free_margin': info.get('freeMargin', 0),
                    'margin_level': info.get('marginLevel', 0)
                }
            logger.warning(f"⚠️ Failed to retrieve balance: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Balance fetch error: {e}")
        return None

    async def get_forex_prices(self):
        if not self.mt_token or not self.account_id:
            logger.error("❌ Not initialized for price fetch.")
            return {}

        headers = {'auth-token': self.mt_token}
        prices, count = {}, 0

        for pair in self.forex_pairs:
            try:
                url = f'{self.base_urls["streaming"]}/users/current/accounts/{self.account_id}/symbols/{pair}/current-price'
                response = requests.get(url, headers=headers, timeout=5, verify=False)

                if response.status_code == 200:
                    d = response.json()
                    prices[pair] = {
                        'bid': d.get('bid', 0),
                        'ask': d.get('ask', 0),
                        'last': (d.get('bid', 0) + d.get('ask', 0)) / 2,
                        'spread': d.get('ask', 0) - d.get('bid', 0),
                        'timestamp': d.get('time', datetime.now().isoformat())
                    }
                    count += 1
            except Exception as e:
                logger.debug(f"⚠️ Price error for {pair}: {e}")
            await asyncio.sleep(0.1)

        if count:
            logger.info(f"✅ {count} prices from MetaAPI")
            return prices
        logger.warning("⚠️ Prices failed. Trying fallback.")
        return await self._get_free_forex_prices()

    async def _get_free_forex_prices(self):
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                url = "https://api.exchangerate-api.com/v4/latest/USD"
                async with s.get(url, timeout=10) as r:
                    if r.status == 200:
                        d = await r.json()
                        rates = d.get('rates', {})
                        prices = {}

                        for p in self.forex_pairs:
                            base, quote = p[:3], p[3:]
                            if base == 'USD' and quote in rates:
                                rate = rates[quote]
                            elif quote == 'USD' and base in rates:
                                rate = 1 / rates[base]
                            elif base in rates and quote in rates:
                                rate = rates[quote] / rates[base]
                            else:
                                continue

                            prices[p] = {
                                'bid': rate * 0.9999,
                                'ask': rate * 1.0001,
                                'last': rate,
                                'spread': rate * 0.0002,
                                'timestamp': datetime.now().isoformat()
                            }

                        logger.info(f"✅ Fallback prices: {len(prices)} pairs")
                        return prices
        except Exception as e:
            logger.error(f"❌ Fallback failed: {e}")
        return {}

    def get_supported_pairs(self):
        return self.forex_pairs
