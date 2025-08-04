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
        
        # Convert timeframe to MetaAPI format
        timeframe_map = {
            '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
            '1h': 'H1', '4h': 'H4', '1d': 'D1', '1w': 'W1'
        }
        mt_timeframe = timeframe_map.get(timeframe, 'H1')
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=limit)
        
        # Try multiple API endpoints
        endpoints_to_try = [
            # Option 1: Standard candles endpoint
            f'https://mt-client-api-v1.agiliumtrade.agiliumtrade.ai/users/current/accounts/{self.account_id}/historical-market-data/symbols/{symbol}/candles',
            
            # Option 2: With specific timeframe
            f'https://mt-client-api-v1.agiliumtrade.agiliumtrade.ai/users/current/accounts/{self.account_id}/historical-market-data/symbols/{symbol}/timeframes/{mt_timeframe}/candles',
            
            # Option 3: Alternative format
            f'https://mt-client-api-v1.agiliumtrade.agiliumtrade.ai/users/current/accounts/{self.account_id}/candles/{symbol}'
        ]
        
        for endpoint in endpoints_to_try:
            try:
                logger.info(f"🔍 Trying endpoint: {endpoint}")
                
                response = requests.get(
                    endpoint,
                    headers=headers,
                    params={
                        'startTime': start_time.isoformat(),
                        'endTime': end_time.isoformat(),
                        'limit': limit,
                        'timeframe': mt_timeframe  # Add timeframe as parameter
                    },
                    timeout=10,
                    verify=False
                )
                
                logger.info(f"📊 Response status: {response.status_code}")
                
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
                            candle.get('volume', 1000)
                        ])
                    
                    logger.info(f"✅ Retrieved {len(ohlcv)} LIVE candles for {symbol}")
                    return ohlcv
                    
                elif response.status_code == 404:
                    logger.warning(f"⚠️ Endpoint not found: {endpoint}")
                    continue  # Try next endpoint
                    
                else:
                    logger.error(f"❌ OHLCV request failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to fetch from {endpoint}: {e}")
                continue
        
        # If all endpoints fail, try getting current price instead
        logger.warning("⚠️ All historical data endpoints failed, trying current price")
        return await self._get_current_price_as_ohlcv(symbol)
        
    except Exception as e:
        logger.error(f"❌ Failed to get forex OHLCV: {e}")
        return []

async def _get_current_price_as_ohlcv(self, symbol):
    """Fallback: Get current price and create OHLCV-like data"""
    try:
        headers = {
            'auth-token': self.mt_token,
            'Content-Type': 'application/json'
        }
        
        response = requests.get(
            f'https://mt-client-api-v1.agiliumtrade.agiliumtrade.ai/users/current/accounts/{self.account_id}/symbols/{symbol}/current-price',
            headers=headers,
            timeout=5,
            verify=False
        )
        
        if response.status_code == 200:
            price_data = response.json()
            current_price = (price_data.get('bid', 0) + price_data.get('ask', 0)) / 2
            
            # Create basic OHLCV with current price
            ohlcv = [[
                int(datetime.now().timestamp()),
                current_price,
                current_price,
                current_price,
                current_price,
                1000
            ]]
            
            logger.info(f"📊 Using current price for {symbol}: {current_price}")
            return ohlcv
            
    except Exception as e:
        logger.error(f"❌ Fallback price fetch failed: {e}")
        
    return []
