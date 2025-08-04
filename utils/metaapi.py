import os
import requests
import logging

# Optional: Load from .env if running locally
from dotenv import load_dotenv
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetaAPI Debug")

# Configuration
METAAPI_TOKEN = os.getenv("METAAPI_TOKEN")
METAAPI_ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID")
BASE_URL = "https://mt-client-api-48ec6j1qlzb1cpk1.new-york.agiliumtrade.ai"  # ✅ Your verified endpoint

def test_metaapi_quote(symbol="EURUSD"):
    """
    Test MetaAPI connection and fetch live Forex quote
    """
    if not METAAPI_TOKEN or not METAAPI_ACCOUNT_ID:
        logger.error("❌ Missing METAAPI_TOKEN or METAAPI_ACCOUNT_ID environment variables.")
        return

    url = f"{BASE_URL}/users/current/accounts/{METAAPI_ACCOUNT_ID}/quotes/{symbol}"
    headers = {
        "Authorization": f"Bearer {METAAPI_TOKEN}",
        "Content-Type": "application/json"
    }

    logger.info(f"🔍 Requesting quote for: {symbol}")
    logger.info(f"📡 Endpoint: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        logger.info(f"📄 Status Code: {response.status_code}")
        logger.info(f"📦 Raw Response: {response.text}")

        if response.status_code == 200:
            data = response.json()
            bid = data.get("bid")
            ask = data.get("ask")
            price = bid or data.get("price")
            logger.info(f"✅ Bid: {bid} | Ask: {ask} | Price: {price}")
            return price
        else:
            logger.error(f"❌ API Error: {response.status_code} - {response.text}")
    except Exception as e:
        logger.exception(f"❌ Request Exception: {e}")

    return None


if __name__ == "__main__":
    print("=" * 40)
    print("🧪 METAAPI DEBUG TEST")
    print("=" * 40)
    result = test_metaapi_quote("EURUSD")
    if result:
        print(f"✅ MetaAPI LIVE PRICE: {result}")
    else:
        print("❌ MetaAPI test failed — check logs above.")
