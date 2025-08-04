# utils/metaapi_client.py

import os
import requests

def get_forex_quote(symbol="EURUSD"):
    """
    Fetch real-time Forex quote using MetaAPI.
    Falls back to Finnhub or CurrencyLayer if MetaAPI fails.
    """
    metaapi_token = os.environ.get("METAAPI_TOKEN")
    account_id = os.environ.get("METAAPI_ACCOUNT_ID")
    base_url = "https://mt-client-api-48ec6j1qlzb1cpk1.new-york.agiliumtrade.ai"  # Replace with your actual URL

    headers = {
        "Authorization": f"Bearer {metaapi_token}",
        "Content-Type": "application/json"
    }

    url = f"{base_url}/users/current/accounts/{account_id}/quotes/{symbol}"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        price = data.get("bid", 0) or data.get("price", 0)
        if price:
            print(f"✅ MetaAPI price for {symbol}: {price}")
            return price
        else:
            print(f"⚠️ MetaAPI returned no price for {symbol}, falling back...")
            return fallback_quote(symbol)

    except Exception as e:
        print(f"⚠️ MetaAPI ERROR: {e}")
        return fallback_quote(symbol)


def fallback_quote(symbol="EURUSD"):
    """
    Try backup APIs (Finnhub > CurrencyLayer) if MetaAPI fails.
    """
    # Finnhub fallback
    try:
        print("🔁 Trying Finnhub fallback...")
        from utils.finnhub_client import get_finnhub_price
        return get_finnhub_price(symbol)
    except Exception as e:
        print(f"⚠️ Finnhub failed: {e}")

    # CurrencyLayer fallback
    try:
        print("🔁 Trying CurrencyLayer fallback...")
        from utils.currency_layer_client import get_currency_layer_price
        return get_currency_layer_price(symbol)
    except Exception as e:
        print(f"⚠️ CurrencyLayer failed: {e}")

    print("❌ All data sources failed.")
    return None
