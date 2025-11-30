"""
Babs AI Trading System - API Connectors
Multi-source data fetching and AI model connections with fallback
"""
import os
import requests
from config import Config

# Conditional imports for AI clients
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    genai = None


class AIModelConnector:
    """
    Handles connections to multiple AI APIs (Bytez, OpenRouter, Google GenAI)
    with a built-in fallback mechanism.
    """
    def __init__(self):
        # API Keys from environment variables
        self.bytez_key = Config.BYTZE_KEY
        self.openrouter_key = Config.OPENROUTER_KEY
        self.google_genai_key = Config.GOOGLE_GENAI_KEY or Config.OPENAI_KEY

        # Initialize clients
        self.bytez_client = self._init_openai_compatible_client(
            api_key=self.bytez_key,
            base_url="https://api.bytez.com/v1"
        )
        self.openrouter_client = self._init_openai_compatible_client(
            api_key=self.openrouter_key,
            base_url="https://openrouter.ai/api/v1"
        )
        # Google GenAI client
        self.google_genai_client = self._init_google_genai_client(self.google_genai_key)

        # Define preferred models
        self.text_model = "gpt-4-turbo-preview"
        self.image_model = "dall-e-2"
        self.google_text_model = "gemini-pro"

    def _init_openai_compatible_client(self, api_key, base_url):
        """Initialize OpenAI-compatible client"""
        if api_key and HAS_OPENAI:
            try:
                return OpenAI(api_key=api_key, base_url=base_url)
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
        return None

    def _init_google_genai_client(self, api_key):
        """Initialize Google GenAI client"""
        if api_key and HAS_GENAI:
            try:
                genai.configure(api_key=api_key)
                return genai
            except Exception as e:
                print(f"Failed to initialize Google GenAI: {e}")
        return None

    def generate_text(self, prompt: str, system_prompt: str = None):
        """
        Generates text using Bytez -> OpenRouter -> Google GenAI fallback.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 1. Try Bytez
        if self.bytez_client:
            try:
                response = self.bytez_client.chat.completions.create(
                    model=self.text_model,
                    messages=messages,
                    temperature=0.7
                )
                return response.choices[0].message.content, "Bytez"
            except Exception as e:
                print(f"Bytez text generation failed: {e}")

        # 2. Try OpenRouter
        if self.openrouter_client:
            try:
                response = self.openrouter_client.chat.completions.create(
                    model="openai/gpt-4-turbo-preview",
                    messages=messages,
                    temperature=0.7
                )
                return response.choices[0].message.content, "OpenRouter"
            except Exception as e:
                print(f"OpenRouter text generation failed: {e}")

        # 3. Try Google GenAI
        if self.google_genai_client:
            try:
                model = self.google_genai_client.GenerativeModel(self.google_text_model)
                response = model.generate_content(prompt)
                return response.text, "Google GenAI"
            except Exception as e:
                print(f"Google GenAI text generation failed: {e}")

        return "Error: All AI text services failed.", None

    def generate_image(self, prompt: str):
        """
        Generates an image URL using Bytez -> OpenRouter fallback.
        """
        # 1. Try Bytez
        if self.bytez_client:
            try:
                response = self.bytez_client.images.generate(
                    model=self.image_model,
                    prompt=prompt,
                    n=1,
                    size="512x512"
                )
                return response.data[0].url, "Bytez"
            except Exception as e:
                print(f"Bytez image generation failed: {e}")

        # 2. Try OpenRouter
        if self.openrouter_client:
            try:
                response = self.openrouter_client.images.generate(
                    model=self.image_model,
                    prompt=prompt,
                    n=1,
                    size="512x512"
                )
                return response.data[0].url, "OpenRouter"
            except Exception as e:
                print(f"OpenRouter image generation failed: {e}")

        return "Error: All AI image services failed.", None


class MultiSourceDataFetcher:
    """
    Fetches market data from Alpha Vantage with a fallback to Binance for crypto.
    """
    def __init__(self):
        self.alpha_vantage_key = Config.ALPHA_VANTAGE_KEY

    def get_validated_data(self, symbol: str):
        """Get validated market data from multiple sources"""
        # Check if the symbol is a crypto pair
        if symbol.upper().endswith("USDT") or symbol.upper().endswith("USD") and "BTC" in symbol.upper():
            # Try Alpha Vantage first for crypto
            try:
                data = self._get_alpha_vantage_crypto(symbol)
                if data:
                    return data
            except Exception as e:
                print(f"Alpha Vantage crypto fetch failed for {symbol}: {e}")
            
            # Fallback to Binance public API for crypto
            try:
                data = self._get_binance_crypto(symbol)
                if data:
                    return data
            except Exception as e:
                print(f"Binance crypto fetch failed for {symbol}: {e}")
                raise Exception(f"All crypto data sources failed for {symbol}")

        else:
            # For non-crypto, use Alpha Vantage
            try:
                return self._get_alpha_vantage_forex(symbol)
            except Exception as e:
                print(f"Alpha Vantage forex fetch failed for {symbol}: {e}")
                # Return mock data for demonstration
                return self._get_mock_data(symbol)

    def _get_alpha_vantage_crypto(self, symbol: str):
        """Get crypto data from Alpha Vantage"""
        if not self.alpha_vantage_key:
            return None
        from_currency = symbol.upper().replace("USDT", "").replace("/", "")
        url = f"https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&from_symbol={from_currency}&to_symbol=USD&market=USD&interval=60min&apikey={self.alpha_vantage_key}"
        response = requests.get(url, timeout=10)
        data = response.json()
        if "Time Series Crypto (60min)" in data:
            latest_timestamp = sorted(data["Time Series Crypto (60min)"].keys())[0]
            latest_data = data["Time Series Crypto (60min)"][latest_timestamp]
            return {
                "open": float(latest_data["1. open"]),
                "high": float(latest_data["2. high"]),
                "low": float(latest_data["3. low"]),
                "close": float(latest_data["4. close"]),
                "volume": float(latest_data["5. volume"]),
                "source": "Alpha Vantage"
            }
        return None

    def _get_binance_crypto(self, symbol: str):
        """Get crypto data from Binance public API"""
        binance_symbol = symbol.upper().replace("/", "")
        url = f"https://api.binance.com/api/v3/klines?symbol={binance_symbol}&interval=1h&limit=1"
        response = requests.get(url, timeout=10)
        data = response.json()
        if data and len(data) > 0:
            kline = data[0]
            return {
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
                "source": "Binance (Public)"
            }
        return None

    def _get_alpha_vantage_forex(self, symbol: str):
        """Get forex data from Alpha Vantage"""
        if not self.alpha_vantage_key:
            return self._get_mock_data(symbol)
        parts = symbol.split("/")
        if len(parts) != 2:
            return self._get_mock_data(symbol)
        from_symbol, to_symbol = parts
        url = f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={from_symbol}&to_symbol={to_symbol}&interval=60min&apikey={self.alpha_vantage_key}"
        response = requests.get(url, timeout=10)
        data = response.json()
        if "Time Series FX (60min)" in data:
            latest_timestamp = sorted(data["Time Series FX (60min)"].keys())[0]
            latest_data = data["Time Series FX (60min)"][latest_timestamp]
            return {
                "open": float(latest_data["1. open"]),
                "high": float(latest_data["2. high"]),
                "low": float(latest_data["3. low"]),
                "close": float(latest_data["4. close"]),
                "source": "Alpha Vantage"
            }
        return self._get_mock_data(symbol)

    def _get_mock_data(self, symbol: str):
        """Return mock data for demonstration"""
        import random
        base_price = 1.0950 if 'EUR' in symbol else 183.50 if 'JPY' in symbol else 1985.0
        volatility = 0.002 if 'EUR' in symbol else 0.5 if 'JPY' in symbol else 10.0
        
        return {
            "open": base_price + random.uniform(-volatility, volatility),
            "high": base_price + abs(random.uniform(0, volatility)),
            "low": base_price - abs(random.uniform(0, volatility)),
            "close": base_price + random.uniform(-volatility, volatility),
            "volume": random.randint(1000, 10000),
            "source": "Mock Data"
        }
