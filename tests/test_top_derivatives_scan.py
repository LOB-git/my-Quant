import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

MODULE_PATH = Path(__file__).resolve().parents[1] / 'dashboard.py'


def load_app_module():
    spec = importlib.util.spec_from_file_location('quant_app', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DerivativeMarketFallbackTests(unittest.TestCase):
    def test_blocked_public_apis_return_empty_symbols_without_crashing(self):
        app = load_app_module()
        with patch.object(app.ccxt, 'binanceus', side_effect=RuntimeError('blocked')):
            with patch.object(app.ccxt, 'binance', side_effect=RuntimeError('blocked')):
                with patch.object(app.ccxt, 'bybit', side_effect=RuntimeError('blocked')):
                    with patch.object(app.requests, 'get', side_effect=requests.exceptions.RequestException('blocked')):
                        self.assertEqual(app.resolve_derivative_market_symbols(), [])

    def test_http_fallback_parses_usdt_perpetual_symbols(self):
        app = load_app_module()

        class MockResponse:
            def __init__(self, payload):
                self._payload = payload

            def json(self):
                return self._payload

        payload = {
            'result': {
                'list': [
                    {'symbol': 'BTCUSDT', 'category': 'linear'},
                    {'symbol': 'ETHUSDT', 'category': 'linear'},
                    {'symbol': 'XRPUSDT', 'category': 'spot'},
                ]
            }
        }

        with patch.object(app.ccxt, 'binanceus', side_effect=RuntimeError('blocked')):
            with patch.object(app.ccxt, 'binance', side_effect=RuntimeError('blocked')):
                with patch.object(app.ccxt, 'bybit', side_effect=RuntimeError('blocked')):
                    with patch.object(app.requests, 'get', return_value=MockResponse(payload)):
                        symbols = app.resolve_derivative_market_symbols()
                        self.assertIn('BTC/USDT', symbols)
                        self.assertIn('ETH/USDT', symbols)
                        self.assertNotIn('XRP/USDT', symbols)

    def test_historical_z_score_components_include_spy_eminis(self):
        app = load_app_module()
        self.assertIn('ES=F', app.HISTORICAL_Z_COMPONENT_SYMBOLS)
        self.assertIn('ES=F', app.indices_for_qs)


if __name__ == '__main__':
    unittest.main()
