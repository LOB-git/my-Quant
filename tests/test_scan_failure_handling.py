import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import ccxt

import binance_futures as bf

MODULE_PATH = Path(__file__).resolve().parents[1] / 'dashboard.py'
MARKET = {
    'symbol': 'BTC/USDT:USDT', 'base': 'BTC', 'quote': 'USDT', 'swap': True, 'linear': True,
    'active': True, 'info': {'underlyingType': 'COIN'},
}


def load_app():
    spec = importlib.util.spec_from_file_location('quant_app_scan', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeUsdm:
    id = 'binanceusdm'

    def __init__(self, ticker_errors):
        self.ticker_errors = list(ticker_errors)
        self.ticker_calls = 0

    def fetch_markets(self):
        return [MARKET]

    def fetch_tickers(self):
        self.ticker_calls += 1
        if self.ticker_errors:
            raise self.ticker_errors.pop(0)
        return {'BTC/USDT:USDT': {'quoteVolume': 1000.0, 'active': True}}

    def fetch_funding_rates(self):
        return {}

    def fetch_open_interests(self):
        return {}

    def fetch_ohlcv(self, *_, **__):
        return None


class RetryTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(bf.time, 'sleep')
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def failing_then_ok(self, failures):
        errors = list(failures)

        def call():
            if errors:
                raise errors.pop(0)
            return 'ok'
        return call

    def test_brief_network_faults_are_retried(self):
        self.assertEqual(bf.with_retries(self.failing_then_ok(
            [ccxt.RequestTimeout('slow'), ccxt.ExchangeNotAvailable('502')],
        )), 'ok')
        self.assertEqual(self.sleep.call_count, 2)

    def test_gives_up_after_the_last_attempt(self):
        with self.assertRaises(ccxt.RequestTimeout):
            bf.with_retries(self.failing_then_ok([ccxt.RequestTimeout('slow')] * 3))

    def test_ban_and_region_block_are_not_retried(self):
        for message in ['binanceusdm GET x 418 banned', 'binanceusdm GET x 451 restricted']:
            with self.assertRaises(ccxt.DDoSProtection):
                bf.with_retries(self.failing_then_ok([ccxt.DDoSProtection(message), None]))
        self.sleep.assert_not_called()

    def test_non_network_errors_are_not_retried(self):
        with self.assertRaises(ccxt.BadSymbol):
            bf.with_retries(self.failing_then_ok([ccxt.BadSymbol('x')]))
        self.sleep.assert_not_called()


class FailureMessageTests(unittest.TestCase):
    def test_each_cause_gets_its_own_message(self):
        cases = [
            (ccxt.DDoSProtection('GET x 418 teapot'), 'banned'),
            (ccxt.RateLimitExceeded('GET x 429'), 'rate limit'),
            (ccxt.ExchangeNotAvailable('GET x 451 restricted'), 'location'),
            (ccxt.PermissionDenied('GET x 403'), 'location'),
            (ccxt.RequestTimeout('GET x timed out'), 'timed out'),
            (ccxt.NetworkError('reset'), 'network error'),
        ]
        for error, expected in cases:
            self.assertIn(expected, bf.describe_exchange_failure(error))


class ScanTests(unittest.TestCase):
    def run_scan(self, fake):
        app = load_app()
        with patch.object(app.ccxt, 'binanceusdm', return_value=fake), \
                patch.object(bf.time, 'sleep'), \
                patch.object(app.st, 'warning') as warning, patch.object(app.st, 'progress'):
            result = app.scan_top_derivative_assets(top_n=1, binance_usdm_only=True, include_market_caps=False)
        return result, warning

    def test_one_failed_ticker_request_no_longer_stops_the_scan(self):
        fake = FakeUsdm([ccxt.RequestTimeout('slow')])
        _, warning = self.run_scan(fake)
        self.assertEqual(fake.ticker_calls, 2)
        warning.assert_not_called()

    def test_persistent_failure_reports_the_real_cause(self):
        fake = FakeUsdm([ccxt.RequestTimeout('slow')] * 3)
        result, warning = self.run_scan(fake)
        self.assertTrue(result.empty)
        message = warning.call_args[0][0]
        self.assertIn('timed out', message)
        self.assertIn('binanceusdm', message)
        self.assertNotIn('blocked', message)


if __name__ == '__main__':
    unittest.main()
