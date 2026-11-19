import unittest

import numpy as np
import pandas as pd

import implied_volatility_panel as ivp

DAY = ivp.DAY_MS
START = 1_700_000_000_000 // DAY * DAY  # a UTC midnight


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def fake_get(pages):
    """Serves get_volatility_index_data pages in order; records the params it was asked for."""
    calls = []

    def get(url, params, timeout):
        calls.append((url.rsplit('/', 1)[-1], dict(params)))
        return FakeResponse({'result': pages[len(calls) - 1]})
    return get, calls


class CurrencyTests(unittest.TestCase):
    def test_maps_app_symbols_to_deribit_currencies(self):
        for symbol, expected in [('BTC/USD', 'BTC'), ('BTC-USD', 'BTC'), ('BTCUSDT', 'BTC'), ('eth/usdt:usdt', 'ETH'),
                                 ('ETH', 'ETH'), ('SOL/USDT', None), ('BTCDOM/USDT', None), ('SPY', None), (None, None)]:
            with self.subTest(symbol=symbol):
                self.assertEqual(ivp.deribit_currency(symbol), expected)


class FetchTests(unittest.TestCase):
    def test_dvol_follows_continuation_dedupes_and_converts_to_fractions(self):
        newer = {'data': [[START + DAY, 1, 1, 1, 45.0], [START + 2 * DAY, 1, 1, 1, 50.0]], 'continuation': START + DAY}
        older = {'data': [[START, 1, 1, 1, 40.0], [START + DAY, 1, 1, 1, 45.0]], 'continuation': None}
        get, calls = fake_get([newer, older])
        series = ivp.fetch_dvol('BTC', days=10, get=get, now_ms=START + 3 * DAY)
        self.assertEqual(list(series), [0.40, 0.45, 0.50])
        self.assertTrue(series.index.is_monotonic_increasing)
        self.assertEqual(calls[0][1]['currency'], 'BTC')
        self.assertEqual(calls[1][1]['end_timestamp'], START + DAY)  # second page ends where the first said

    def test_daily_closes_use_the_perpetual(self):
        get, calls = fake_get([{'ticks': [START, START + DAY], 'close': [100.0, 101.0], 'status': 'ok'}])
        closes = ivp.fetch_daily_closes('ETH', get=get, now_ms=START + 2 * DAY)
        self.assertEqual(list(closes), [100.0, 101.0])
        self.assertEqual(calls[0][1]['instrument_name'], 'ETH-PERPETUAL')

    def test_day_open_is_the_midnight_hourly_candle(self):
        now = START + 5 * 3_600_000 + 123
        get, calls = fake_get([{'ticks': [START, START + 3_600_000], 'open': [64134.5, 63393.5]}])
        day, price = ivp.fetch_day_open('BTC', get=get, now_ms=now)
        self.assertEqual(day, pd.Timestamp(START, unit='ms', tz='UTC'))
        self.assertEqual(price, 64134.5)
        self.assertEqual(calls[0][1]['resolution'], '60')
        self.assertEqual(calls[0][1]['start_timestamp'], START)
        get, _ = fake_get([{'ticks': [START + 3_600_000], 'open': [1.0]}])
        self.assertIsNone(ivp.fetch_day_open('BTC', get=get, now_ms=now))

    def test_dvol_keeps_midnight_open(self):
        get, _ = fake_get([{'data': [[START, 47.28, 54.6, 43.3, 44.21]], 'continuation': None}])
        series = ivp.fetch_dvol('BTC', days=2, get=get, now_ms=START + DAY)
        self.assertAlmostEqual(ivp.dvol_at(series, pd.Timestamp(START, unit='ms', tz='UTC')), 0.4728)

    def test_deribit_error_raises(self):
        get = lambda *a, **k: FakeResponse({'error': {'message': 'bad currency'}})
        with self.assertRaisesRegex(RuntimeError, 'bad currency'):
            ivp.fetch_dvol('XRP', get=get)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range('2025-01-01', periods=400, freq='D', tz='UTC')
        rng = np.random.default_rng(0)
        self.closes = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))), index=index)
        self.dvol = pd.Series(np.linspace(0.30, 0.70, 400), index=index)

    def test_moves_use_365_days_and_spread_is_iv_minus_rv(self):
        s = ivp.iv_summary(self.dvol, self.closes)
        rv = np.log(self.closes).diff().tail(20).std() * np.sqrt(365)
        self.assertAlmostEqual(s['iv'], 0.70)
        self.assertAlmostEqual(s['rv'], rv)
        self.assertAlmostEqual(s['spread'], 0.70 - rv)
        self.assertEqual(s['rv_outlook'], 'INCREASE' if 0.70 > rv else 'DECREASE')
        self.assertAlmostEqual(s['daily_move_pct'], 0.70 / np.sqrt(365))
        self.assertAlmostEqual(s['weekly_move_pct'], 0.70 * np.sqrt(7 / 365))
        low, high = s['daily_range']
        self.assertAlmostEqual((high - low) / 2, s['price'] * s['daily_move_pct'])

    def test_daily_range_is_anchored_to_the_utc_open_and_dvol_at_midnight(self):
        today = self.closes.index[-1]
        day = (today, 101.0)
        s = ivp.iv_summary(self.dvol, self.closes, day)
        day_iv = self.dvol.iloc[-2]                      # no candle opens -> last DVOL close before midnight
        self.assertTrue(s['day_anchored'])
        self.assertEqual(s['day_start'], today)
        self.assertAlmostEqual(s['day_open'], 101.0)
        self.assertAlmostEqual(s['day_iv'], day_iv)
        self.assertAlmostEqual(s['daily_move_pct'], day_iv / np.sqrt(365))
        low, high = s['daily_range']
        self.assertAlmostEqual((low + high) / 2, 101.0)
        self.assertAlmostEqual(s['day_sigma'], (s['price'] / 101.0 - 1) / s['daily_move_pct'])
        # fixed for the day: a different live price / live DVOL today does not move the range
        moved = self.closes.copy()
        moved.iloc[-1] = 101.0 * 1.2
        live = self.dvol.copy()
        live.iloc[-1] = 0.99
        s2 = ivp.iv_summary(live, moved, day)
        self.assertEqual(s2['daily_range'], s['daily_range'])
        self.assertEqual(s2['day_position'], 'ABOVE range')
        self.assertAlmostEqual(s2['iv'], 0.99)             # headline DVOL is still live
        # the midnight DVOL candle open wins when available
        live.attrs['open'] = {today: 0.55}
        self.assertAlmostEqual(ivp.iv_summary(live, moved, day)['day_iv'], 0.55)

    def test_without_opens_range_follows_live_price(self):
        s = ivp.iv_summary(self.dvol, self.closes)
        self.assertFalse(s['day_anchored'])
        self.assertAlmostEqual(sum(s['daily_range']) / 2, s['price'])

    def test_median_and_percentile_cover_the_last_year_only(self):
        s = ivp.iv_summary(self.dvol, self.closes)
        self.assertAlmostEqual(s['iv_median'], self.dvol.tail(365).median())
        self.assertEqual(s['iv_percentile'], 100.0)  # rising series: today is the highest

    def test_iv_below_rv_says_decrease(self):
        s = ivp.iv_summary(self.dvol * 0 + 0.01, self.closes)
        self.assertEqual(s['rv_outlook'], 'DECREASE')

    def test_too_little_data_returns_none(self):
        self.assertIsNone(ivp.iv_summary(pd.Series(dtype=float), self.closes))
        self.assertIsNone(ivp.iv_summary(self.dvol, self.closes.head(10)))

    def test_chart_has_both_lines(self):
        fig = ivp.iv_chart(ivp.iv_summary(self.dvol, self.closes), 'BTC')
        self.assertEqual([t.name for t in fig.data], ['Implied (DVOL)', 'Realised 20d'])


if __name__ == '__main__':
    unittest.main()
