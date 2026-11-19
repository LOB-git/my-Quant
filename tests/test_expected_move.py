import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from scipy.stats import norm

import expected_move as em

MODULE_PATH = Path(__file__).resolve().parents[1] / 'dashboard.py'


def chain(forward, move, strikes, spread=0.0, quoted=True):
    """Black-Scholes (zero rates) calls and puts for a return standard deviation `move` to expiry."""
    rows_c, rows_p = [], []
    for strike in strikes:
        d1 = (np.log(forward / strike) + move ** 2 / 2) / move
        call = forward * norm.cdf(d1) - strike * norm.cdf(d1 - move)
        put = call - (forward - strike)
        for rows, price in ((rows_c, call), (rows_p, put)):
            rows.append({'strike': strike, 'bid': price - spread if quoted else 0.0,
                         'ask': price + spread if quoted else 0.0, 'lastPrice': price,
                         'impliedVolatility': 0.01})  # deliberately wrong: must not be used
    return pd.DataFrame(rows_c), pd.DataFrame(rows_p)


class TradingDaysTests(unittest.TestCase):
    def test_counts_what_is_left_of_today_plus_later_sessions(self):
        wednesday_noon = pd.Timestamp('2026-11-18 12:45', tz=em.NY)   # half of the session left
        self.assertAlmostEqual(em.trading_days_to_expiry(wednesday_noon, '2026-11-19'), 1.5)
        self.assertAlmostEqual(em.trading_days_to_expiry(wednesday_noon, '2026-11-23'), 3.5)  # weekend skipped
        before_open = pd.Timestamp('2026-11-18 08:00', tz=em.NY)
        self.assertAlmostEqual(em.trading_days_to_expiry(before_open, '2026-11-19'), 2.0)
        after_close = pd.Timestamp('2026-11-18 17:00', tz=em.NY)
        self.assertAlmostEqual(em.trading_days_to_expiry(after_close, '2026-11-19'), 1.0)
        saturday = pd.Timestamp('2026-11-21 10:00', tz=em.NY)
        self.assertAlmostEqual(em.trading_days_to_expiry(saturday, '2026-11-23'), 1.0)
        self.assertAlmostEqual(em.trading_days_to_expiry(pd.Timestamp('2026-11-18 21:00', tz='UTC'), '2026-11-19'), 1.0)


class StraddleTests(unittest.TestCase):
    def test_recovers_the_move_and_forward_even_when_spot_is_stale(self):
        calls, puts = chain(728.1, 0.017, np.arange(715, 746, 1.0), spread=0.01)
        result = em.straddle_move(calls, puts, spot=729.95)   # Yahoo's delayed spot
        self.assertAlmostEqual(result['forward'], 728.1, places=6)
        self.assertEqual(result['strike'], 728.0)
        self.assertAlmostEqual(result['move'], 0.017, places=4)
        self.assertEqual(result['quote'], 'bid/ask mid')

    def test_falls_back_to_last_trade_without_quotes(self):
        calls, puts = chain(500, 0.02, np.arange(490, 511, 1.0), quoted=False)
        result = em.straddle_move(calls, puts, spot=500)
        self.assertEqual(result['quote'], 'last trade')
        self.assertAlmostEqual(result['move'], 0.02, places=4)

    def test_no_chain_gives_none(self):
        self.assertIsNone(em.straddle_move(pd.DataFrame(), pd.DataFrame(), 100))


class OptionsDailyMoveTests(unittest.TestCase):
    now = pd.Timestamp('2026-11-18 12:45', tz=em.NY)

    def test_skips_today_and_scales_by_trading_days(self):
        loaded = []

        def load(expiry):
            loaded.append(expiry)
            return chain(100, 0.015 * np.sqrt(1.5), np.arange(95, 106, 1.0))

        result = em.options_daily_move(['2026-11-18', '2026-11-19', '2026-11-20'], load, 100, self.now)
        self.assertEqual(loaded, ['2026-11-19'])
        self.assertEqual(result['expiry'], '2026-11-19')
        self.assertAlmostEqual(result['trading_days'], 1.5)
        self.assertAlmostEqual(result['daily_move'], 0.015, places=4)

    def test_moves_to_the_next_expiry_when_a_chain_fails(self):
        def load(expiry):
            if expiry == '2026-11-19':
                raise RuntimeError('rate limited')
            return chain(100, 0.015 * np.sqrt(2.5), np.arange(95, 106, 1.0))

        result = em.options_daily_move(['2026-11-19', '2026-11-20'], load, 100, self.now)
        self.assertEqual(result['expiry'], '2026-11-20')
        self.assertAlmostEqual(result['daily_move'], 0.015, places=4)

    def test_only_same_day_expiries_gives_none(self):
        self.assertIsNone(em.options_daily_move(['2026-11-18'], lambda e: chain(100, 0.01, [100.0]), 100, self.now))


class RangeHitTests(unittest.TestCase):
    def prices(self, true_sigma, days=4000, seed=7):
        rng = np.random.default_rng(seed)
        close = 100 * np.exp(np.cumsum(rng.normal(0, true_sigma, days)))
        return pd.Series(close, index=pd.bdate_range('2010-01-01', periods=days))

    def test_a_correct_range_holds_about_68_percent(self):
        close = self.prices(0.01)
        hits = em.range_hits(close, pd.Series(0.01, index=close.index))
        self.assertAlmostEqual(hits['Close inside'], em.ONE_SIGMA, delta=0.02)
        self.assertAlmostEqual(hits['Beyond 2x range'], em.TWO_SIGMA_OUT, delta=0.01)
        self.assertAlmostEqual(hits['Actual ÷ predicted'], 1.0, delta=0.03)
        self.assertEqual(hits['Days'], len(close) - 1)

    def test_too_wide_and_too_narrow_ranges_are_called_out(self):
        close = self.prices(0.01)
        wide = em.range_hits(close, pd.Series(0.015, index=close.index))
        narrow = em.range_hits(close, pd.Series(0.007, index=close.index))
        self.assertGreater(wide['Close inside'], 0.8)
        self.assertLess(narrow['Close inside'], 0.55)
        self.assertIn('too wide', em.verdict(wide['Close inside']))
        self.assertIn('too narrow', em.verdict(narrow['Close inside']))
        self.assertIn('well sized', em.verdict(0.69))

    def test_range_uses_the_previous_days_move_and_close(self):
        close = pd.Series([100.0, 101.0, 103.0] + [103.0] * 20, index=pd.bdate_range('2026-01-01', periods=23))
        sigma = pd.Series([0.015, 0.01] + [0.01] * 21, index=close.index)
        frame = em.range_hits(close, sigma)
        # Day 2: 100 -> 101 is +1% against 1.5% (inside); day 3: 101 -> 103 is +2% against 1% (above).
        self.assertEqual(frame['Days'], 22)
        self.assertAlmostEqual(frame['Closed above'], 1 / 22)

    def test_high_low_check_and_accuracy_table(self):
        close = self.prices(0.01, days=300)
        prices = pd.DataFrame({'close': close, 'high': close * 1.001, 'low': close * 0.999})
        vix = pd.Series(0.01 * np.sqrt(252) * 100, index=close.index)
        table = em.range_accuracy(prices, vix, '^VIX')
        self.assertEqual(list(table['Method']), ['Historical (20-day)', 'Implied (^VIX)'])
        self.assertTrue((table['High & low inside'] <= table['Close inside']).all())
        self.assertAlmostEqual(table.iloc[1]['Avg range'], 0.01)

    def test_daily_frame_drops_time_zones(self):
        index = pd.date_range('2026-01-05', periods=3, freq='D', tz='America/New_York')
        frame = em.daily_frame(pd.DataFrame({'Close': [1.0, 2, 3], 'High': [1.0, 2, 3], 'Low': [1.0, 2, 3]}, index))
        self.assertIsNone(frame.index.tz)
        self.assertEqual(list(frame.columns), ['close', 'high', 'low'])


class DashboardIntegrationTests(unittest.TestCase):
    def test_dashboard_reads_the_straddle_not_yahoos_iv_column(self):
        spec = importlib.util.spec_from_file_location('quant_app_em', MODULE_PATH)
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        tomorrow = (pd.Timestamp.now(tz=em.NY) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        calls, puts = chain(600, 0.02, np.arange(580, 621, 1.0))
        ticker = mock.Mock(options=(tomorrow,))
        ticker.option_chain.return_value = mock.Mock(calls=calls, puts=puts)
        with mock.patch.object(app.yf, 'Ticker', return_value=ticker):
            result = app.get_implied_volatility_analysis.__wrapped__('SPY', 600.0, None)
        days = em.trading_days_to_expiry(pd.Timestamp.now(tz=em.NY), tomorrow)
        self.assertTrue(result['is_implied'])
        self.assertEqual(result['expiration'], tomorrow)
        self.assertAlmostEqual(result['daily_move_pct'], 0.02 / np.sqrt(days), places=4)
        self.assertIn('straddle', result['source'])


if __name__ == '__main__':
    unittest.main()
