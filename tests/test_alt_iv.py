import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import alt_iv
import implied_volatility_panel as ivp

NOW = pd.Timestamp('2026-11-19 10:00', tz='UTC')


class FakeResponse:
    def __init__(self, rows):
        self.rows = rows

    def raise_for_status(self):
        pass

    def json(self):
        return {'result': self.rows}


def book(rows):
    """rows: (name, mark_iv %, underlying_price)."""
    return [{'instrument_name': n, 'mark_iv': iv, 'underlying_price': f, 'open_interest': 1.0} for n, iv, f in rows]


SOL_BOOK = book([
    ('SOL_USDC-20NOV26-75-C', 90.0, 75.0), ('SOL_USDC-20NOV26-75-P', 92.0, 75.0),     # < 2 days: skipped
    ('SOL_USDC-27NOV26-75-C', 64.0, 75.1), ('SOL_USDC-27NOV26-75-P', 66.0, 75.1),
    ('SOL_USDC-27NOV26-80-C', 70.0, 75.1),
    ('SOL_USDC-25DEC26-76-C', 61.0, 75.9), ('SOL_USDC-25DEC26-76-P', 63.0, 75.9),
    ('SOL_USDC-25DEC26-74-C', 99.0, 75.9),
    ('SOLX_USDC-27NOV26-1-C', 10.0, 1.0),                                              # other coin
    ('XRP_USDC-27NOV26-0d98-C', 71.0, 0.98),
])


class ChainTests(unittest.TestCase):
    def chain(self, coin='SOL'):
        return alt_iv.option_chain(coin, get=lambda *a, **k: FakeResponse(SOL_BOOK))

    def test_parses_only_that_coins_options(self):
        chain = self.chain()
        self.assertTrue(chain['instrument_name'].str.startswith('SOL_USDC-').all())
        self.assertEqual(len(chain), 8)
        self.assertEqual(chain['expiry'].min(), pd.Timestamp('2026-11-20 08:00', tz='UTC'))
        self.assertAlmostEqual(self.chain('XRP')['strike'].iloc[0], 0.98)       # '0d98' strikes

    def test_atm_term_skips_near_expiries_and_uses_each_forward(self):
        term = alt_iv.atm_term(self.chain(), NOW)
        self.assertEqual(len(term), 2)
        self.assertEqual(list(term['strike']), [75.0, 76.0])
        np.testing.assert_allclose(term['iv'], [0.65, 0.62])
        self.assertAlmostEqual(term['days'].iloc[0], 7 + 22 / 24)

    def test_30_day_iv_interpolates_total_variance(self):
        term = pd.DataFrame({'days': [8.0, 36.0], 'iv': [0.70, 0.60]})
        iv, near, far = alt_iv.iv_at_days(term)
        w = 0.70 ** 2 * 8 + (0.60 ** 2 * 36 - 0.70 ** 2 * 8) * (30 - 8) / 28
        self.assertAlmostEqual(iv, np.sqrt(w / 30))
        self.assertEqual((near, far), (8.0, 36.0))
        self.assertEqual(alt_iv.iv_at_days(pd.DataFrame({'days': [40.0, 90.0], 'iv': [0.5, 0.4]})), (0.5, 40.0, 40.0))
        self.assertIsNone(alt_iv.iv_at_days(pd.DataFrame()))

    def test_live_iv(self):
        reading = alt_iv.live_iv('SOL', get=lambda *a, **k: FakeResponse(SOL_BOOK), now=NOW)
        self.assertTrue(0.6 < reading['iv'] < 0.65)
        self.assertEqual(reading['time'], NOW)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, 'snaps.csv')

    def tearDown(self):
        self.dir.cleanup()

    def save(self, coin, when, iv):
        return alt_iv.record_snapshot(coin, {'iv': iv, 'near_days': 8, 'far_days': 36,
                                             'time': pd.Timestamp(when, tz='UTC')}, self.path)

    def test_saves_and_skips_readings_too_close_together(self):
        self.assertTrue(self.save('SOL', '2026-11-19 10:00', 0.6))
        self.assertFalse(self.save('SOL', '2026-11-19 10:05', 0.61))
        self.assertTrue(self.save('XRP', '2026-11-19 10:05', 0.7))
        self.assertTrue(self.save('SOL', '2026-11-19 11:00', 0.62))
        snaps = alt_iv.load_snapshots(self.path)
        self.assertEqual(list(snaps['Coin']), ['SOL', 'XRP', 'SOL'])
        self.assertFalse(alt_iv.record_snapshot('SOL', None, self.path))

    def test_series_has_midnight_opens_and_late_closes_only(self):
        for when, iv in [('2026-11-17 23:30', 0.50), ('2026-11-18 00:40', 0.52), ('2026-11-18 12:00', 0.55),
                         ('2026-11-18 23:10', 0.58), ('2026-11-19 03:00', 0.60)]:
            self.save('SOL', when, iv)
        series = alt_iv.snapshot_series(alt_iv.load_snapshots(self.path), 'SOL')
        day17, day18, day19 = (pd.Timestamp(d, tz='UTC') for d in ('2026-11-17', '2026-11-18', '2026-11-19'))
        self.assertEqual(series.to_dict(), {day17: 0.50, day18: 0.58})         # 11-19 has no reading after 22:00
        self.assertEqual(series.attrs['open'][day18], 0.50)                   # 23:30 is 30 min from midnight
        self.assertEqual(series.attrs['open'][day19], 0.58)                   # 23:10 the night before
        self.assertNotIn(day17, series.attrs['open'])
        self.assertEqual(ivp.dvol_at(series, day18), 0.50)
        live = alt_iv.with_live(series, {'iv': 0.66, 'time': NOW})
        self.assertEqual(live.iloc[-1], 0.66)
        self.assertEqual(live.attrs['open'], series.attrs['open'])

    def test_empty(self):
        series = alt_iv.snapshot_series(alt_iv.load_snapshots(self.path), 'SOL')
        self.assertTrue(series.empty)
        self.assertEqual(series.attrs['open'], {})


class PanelTests(unittest.TestCase):
    def closes(self):
        index = pd.date_range('2026-09-01', periods=80, freq='D', tz='UTC')
        rng = np.random.default_rng(0)
        return pd.Series(0.98 * np.exp(np.cumsum(rng.normal(0, 0.03, 80))), index=index)

    def test_without_a_midnight_reading_the_range_uses_the_open_and_live_iv(self):
        closes = self.closes()
        day = (closes.index[-1], 0.97)
        s = ivp.alt_summary('XRP', {'iv': 0.70, 'near_days': 8, 'far_days': 36, 'time': closes.index[-1]},
                            pd.DataFrame(columns=alt_iv.SNAPSHOT_COLUMNS), closes, day)
        self.assertTrue(s['day_anchored'] and s['day_iv_live'])
        self.assertAlmostEqual(sum(s['daily_range']) / 2, 0.97)
        self.assertAlmostEqual(s['daily_move_pct'], 0.70 / np.sqrt(365))
        self.assertEqual(s['history_days'], 1)

    def test_names_and_prices(self):
        self.assertEqual(ivp.perpetual_name('BTC'), 'BTC-PERPETUAL')
        self.assertEqual(ivp.perpetual_name('SOL'), 'SOL_USDC-PERPETUAL')
        self.assertEqual(ivp.IV_COINS, ('BTC', 'ETH', 'SOL', 'XRP', 'AVAX', 'HYPE', 'TRX'))
        self.assertEqual([ivp.price_text(v) for v in (64890.4, 74.803, 7.06, 1.0004, 0.92997, 0.31276, 0.059435, np.nan)],
                         ['64,890', '74.80', '7.060', '1.000', '0.9300', '0.3128', '0.05944', '—'])
        self.assertEqual(ivp.iv_label('ETH'), 'DVOL')
        self.assertEqual(ivp.iv_label('HYPE'), '30-day ATM IV')


class RunnerTests(unittest.TestCase):
    def test_runner_saves_each_coin(self):
        import alert_runner
        readings = {'SOL': {'iv': 0.63}, 'XRP': None}
        saved = []
        with mock.patch.object(alt_iv, 'live_iv', side_effect=lambda coin: readings[coin]), \
                mock.patch.object(alt_iv, 'record_snapshot', side_effect=lambda c, r: saved.append(c) or bool(r)), \
                self.assertLogs(alert_runner.log, 'INFO') as logs:
            alert_runner.save_implied_vol({'coins': ['SOL', 'XRP']})
        self.assertEqual(saved, ['SOL', 'XRP'])
        self.assertIn('SOL 63.0%', logs.output[-1])
        self.assertIn('XRP no data', logs.output[-1])
        self.assertIn('implied_vol', alert_runner.DEFAULT_CONFIG['groups'])


if __name__ == '__main__':
    unittest.main()
