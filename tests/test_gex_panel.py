import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import alert_runner
import gex_panel as gp

NOW = pd.Timestamp('2026-11-13 12:00', tz='UTC')


def book(rows):
    """rows: (instrument, oi, mark_iv %, underlying)"""
    return pd.DataFrame([{'instrument_name': name, 'open_interest': oi, 'mark_iv': iv, 'underlying_price': spot}
                         for name, oi, iv, spot in rows])


SAMPLE = book([
    ('BTC-20NOV26-70000-C', 100, 40, 67000),
    ('BTC-20NOV26-64000-P', 300, 45, 67000),
    ('BTC-27DEC26-80000-C', 500, 42, 67500),
    ('BTC-27DEC26-60000-P', 50, 50, 67500),
    ('BTC-27DEC26-90000-C', 0, 45, 67500),     # no open interest: dropped
    ('BTC-13NOV26-67000-C', 10, 40, 67000),    # expired at 08:00 today: dropped
])


class ParseTests(unittest.TestCase):
    def test_parses_names_drops_dead_options_and_uses_0800_utc_expiry(self):
        options = gp.parse_options(SAMPLE, NOW)
        self.assertEqual(len(options), 4)
        first = options.iloc[0]
        self.assertEqual((first['strike'], first['type']), (70000, 'C'))
        self.assertEqual(first['expiry'], pd.Timestamp('2026-11-20 08:00', tz='UTC'))
        self.assertAlmostEqual(first['T'], (7 * 86400 - 4 * 3600) / gp.YEAR_SECONDS)
        self.assertAlmostEqual(first['iv'], 0.40)

    def test_empty_book(self):
        self.assertTrue(gp.parse_options(pd.DataFrame(), NOW).empty)
        self.assertIsNone(gp.gex_snapshot(gp.parse_options(pd.DataFrame(), NOW), 'BTC'))


class GexMathTests(unittest.TestCase):
    def test_matches_black_scholes_gamma_and_signs(self):
        options = gp.parse_options(SAMPLE, NOW)
        gex = gp.option_gex(options, 67000.0)
        row = options.iloc[0]
        d1 = (np.log(67000 / row['strike']) + 0.5 * row['iv'] ** 2 * row['T']) / (row['iv'] * np.sqrt(row['T']))
        gamma = np.exp(-d1 ** 2 / 2) / np.sqrt(2 * np.pi) / (67000 * row['iv'] * np.sqrt(row['T']))
        self.assertAlmostEqual(gex[0], gamma * 100 * 67000 ** 2 * 0.01)
        self.assertTrue((gex[options['type'] == 'C'] > 0).all())
        self.assertTrue((gex[options['type'] == 'P'] < 0).all())

    def test_flip_level_is_where_net_gex_changes_sign(self):
        options = gp.parse_options(SAMPLE, NOW)
        flip = gp.flip_level(options, 67000.0)
        self.assertIsNotNone(flip)
        below, above = gp.option_gex(options, flip * 0.99).sum(), gp.option_gex(options, flip * 1.01).sum()
        self.assertLess(below * above, 0)

    def test_only_calls_has_no_flip(self):
        calls = gp.parse_options(book([('BTC-27DEC26-80000-C', 100, 40, 67000)]), NOW)
        self.assertIsNone(gp.flip_level(calls, 67000.0))

    def test_snapshot_totals_and_walls(self):
        snap = gp.gex_snapshot(gp.parse_options(SAMPLE, NOW), 'BTC', NOW)
        self.assertAlmostEqual(snap['net_gex'], snap['call_gex'] + snap['put_gex'])
        self.assertEqual(snap['regime'], 'POSITIVE' if snap['net_gex'] > 0 else 'NEGATIVE')
        self.assertIn(snap['call_wall'], (70000.0, 80000.0))
        self.assertIn(snap['put_wall'], (64000.0, 60000.0))
        self.assertEqual(snap['next_expiry'], pd.Timestamp('2026-11-20 08:00', tz='UTC'))
        self.assertEqual(snap['total_oi'], 950)
        gex = snap['options']['gex'].abs()
        near = snap['options']['expiry'] == snap['next_expiry']  # 6 days 20 h away: inside 7 days
        self.assertAlmostEqual(snap['share_7d'], gex[near].sum() / gex.sum())

    def test_subset_can_use_the_full_book_spot(self):
        options = gp.parse_options(SAMPLE, NOW)
        nearest = options[options['expiry'] == options['expiry'].min()]
        self.assertEqual(gp.gex_snapshot(nearest, 'BTC', NOW)['spot'], 67000.0)
        self.assertEqual(gp.gex_snapshot(nearest, 'BTC', NOW, spot=67250.0)['spot'], 67250.0)

    def test_chart_has_call_and_put_bars(self):
        snap = gp.gex_snapshot(gp.parse_options(SAMPLE, NOW), 'BTC', NOW)
        fig = gp.gex_chart(snap, 'BTC')
        self.assertEqual([t.name for t in fig.data], ['Calls (+)', 'Puts (−)'])


class LogTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, 'gex.csv')
        self.snap = gp.gex_snapshot(gp.parse_options(SAMPLE, NOW), 'BTC', NOW)

    def tearDown(self):
        self.dir.cleanup()

    def test_one_row_per_currency_per_hour(self):
        self.assertTrue(gp.record_snapshot(self.snap, 'runner', self.path))
        later_same_hour = dict(self.snap, captured_at=NOW + pd.Timedelta(minutes=40))
        self.assertFalse(gp.record_snapshot(later_same_hour, 'dashboard', self.path))
        self.assertTrue(gp.record_snapshot(dict(self.snap, currency='ETH'), 'runner', self.path))
        self.assertTrue(gp.record_snapshot(dict(self.snap, captured_at=NOW + pd.Timedelta(hours=1)), 'runner', self.path))
        history = gp.load_gex_history(self.path)
        self.assertEqual(list(history.columns), gp.HISTORY_COLUMNS)
        self.assertEqual(len(history), 3)
        self.assertEqual(history['Snapshot Hour (UTC)'].iloc[0], '2026-11-13 12:00')

    def test_log_reports_failures_per_currency(self):
        def fetch(currency, get=None):
            if currency == 'ETH':
                raise RuntimeError('down')
            return SAMPLE
        with mock.patch.object(gp, 'fetch_option_book', fetch):
            results = gp.log_gex_snapshots(('BTC', 'ETH'), path=self.path)
        self.assertEqual(results['BTC'], 'saved')
        self.assertTrue(results['ETH'].startswith('failed'))


class RunnerTests(unittest.TestCase):
    def test_gex_group_runs_hourly_without_telegram(self):
        config = {'interval_seconds': 300, 'groups': {'gex': {'enabled': True, 'every_seconds': 3600,
                                                              'currencies': ['BTC']}}}
        calls, last_run = [], {}
        for seconds in (0, 300, 3600):
            with mock.patch.object(alert_runner, 'save_gex', lambda group, dry: calls.append(seconds)), \
                    mock.patch.object(alert_runner, 'save_state', lambda state: None), \
                    mock.patch.object(alert_runner.time, 'monotonic', lambda: seconds):
                alert_runner.run_cycle(config, None, None, '', '', {}, last_run)
        self.assertEqual(calls, [0, 3600])

    def test_default_config_has_gex(self):
        self.assertEqual(alert_runner.DEFAULT_CONFIG['groups']['gex']['currencies'], ['BTC', 'ETH'])


if __name__ == '__main__':
    unittest.main()
