import os
import tempfile
import unittest

import pandas as pd

import watchlist_history as wh


def matches(*specs):
    """specs: (symbol, verdict, entry). Every row is a valid watch list match."""
    return pd.DataFrame([{
        'Volume Rank': rank, 'Symbol': symbol, 'Verdict': verdict, 'Regime': 'Mean-Reverting',
        'Quant Score Q': 0.4 if verdict == 'LONG' else -0.4, '$ Risk': 10.0, 'Leverage': 0.25,
        'Entry': entry, 'Stop': entry * 0.98, 'Target': entry * 1.04, 'Size (units)': 1.0, 'Notional': entry,
        'BT Expectancy R': 0.2, 'BT Win Rate': 0.5,
    } for rank, (symbol, verdict, entry) in enumerate(specs, start=1)])


class WatchlistHistoryTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, 'history.csv')

    def tearDown(self):
        self.dir.cleanup()

    def record(self, frame, when, leverage=0.4, timeframe='1h'):
        return wh.record_matches(frame, pd.Timestamp(when, tz='UTC'), timeframe, 10.0, leverage, path=self.path)

    def test_empty_history_before_anything_is_saved(self):
        history = wh.load_history(self.path)
        self.assertTrue(history.empty)
        self.assertEqual(wh.available_dates(history), [])
        self.assertTrue(wh.day_summary(history, '2026-11-11').empty)

    def test_records_rows_and_skips_the_same_scan_twice(self):
        frame = matches(('AAA/USDT', 'LONG', 1.0), ('BBB/USDT', 'SHORT', 2.0))
        self.assertEqual(self.record(frame, '2026-11-11 09:00'), 2)
        self.assertEqual(self.record(frame, '2026-11-11 09:00'), 0)  # page re-render, same scan
        self.assertEqual(len(wh.load_history(self.path)), 2)

    def test_a_new_scan_or_new_leverage_limit_is_a_new_record(self):
        frame = matches(('AAA/USDT', 'LONG', 1.0))
        self.record(frame, '2026-11-11 09:00')
        self.assertEqual(self.record(frame, '2026-11-11 10:00'), 1)
        self.assertEqual(self.record(frame, '2026-11-11 10:00', leverage=0.3), 1)
        self.assertEqual(len(wh.load_history(self.path)), 3)

    def test_empty_matches_write_nothing(self):
        self.assertEqual(self.record(matches(), '2026-11-11 09:00'), 0)
        self.assertFalse(os.path.exists(self.path))

    def test_days_are_ghana_dates_and_listed_newest_first(self):
        self.record(matches(('AAA/USDT', 'LONG', 1.0)), '2026-11-10 23:30')
        self.record(matches(('BBB/USDT', 'SHORT', 2.0)), '2026-11-11 00:15')
        history = wh.load_history(self.path)
        self.assertEqual(wh.available_dates(history), ['2026-11-11', '2026-11-10'])
        self.assertEqual(list(wh.rows_for_day(history, '2026-11-10')['Symbol']), ['AAA/USDT'])
        self.assertEqual(list(wh.rows_for_day(history, pd.Timestamp('2026-11-11').date())['Symbol']), ['BBB/USDT'])

    def test_day_summary_has_one_row_per_asset_with_latest_plan(self):
        self.record(matches(('AAA/USDT', 'LONG', 1.0), ('BBB/USDT', 'SHORT', 2.0)), '2026-11-11 08:00')
        self.record(matches(('AAA/USDT', 'SHORT', 1.5)), '2026-11-11 13:00')
        self.record(matches(('CCC/USDT', 'LONG', 3.0)), '2026-11-12 08:00')  # other day
        summary = wh.day_summary(wh.load_history(self.path), '2026-11-11').set_index('Symbol')
        self.assertEqual(sorted(summary.index), ['AAA/USDT', 'BBB/USDT'])
        aaa = summary.loc['AAA/USDT']
        self.assertEqual((aaa['First Seen'], aaa['Last Seen'], aaa['Times Seen']), ('08:00', '13:00', 2))
        self.assertEqual(aaa['Verdict'], 'SHORT')            # latest scan
        self.assertEqual(aaa['Verdicts'], 'LONG / SHORT')    # everything seen that day
        self.assertAlmostEqual(aaa['Entry'], 1.5)
        self.assertEqual(aaa['Leverage Limits'], '<0.40x')

    def test_rank_column_renamed_by_the_panel_is_kept(self):
        frame = matches(('AAA/USDT', 'LONG', 1.0))
        self.record(frame, '2026-11-11 09:00')
        self.assertEqual(int(wh.load_history(self.path)['Volume Rank'].iloc[0]), 1)


if __name__ == '__main__':
    unittest.main()
