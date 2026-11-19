import unittest

import numpy as np
import pandas as pd

import quant_edge as qe
import us_quant_scan_panel as us
from tests.test_quant_edge import SETTINGS, ohlc_from_close


class UsQuantScanTests(unittest.TestCase):
    def test_futures_sizes_convert_to_contracts_and_indices_are_flagged(self):
        table = pd.DataFrame({'Symbol': ['NQ=F', 'SPY', '^VIX'], 'Size (units)': [40.0, 10.0, 5.0],
                              'Reason': ['ok', 'ok', 'ok']})
        result = us.add_us_contract_columns(table).set_index('Symbol')
        self.assertEqual(result.loc['NQ=F', 'E-mini Contracts'], 2.0)
        self.assertEqual(result.loc['NQ=F', 'Micro Contracts'], 20.0)
        self.assertTrue(np.isnan(result.loc['SPY', 'E-mini Contracts']))
        self.assertIn('UVXY', result.loc['^VIX', 'Reason'])

    def test_vix_conflict_only_when_vix_agrees_with_equities(self):
        same = pd.DataFrame({'Symbol': ['SPY', '^VIX'], 'Verdict': ['LONG', 'LONG']})
        opposite = pd.DataFrame({'Symbol': ['SPY', '^VIX'], 'Verdict': ['LONG', 'SHORT']})
        self.assertTrue(us.vix_conflict(same))
        self.assertFalse(us.vix_conflict(opposite))

    def test_session_bar_rate_is_measured_from_timestamps(self):
        # 7 bars per weekday only: 5 trading days per 7 calendar days.
        days = pd.bdate_range('2026-01-05', periods=60)
        index = pd.DatetimeIndex([day + pd.Timedelta(hours=14 + h) for day in days for h in range(7)])
        self.assertAlmostEqual(qe.bars_per_year_from_index(index), 7 * 5 / 7 * 365, delta=60)

    def test_zero_volume_index_still_gets_a_score(self):
        rng = np.random.default_rng(1)
        frame = ohlc_from_close(20 * np.exp(np.cumsum(rng.normal(0, 0.01, 800))))
        frame['volume'], frame['cmf'] = 0.0, np.nan  # Like ^VIX on Yahoo.
        row = qe.quant_edge_row('^VIX', frame, None, SETTINGS)
        self.assertFalse(np.isnan(row['Quant Score Q']))
        self.assertGreater(row['Volatility (ann.)'], 0)


if __name__ == '__main__':
    unittest.main()
