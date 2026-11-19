import unittest

import numpy as np
import pandas as pd

import top_derivatives_panel as panel
from qqq_nq_signals import demand_supply_score, historical_z_components


def history(count=400, seed=7):
    rng = np.random.default_rng(seed)
    index = pd.date_range('2026-01-01', periods=count, freq='1h', name='date')
    close = 100 + np.cumsum(rng.normal(0, 0.6, count))
    frame = pd.DataFrame({
        'close': close, 'open': close + rng.normal(0, 0.4, count), 'volume': rng.uniform(5, 50, count),
    }, index=index)
    frame['high'] = frame[['open', 'close']].max(axis=1) + 0.5
    frame['low'] = frame[['open', 'close']].min(axis=1) - 0.5
    frame['momentum'] = frame['close'].pct_change(30)
    frame['atr14'] = (frame['high'] - frame['low']).rolling(14).mean()
    frame['z_score'] = (frame['close'] - frame['close'].rolling(20).mean()) / frame['close'].rolling(20).std()
    return frame


class RankingTests(unittest.TestCase):
    def test_rank_follows_24h_volume_not_scan_order(self):
        ranked = panel.rank_scan(pd.DataFrame({'symbol': ['A', 'B', 'C'], '24h_volume': [5.0, 50.0, 20.0]}))
        self.assertEqual(ranked['symbol'].tolist(), ['B', 'C', 'A'])
        self.assertEqual(ranked['Rank'].tolist(), [1, 2, 3])
        self.assertEqual(ranked.index.tolist(), [0, 1, 2])


class UsSectionParityTests(unittest.TestCase):
    """Recalculate each result the way the US tab's inline code does."""

    def setUp(self):
        self.data = history()
        self.day = pd.Timestamp('2026-01-12').date()
        self.view = panel.build_historical_z_view(self.data, 'BTC/USDT', '1h', self.day)
        full = historical_z_components(self.data)
        self.mask = np.asarray(self.data.index.date) == self.day
        self.z_df = full.loc[self.mask]
        self.full_changes = full.diff().loc[self.mask]

    def test_components_are_calculated_on_full_history_then_filtered(self):
        np.testing.assert_allclose(
            self.view['components'].to_numpy(), self.z_df.to_numpy(), equal_nan=True,
        )

    def test_alert_rows_scores_and_labels_match(self):
        changes = self.full_changes['Flow (z)'].dropna()
        changes = changes[changes != 0]
        alerts = self.view['alerts'].sort_values('Ghana Time').reset_index(drop=True)
        self.assertEqual(len(alerts), len(changes))
        for (moment, flow), (_, row) in zip(changes.items(), alerts.iterrows()):
            position = self.data.index.get_loc(moment)
            price_change = self.data['close'].iloc[position] - self.data['close'].iloc[position - 1]
            score = float(demand_supply_score(
                flow, self.full_changes.at[moment, 'Momentum (z)'],
                self.full_changes.at[moment, 'Volatility (z)'], price_change,
            ))
            self.assertAlmostEqual(row['Direction Score'], score)
            self.assertEqual(row['Alert'], 'Flow Z Rise Alert' if flow > 0 else 'Flow Z Drop Alert')
            expected = 'Demand Alert' if score >= 10 else 'Supply Alert' if score <= -10 else 'Unresolved Alert'
            self.assertEqual(row['Demand/Supply Alert'], expected)
            self.assertEqual(row['Timeframe'], '1h')

    def test_highest_zones_use_moves_within_the_selected_day(self):
        within_day = self.z_df['Flow (z)'].dropna().diff().dropna()
        drop, rise = self.view['highest_drop'], self.view['highest_rise']
        self.assertEqual(pd.Timestamp(drop['Ghana Time']).tz_localize(None), within_day.idxmin().tz_localize(None))
        self.assertEqual(pd.Timestamp(rise['Ghana Time']).tz_localize(None), within_day.idxmax().tz_localize(None))
        self.assertAlmostEqual(drop['Event Size'], abs(within_day.min()))
        self.assertAlmostEqual(rise['Event Size'], within_day.max())

    def test_first_candle_of_the_day_cannot_be_the_highest_zone(self):
        first = self.view['components'].index[0]
        self.assertNotEqual(self.view['highest_drop']['Ghana Time'], first)
        self.assertNotEqual(self.view['highest_rise']['Ghana Time'], first)

    def test_drop_table_lists_only_downward_moves(self):
        drops = self.view['drops']
        self.assertFalse(drops.empty)
        self.assertLessEqual(len(drops), 50)
        self.assertTrue((drops['Drop Size'] > 0).all())
        self.assertTrue(drops['Drop Time'].is_monotonic_decreasing)


class EdgeCaseTests(unittest.TestCase):
    def test_new_listing_without_z_history_produces_no_events(self):
        short = history(count=30)
        view = panel.build_historical_z_view(short, 'NEW/USDT', '1h', short.index[-1].date())
        self.assertTrue(view['alerts'].empty)
        self.assertTrue(view['drops'].empty)
        self.assertIsNone(view['highest_drop'])
        self.assertIsNone(view['highest_rise'])

    def test_date_outside_history_is_empty(self):
        view = panel.build_historical_z_view(history(), 'BTC/USDT', '1h', pd.Timestamp('2020-01-01').date())
        self.assertTrue(view['components'].empty)
        self.assertIsNone(view['highest_rise'])

    def test_undefined_score_is_labelled_not_guessed(self):
        self.assertEqual(panel.flow_alert_label(float('nan')), 'Insufficient factor data')


if __name__ == '__main__':
    unittest.main()
