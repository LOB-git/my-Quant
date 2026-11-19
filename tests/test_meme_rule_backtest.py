import unittest

import numpy as np
import pandas as pd

import meme_rule_backtest as mb

T0 = pd.Timestamp('2026-11-18 10:00')


def candles(closes, volumes=None, start=T0 - pd.Timedelta(hours=1)):
    index = pd.date_range(start, periods=len(closes), freq='5min')
    closes = np.asarray(closes, float)
    opens = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame({'open': opens, 'high': np.maximum(opens, closes), 'low': np.minimum(opens, closes),
                         'close': closes, 'volume_usd': volumes if volumes is not None else np.ones(len(closes))},
                        index=index)


class CostTests(unittest.TestCase):
    def test_round_trip_cost_uses_the_pool_formula(self):
        # $1,000 into a $40,000-liquidity pool ($20,000 per side): impact (1.05)^2 - 1 = 10.25% each way.
        cost = mb.round_trip_cost(1000, 40_000)
        self.assertAlmostEqual(cost, 2 * 0.0025 + 2 * 0.1025 + 2 * 0.005)
        self.assertLess(mb.round_trip_cost(100, 40_000), cost)
        self.assertTrue(np.isinf(mb.round_trip_cost(100, 0)))


class FeatureTests(unittest.TestCase):
    def test_entry_returns_worst_drop_and_surge(self):
        closes = [1.0] * 12 + [1.0, 1.1, 1.2, 0.4, 1.3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.6, 1.7]
        volumes = [1.0] * 9 + [6.0] * 3 + [1.0] * 13       # last 15 min before 10:00: 18 vs a usual 3
        bars = candles(closes, volumes)
        f = mb.candle_features(bars, T0 + pd.Timedelta(minutes=2))   # snapshot at 10:02 -> entry bar 10:05
        entry = bars.at[T0 + pd.Timedelta(minutes=5), 'open']
        self.assertAlmostEqual(f['entry'], entry)
        self.assertAlmostEqual(f['ret_15m'], bars.at[T0 + pd.Timedelta(minutes=15), 'close'] / entry - 1)
        self.assertAlmostEqual(f['ret_60m'], bars.at[T0 + pd.Timedelta(minutes=60), 'close'] / entry - 1)
        self.assertAlmostEqual(f['worst_60m'], 0.4 / entry - 1)
        self.assertAlmostEqual(f['surge'], 6.0)

    def test_snapshot_on_a_bar_boundary_enters_that_bar(self):
        bars = candles(np.linspace(1, 2, 30))
        f = mb.candle_features(bars, T0)
        self.assertAlmostEqual(f['entry'], bars.at[T0, 'open'])

    def test_missing_history_or_future(self):
        bars = candles(np.ones(20))                     # ends 10:35: no 60 minutes after entry
        self.assertIsNone(mb.candle_features(bars, T0))
        young = candles(np.ones(16), start=T0 - pd.Timedelta(minutes=10))
        self.assertTrue(np.isnan(mb.candle_features(young, T0)['surge']))   # no 60 min of history


class RuleTests(unittest.TestCase):
    def frame(self):
        times = [T0, T0 + pd.Timedelta(minutes=20), T0 + pd.Timedelta(minutes=70), T0]
        return pd.DataFrame({
            'Timestamp': times, 'Pair Address': ['A', 'A', 'A', 'B'],
            'Buy Pressure': [0.4, 0.4, 0.6, -0.2], 'surge': [2.5, 3.0, 1.0, 5.0],
            'Liquidity USD': [100_000.0] * 4, 'ret_15m': [0.05, 0.1, -0.2, 0.3],
            'ret_60m': [0.10, 0.2, -0.6, 0.5], 'worst_60m': [-0.05, -0.1, -0.7, -0.1],
        })

    def test_one_trade_per_coin_per_hour(self):
        kept = mb.one_trade_per_hour(self.frame())
        self.assertEqual(sorted(zip(kept['Pair Address'], kept['Timestamp'])),
                         [('A', T0), ('A', T0 + pd.Timedelta(minutes=70)), ('B', T0)])

    def test_grid_filters_and_scores_after_costs(self):
        grid = mb.rule_grid(self.frame(), 100, pressures=(None, 0.3), surges=(None, 2.0)).set_index(
            ['Buy pressure >', 'Volume surge ≥'])
        self.assertEqual(grid.loc[('any', 'any'), 'Trades'], 3)
        rule = grid.loc[(0.3, 2.0)]
        self.assertEqual(rule['Trades'], 1)                       # A at 10:00 only
        cost = mb.round_trip_cost(100, 100_000)
        self.assertAlmostEqual(rule['Avg net'], 0.10 - cost)
        self.assertAlmostEqual(grid.loc[(0.3, 'any'), 'Collapsed 50%+'], 0.5)   # A 10:00 fine, A 11:10 collapsed
        self.assertEqual(mb.evaluate(self.frame().iloc[:0], 100), {'Trades': 0, 'Coins': 0})


if __name__ == '__main__':
    unittest.main()
