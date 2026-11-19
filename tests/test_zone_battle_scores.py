import unittest

import numpy as np
import pandas as pd

from dashboard import calculate_zone_battle_scores


class ZoneBattleScoreTests(unittest.TestCase):
    def test_valid_fvg_gap_generates_zone_battle_result(self):
        idx = pd.date_range('2026-01-01', periods=120, freq='h', tz='UTC')
        base = np.linspace(100, 110, len(idx))
        open_prices = base.copy()
        close_prices = base + 0.7
        high_prices = np.maximum(open_prices, close_prices) + 1.5
        low_prices = np.minimum(open_prices, close_prices) - 1.5

        # Create a bullish FVG gap where the next candle low breaks above the previous candle high.
        high_prices[12] = 100.0
        low_prices[12] = 101.0
        open_prices[12] = 99.0
        close_prices[12] = 103.0

        # Price re-enters the gap zone over the following candles to create a valid reaction window.
        for i in range(13, 20):
            open_prices[i] = 100.6
            close_prices[i] = 100.8 + ((i - 13) * 0.15)
            high_prices[i] = max(open_prices[i], close_prices[i]) + 0.3
            low_prices[i] = min(open_prices[i], close_prices[i]) - 0.2

        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': np.linspace(100, 200, len(idx)),
            'momentum': np.linspace(-0.03, 0.03, len(idx)),
        }, index=idx)

        results, _ = calculate_zone_battle_scores(df, observation_bars=5, max_zones=20)

        self.assertFalse(results.empty)
        self.assertIn('Zone Type', results.columns)
        self.assertIn('Battle Score', results.columns)
        self.assertTrue((results['Zone Type'].isin(['Bullish', 'Bearish'])).all())

        # The score uses the observed reaction candles, so it is only known at the last of them:
        # charts plot it at Verdict Time, never earlier.
        positions = {ts: i for i, ts in enumerate(df.index)}
        for _, row in results.iterrows():
            self.assertEqual(positions[row['Verdict Time']] - positions[row['Zone Entry']],
                             int(row['Bars Observed']) - 1)
        self.assertTrue(results['Verdict Time'].is_monotonic_decreasing)  # latest verdict first

    def test_three_candle_fvg_finds_zones_on_gapless_24_7_candles(self):
        # Crypto-style candles: every candle opens at the previous close, so two-candle gaps never exist.
        rng = np.random.default_rng(7)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, 600)))
        open_ = np.r_[close[0], close[:-1]]
        idx = pd.date_range('2026-01-01', periods=600, freq='5min', tz='UTC')
        df = pd.DataFrame({'open': open_, 'close': close,
                           'high': np.maximum(open_, close) * 1.0005, 'low': np.minimum(open_, close) * 0.9995,
                           'volume': rng.uniform(50, 150, 600)}, index=idx)
        df['momentum'] = df['close'].pct_change(30)
        two, _ = calculate_zone_battle_scores(df, max_zones=1000)
        three, _ = calculate_zone_battle_scores(df, max_zones=1000, fvg_candles=3)
        self.assertTrue(two.empty)
        self.assertGreater(len(three), 10)
        for _, z in three.iterrows():
            formed = df.index.get_loc(z['Zone Formed'])
            if z['Zone Type'] == 'Bullish':
                self.assertAlmostEqual(z['Zone Bottom'], df['high'].iloc[formed - 2])
                self.assertAlmostEqual(z['Zone Top'], df['low'].iloc[formed])
                self.assertGreater(df['close'].iloc[formed - 1], df['open'].iloc[formed - 1])
            else:
                self.assertAlmostEqual(z['Zone Bottom'], df['high'].iloc[formed])
                self.assertAlmostEqual(z['Zone Top'], df['low'].iloc[formed - 2])
                self.assertLess(df['close'].iloc[formed - 1], df['open'].iloc[formed - 1])
            entry = df.index.get_loc(z['Zone Entry'])
            between = df.iloc[formed + 1:entry]
            self.assertFalse(((between['low'] <= z['Zone Top']) & (between['high'] >= z['Zone Bottom'])).any())
        with self.assertRaises(ValueError):
            calculate_zone_battle_scores(df, fvg_candles=4)

    def test_crypto_tabs_use_three_candle_fvg_and_us_tab_does_not(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / 'dashboard.py').read_text(encoding='utf-8')
        self.assertEqual(source.count('fvg_candles=3,'), 2)
        us_tab = source[source.index('tab_body(tab3)'):source.index('tab_body(tab6)')]
        self.assertIn('calculate_zone_battle_scores(', us_tab)
        self.assertNotIn('fvg_candles=3', us_tab)

    def test_charts_use_verdict_time(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / 'dashboard.py').read_text(encoding='utf-8')
        self.assertNotIn("x=score_chart_df['Zone Entry']", source)
        self.assertEqual(source.count("x=score_chart_df['Verdict Time']"), 3)


if __name__ == '__main__':
    unittest.main()
