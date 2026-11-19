import math
import unittest

import numpy as np
import pandas as pd

import derivative_crypto_panel as panel
import quant_edge as qe


SETTINGS = {'equity': 1000.0, 'risk_pct': 1.0, 'atr_multiple': 1.5, 'reward_risk': 2.0,
            'threshold': 0.25, 'max_hold': 48, 'fee_bps': 10.0}


def ohlc_from_close(close):
    close = pd.Series(close, index=pd.date_range('2026-01-01', periods=len(close), freq='1h'))
    open_ = close.shift(1).fillna(close.iloc[0])
    frame = pd.DataFrame({'open': open_, 'close': close,
                          'high': np.maximum(open_, close) * 1.001, 'low': np.minimum(open_, close) * 0.999,
                          'volume': 10.0})
    mid, std = close.rolling(20).mean(), close.rolling(20).std()
    frame['z_score'] = ((close - mid) / std).fillna(0.0)
    frame['cmf'] = 0.0
    frame['atr14'] = (frame['high'] - frame['low']).ewm(alpha=1 / 14, adjust=False).mean()
    return frame


class StatisticTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_hurst_separates_random_walk_trend_and_mean_reversion(self):
        noise = self.rng.normal(0, 0.01, 3000)
        walk = 100 * np.exp(np.cumsum(noise))
        persistent = np.zeros(3000)
        for t in range(1, 3000):  # Positively autocorrelated returns.
            persistent[t] = 0.5 * persistent[t - 1] + noise[t]
        trend = 100 * np.exp(np.cumsum(persistent))
        level = np.zeros(3000)
        for t in range(1, 3000):  # OU process around zero.
            level[t] = 0.9 * level[t - 1] + noise[t]
        reverting = 100 * np.exp(level)
        self.assertAlmostEqual(qe.hurst_exponent(walk), 0.5, delta=0.06)
        self.assertGreater(qe.hurst_exponent(trend), 0.55)
        self.assertLess(qe.hurst_exponent(reverting), 0.4)

    def test_ou_half_life_matches_the_known_decay(self):
        level = np.zeros(20000)
        noise = self.rng.normal(0, 0.01, 20000)
        for t in range(1, 20000):
            level[t] = 0.9 * level[t - 1] + noise[t]
        expected = -math.log(2) / math.log(0.9)
        self.assertAlmostEqual(qe.ou_half_life(pd.Series(100 * np.exp(level))), expected, delta=1.0)

    def test_variance_ratio_is_one_for_independent_returns(self):
        ratio, z = qe.variance_ratio(self.rng.normal(0, 0.01, 5000))
        self.assertAlmostEqual(ratio, 1.0, delta=0.1)
        self.assertLess(abs(z), 3)

    def test_yang_zhang_recovers_the_true_variance(self):
        sigma, bars, steps = 0.01, 4000, 200
        path = 100 * np.exp(np.cumsum(self.rng.normal(0, sigma / math.sqrt(steps), bars * steps)))
        path = path.reshape(bars, steps)  # Real intrabar paths, so high/low carry range information.
        frame = pd.DataFrame({'open': path[:, 0], 'high': path.max(axis=1),
                              'low': path.min(axis=1), 'close': path[:, -1]})
        self.assertAlmostEqual(qe.yang_zhang_variance(frame) / sigma ** 2, 1.0, delta=0.15)

    def test_cornish_fisher_reduces_to_normal_var_without_skew_or_kurtosis(self):
        returns = pd.Series(self.rng.normal(0, 0.01, 200000))
        var, cvar = qe.tail_risk(returns)
        self.assertAlmostEqual(var, 1.645 * 0.01, delta=0.0005)
        self.assertGreater(cvar, var)

    def test_psr_rises_with_positive_drift(self):
        self.assertGreater(qe.probabilistic_sharpe(self.rng.normal(0.002, 0.01, 1000)), 0.95)
        self.assertLess(qe.probabilistic_sharpe(self.rng.normal(-0.002, 0.01, 1000)), 0.05)

    def test_regime_weight_is_clipped(self):
        self.assertEqual(qe.regime_weight(0.3), 0.0)
        self.assertEqual(qe.regime_weight(0.7), 1.0)
        self.assertAlmostEqual(qe.regime_weight(0.5), 0.5)
        self.assertEqual(qe.regime_weight(np.nan), 0.5)


class CleaningTests(unittest.TestCase):
    def test_bad_print_wick_is_capped_and_normal_bars_are_untouched(self):
        rng = np.random.default_rng(5)
        frame = ohlc_from_close(100 * np.exp(np.cumsum(rng.normal(0, 0.002, 300))))
        frame.iloc[200, frame.columns.get_loc('high')] *= 1.15  # A 15% bad print.
        cleaned = qe.clean_wicks(frame)
        self.assertLess(cleaned['high'].iloc[200] / frame['close'].iloc[200], 1.05)
        untouched = frame.index != frame.index[200]
        pd.testing.assert_series_equal(cleaned['high'][untouched], frame['high'][untouched])
        pd.testing.assert_series_equal(cleaned['low'], frame['low'])


class VolatilityChoiceTests(unittest.TestCase):
    def test_noisy_wicks_fall_back_to_close_to_close(self):
        returns = pd.Series(np.random.default_rng(2).normal(0, 0.01, 1000))
        clean = qe.annualized_volatility(0.01 ** 2, returns, 100)
        noisy = qe.annualized_volatility(0.05 ** 2, returns, 100)
        self.assertEqual(clean['Vol Method'], 'Yang-Zhang')
        self.assertAlmostEqual(clean['Volatility (ann.)'], 0.1)
        self.assertTrue(noisy['Vol Method'].startswith('Close-to-close'))
        self.assertAlmostEqual(noisy['Volatility (ann.)'], 0.1, delta=0.01)


class BacktestTests(unittest.TestCase):
    def frame(self, highs, lows, closes):
        return pd.DataFrame({'high': highs, 'low': lows, 'close': closes, 'atr14': 1.0})

    def test_target_hit_scores_reward_risk_less_fees(self):
        candles = self.frame([100, 101, 103], [100, 99.5, 101], [100, 100.5, 102.5])
        results = qe.barrier_backtest(candles, [1.0, 0, 0], 0.25, 1.0, 2.0, 10, fee_bps=10)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0], 2.0 - 0.001 * 100 / 1.0)

    def test_stop_counts_first_when_one_candle_touches_both(self):
        candles = self.frame([100, 103], [100, 98], [100, 100])
        results = qe.barrier_backtest(candles, [1.0, 0], 0.25, 1.0, 2.0, 10, fee_bps=0)
        self.assertEqual(results.tolist(), [-1.0])

    def test_short_signal_profits_from_a_fall_and_time_exit_is_marked_to_market(self):
        candles = self.frame([100, 100, 100], [100, 99.5, 99.5], [100, 99.7, 99.5])
        results = qe.barrier_backtest(candles, [-1.0, 0, 0], 0.25, 1.0, 2.0, 2, fee_bps=0)
        self.assertAlmostEqual(results[0], 0.5)

    def test_kelly_matches_the_closed_form(self):
        stats = qe.trade_statistics(np.array([2.0, 2.0, -1.0, -1.0]))
        self.assertAlmostEqual(stats['Kelly f*'], 0.5 - 0.5 / 2.0)
        self.assertAlmostEqual(stats['Profit Factor'], 2.0)
        self.assertAlmostEqual(stats['Expectancy R'], 0.5)


class TradePlanTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.candles = ohlc_from_close(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1500))))

    def test_noise_never_produces_an_oversized_plan(self):
        row = qe.quant_edge_row('X/USDT', self.candles, 24, SETTINGS)
        self.assertIn(row['Verdict'], {'LONG', 'SHORT', 'FLAT', 'NO EDGE'})
        self.assertLessEqual(row['Risk %'], SETTINGS['risk_pct'] / 100 + 1e-12)
        self.assertLessEqual(row['Leverage'], qe.MAX_LEVERAGE + 1e-9)
        if row['Verdict'] in {'LONG', 'SHORT'}:
            self.assertLessEqual(row['$ Risk'], SETTINGS['equity'] * SETTINGS['risk_pct'] / 100 + 1e-6)
        else:
            self.assertEqual(row['Size (units)'], 0.0)

    def test_unreachable_threshold_is_flat(self):
        row = qe.quant_edge_row('X/USDT', self.candles, 24, {**SETTINGS, 'threshold': 0.99})
        self.assertEqual(row['Verdict'], 'FLAT')
        self.assertTrue(np.isnan(row['Stop']))

    def test_long_plan_places_stop_below_and_target_above(self):
        row = qe.quant_edge_row('X/USDT', self.candles, 24, {**SETTINGS, 'threshold': 0.0})
        side = 1 if row['Quant Score Q'] > 0 else -1
        risk = SETTINGS['atr_multiple'] * qe.clean_wicks(self.candles)['atr14'].iloc[-1]
        self.assertAlmostEqual(row['Stop'], row['Entry'] - side * risk)
        self.assertAlmostEqual(row['Target'], row['Entry'] + side * 2 * risk)

    def test_panel_table_isolates_a_failing_symbol(self):
        table = panel.build_quant_table(
            {'X/USDT': self.candles, 'BAD/USDT': self.candles.drop(columns='z_score')}, '1h', SETTINGS,
        )
        self.assertEqual(table.set_index('Symbol').loc['BAD/USDT', 'Verdict'], 'ERROR')
        self.assertNotEqual(table.set_index('Symbol').loc['X/USDT', 'Verdict'], 'ERROR')


if __name__ == '__main__':
    unittest.main()


class QuantScanTests(unittest.TestCase):
    def fetch(self, symbol, **options):
        if symbol == 'NEW/USDT':
            return ohlc_from_close(np.linspace(1, 2, 50))  # Too little history.
        if symbol == 'DOWN/USDT':
            raise RuntimeError('rate limited')
        seed = sum(map(ord, symbol))
        return ohlc_from_close(100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, 1000))))

    def test_scan_keeps_volume_order_and_skips_short_or_failed_symbols(self):
        symbols = ['A/USDT', 'NEW/USDT', 'B/USDT', 'DOWN/USDT', 'C/USDT']
        seen = []
        table, failed, histories = panel.scan_quant_edges(
            symbols, self.fetch, '1h', SETTINGS, workers=3, progress=lambda share, s: seen.append(share),
        )
        self.assertEqual(table['Symbol'].tolist(), ['A/USDT', 'B/USDT', 'C/USDT'])
        self.assertEqual(table['Volume Rank'].tolist(), [1, 3, 5])
        self.assertEqual(sorted(failed), ['DOWN/USDT', 'NEW/USDT'])
        self.assertEqual(seen[-1], 1.0)
        self.assertIn(table.iloc[0]['Backtest Gate'], {'PASS', 'FAIL'})

    def test_ranking_puts_trades_first_by_expectancy(self):
        table = pd.DataFrame({
            'Symbol': list('ABCDE'), 'Verdict': ['FLAT', 'SHORT', 'NO EDGE', 'LONG', 'LONG'],
            'BT Expectancy R': [0.9, 0.2, 0.5, 0.1, 0.3],
        })
        self.assertEqual(panel.rank_quant_scan(table)['Symbol'].tolist(), ['E', 'B', 'D', 'A', 'C'])

    def test_watchlist_needs_a_passing_gate_and_a_near_threshold_score(self):
        table = pd.DataFrame({
            'Symbol': list('ABCD'), 'Verdict': ['FLAT', 'FLAT', 'FLAT', 'LONG'],
            'Backtest Gate': ['PASS', 'FAIL', 'PASS', 'PASS'], 'Quant Score Q': [-0.22, 0.24, 0.05, 0.4],
        })
        self.assertEqual(panel.near_trigger(table, 0.25)['Symbol'].tolist(), ['A'])
