import unittest

import numpy as np
import pandas as pd

import flow_factor_verdict as ffv
from historical_flow_analysis import highest_flow_zone


def candles(seed, bars=1600, freq='1h', drift=0.0, start='2026-01-01'):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, bars)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, bars))
    index = pd.date_range(start, periods=bars, freq=freq)  # UTC-naive like fetch_and_analyze
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close,
                       'volume': rng.uniform(100, 200, bars)}, index=index)
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    df['atr14'] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    mid, sd = df.close.rolling(20).mean(), df.close.rolling(20).std()
    df['z_score'] = (df.close - mid) / sd
    df['momentum'] = df.close.pct_change(30)
    return df


class FactorTests(unittest.TestCase):
    def test_contributions_and_score_match_the_highest_flow_zone_formula(self):
        f = ffv.factor_frame(candles(1, bars=300), horizon=12)
        row = f.dropna(subset=['Direction Score']).iloc[-1]
        event = pd.DataFrame([{'Flow Z Change': row['Flow Z Change'], 'Momentum Z Change': row['Momentum Z Change'],
                               'Volatility Z Change': row['Volatility Z Change'], 'Price Change': row['Price Change']}])
        direction = 'rise' if row['Flow Z Change'] > 0 else 'drop'
        zone = highest_flow_zone(event, direction)
        self.assertAlmostEqual(row['Flow Contribution'] * 100, zone['Flow Contribution'])
        self.assertAlmostEqual(row['Momentum Contribution'] * 100, zone['Momentum Contribution'])
        self.assertAlmostEqual(row['Volatility Contribution'] * 100, zone['Volatility Contribution'])
        self.assertAlmostEqual(row['Direction Score'], zone['Direction Score'])

    def test_events_are_trailing_extremes_and_causal(self):
        f = ffv.factor_frame(candles(2, bars=400), horizon=12)
        rises = f.index[f['Rise Event'] == 1]
        self.assertGreater(len(rises), 5)
        for t in rises[:20]:
            window = f.loc[:t, 'Flow Z Change'].tail(ffv.EVENT_WINDOW)
            self.assertEqual(f.at[t, 'Flow Z Change'], window.max())
        # Changing the future must not change past factors.
        base = candles(2, bars=400)
        changed = base.copy()
        changed.iloc[-50:, changed.columns.get_loc('close')] *= 1.5
        a, b = ffv.factor_frame(base, 12), ffv.factor_frame(changed, 12)
        cols = [c for c in ffv.FEATURES]
        pd.testing.assert_frame_equal(a[cols].iloc[:300], b[cols].iloc[:300])

    def test_forward_return_is_the_target(self):
        f = ffv.factor_frame(candles(3, bars=300), horizon=4)
        self.assertAlmostEqual(f['Forward Return'].iloc[100], f['Close'].iloc[104] / f['Close'].iloc[100] - 1)
        self.assertTrue(f['Forward Return'].iloc[-4:].isna().all())

    def test_drops_the_forming_candle(self):
        df = candles(4, bars=10)
        now = pd.Timestamp(df.index[-1], tz='UTC') + pd.Timedelta(minutes=30)
        self.assertEqual(len(ffv.completed_candles(df, '1h', now)), 9)


class ModelTests(unittest.TestCase):
    def test_ridge_recovers_a_linear_signal(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(2000, 3))
        y = 2 * X[:, 0] - X[:, 2] + rng.normal(0, 0.1, 2000)
        model = ffv.ridge_fit(X, y, lam=1.0)
        np.testing.assert_allclose(ffv.ridge_predict(model, X[:5]), y[:5], atol=0.5)

    def test_walk_forward_never_trains_on_the_block_it_predicts(self):
        f = ffv.factor_frame(candles(5, bars=1600), horizon=12)
        calls = []
        original = ffv.ridge_fit

        def spy(X, y, lam=ffv.RIDGE):
            calls.append(len(X))
            return original(X, y, lam)
        ffv.ridge_fit = spy
        try:
            res = ffv.walk_forward(f, horizon=12, cost=0.001)
        finally:
            ffv.ridge_fit = original
        self.assertTrue(np.isnan(res['pred'][:ffv.WARMUP]).all())
        self.assertTrue(np.isfinite(res['pred'][ffv.WARMUP + 10]))
        # Training sets grow block by block; the first is limited to bars before WARMUP - horizon.
        self.assertLessEqual(calls[0], ffv.WARMUP - 12)
        self.assertEqual(calls, sorted(calls))

    def test_trades_do_not_overlap(self):
        f = ffv.factor_frame(candles(6, bars=1600), horizon=12)
        res = ffv.walk_forward(f, horizon=12, cost=0.001)
        gaps = np.diff(res['trade_positions'])
        self.assertTrue((gaps >= 12).all())

    def test_bonferroni_threshold(self):
        self.assertAlmostEqual(ffv.required_t(1), 1.6449, places=3)
        self.assertAlmostEqual(ffv.required_t(30), 2.935, places=3)  # Φ⁻¹(1 − 0.05/30)


class VerdictTests(unittest.TestCase):
    def result(self, n=50, t=4.0, forecast=0.5, cost_atr=0.1):
        return {'trades': {'n': n, 't': t, 'hit': .6, 'mean': .002}, 'forecast': forecast, 'cost_atr_now': cost_atr}

    def test_needs_trades_t_and_a_forecast_beyond_costs(self):
        self.assertEqual(ffv.verdict(self.result(), 3.0)[0], 'LONG')
        self.assertEqual(ffv.verdict(self.result(forecast=-0.5), 3.0)[0], 'SHORT')
        self.assertEqual(ffv.verdict(self.result(n=10), 3.0)[0], 'NO EDGE')
        self.assertEqual(ffv.verdict(self.result(t=2.5), 3.0)[0], 'NO EDGE')
        self.assertEqual(ffv.verdict(self.result(forecast=0.05), 3.0)[0], 'NO EDGE')
        self.assertEqual(ffv.verdict(self.result(forecast=np.nan), 3.0)[0], 'NO EDGE')

    def test_random_walks_get_no_edge(self):
        data = {f'C{i}/USDT': candles(10 + i, bars=1800) for i in range(6)}
        table, failed, details, t_needed = ffv.scan(
            list(data), lambda s, **k: data[s], '1h', 12, 1800, 0.001, workers=2)
        self.assertEqual(failed, [])
        self.assertEqual(len(table), 6)
        self.assertTrue((table['Verdict'] == 'NO EDGE').all())
        self.assertEqual(list(table.columns), ffv.TABLE_COLUMNS)

    def test_short_history_is_skipped(self):
        data = {'A/USDT': candles(1, bars=500)}
        table, failed, _, _ = ffv.scan(['A/USDT', 'B/USDT'], lambda s, **k: data.get(s), '1h', 12, 500, 0.001)
        self.assertTrue(table.empty)
        self.assertEqual(failed, ['A/USDT', 'B/USDT'])


if __name__ == '__main__':
    unittest.main()
