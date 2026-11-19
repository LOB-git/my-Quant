import unittest

import numpy as np
import pandas as pd

import flow_cvd_backtest as fb
import real_flow as rf


def klines(close, buy_share, start='2026-01-01', step='5min', volume=1000.0):
    """Kline frame like real_flow.klines_frame from closes and per-candle taker buy shares."""
    index = pd.date_range(start, periods=len(close), freq=step)
    close = np.asarray(close, float)
    quote = volume * close
    return pd.DataFrame({
        'open': close, 'high': close, 'low': close, 'close': close, 'volume': volume, 'quote_volume': quote,
        'trades': 1.0, 'taker_buy': volume * np.asarray(buy_share), 'taker_buy_quote': quote * np.asarray(buy_share),
        'close_time': index + pd.Timedelta(step),
    }, index=index)


NOW = pd.Timestamp('2030-01-01')


class SignalTests(unittest.TestCase):
    def test_buy_share_is_the_last_hour_and_sides_follow_the_threshold(self):
        n = 300
        share = np.full(n, 0.50)
        share[-12:] = 0.60          # the last hour of 5m candles: 60% buying
        share[-24:-12] = 0.40
        k = klines(100 + np.sin(np.arange(n) / 9), share)
        frame = fb.signals(k, close_gap=2.0, buy_above=0.51, now=NOW)   # gap < 2 is always "close"
        self.assertAlmostEqual(frame['buy_share'].iloc[-1], 0.60)
        self.assertAlmostEqual(frame['buy_share'].iloc[-13], 0.40)
        self.assertEqual(frame['flow_side'].iloc[-1], 1)
        self.assertEqual(frame['flow_side'].iloc[-13], -1)
        self.assertEqual(frame['flow_side'].iloc[-30], 0)          # exactly 50%: no trade
        self.assertTrue((frame[fb.VARIANTS[1]] == 0).all())         # never "not close"

    def test_close_and_not_close_split_the_flow_signals(self):
        n = 400
        rng = np.random.default_rng(1)
        k = klines(100 * np.exp(np.cumsum(rng.normal(0, 0.002, n))), rng.uniform(0.4, 0.6, n))
        frame = fb.signals(k, 0.10, 0.51, now=NOW)
        valid = frame['gap'].notna()
        close = frame[fb.VARIANTS[0]] != 0
        not_close = frame[fb.VARIANTS[1]] != 0
        self.assertFalse((close & not_close).any())
        self.assertTrue(((close | not_close) == ((frame['flow_side'] != 0) & valid)).all())
        self.assertTrue((frame.loc[close, 'gap'].abs() < 0.10).all())
        self.assertEqual(frame['gap'].first_valid_index(), frame.index[rf.CROSS_WINDOW - 1])

    def test_unclosed_candles_are_dropped(self):
        k = klines(np.linspace(100, 101, 300), np.full(300, 0.55))
        frame = fb.signals(k, now=k['close_time'].iloc[-2])
        self.assertEqual(len(frame), 299)


class TradeTests(unittest.TestCase):
    def frame(self, closes, sides):
        index = pd.date_range('2026-01-01', periods=len(closes), freq='5min')
        return pd.DataFrame({'close': closes, 'close_time': index + pd.Timedelta('5min'), 'buy_share': 0.6,
                             'gap': 0.0, 'rule': sides}, index=index)

    def test_hold_one_trade_at_a_time_and_fees(self):
        closes = [100, 101, 102, 103, 104, 105, 104, 103]
        sides = [1, 1, 1, -1, 0, 0, 0, 0]
        trades = fb.run_trades(self.frame(closes, sides), 'rule', pd.Timedelta('10min'), fee_bps=10)
        # LONG at 100 -> exit two candles later at 102; next signal after the exit is the SHORT at 103 -> 105.
        self.assertEqual(list(trades['Side']), ['LONG', 'SHORT'])
        self.assertAlmostEqual(trades['Gross %'].iloc[0], 2.0)
        self.assertAlmostEqual(trades['Net %'].iloc[0], 1.9)
        self.assertAlmostEqual(trades['Gross %'].iloc[1], -100 * (105 / 103 - 1))

    def test_signal_without_enough_future_is_not_traded(self):
        trades = fb.run_trades(self.frame([100, 101, 102], [0, 0, 1]), 'rule', pd.Timedelta('10min'))
        self.assertTrue(trades.empty)
        self.assertEqual(fb.summarize(trades)['Trades'], 0)

    def test_summary(self):
        trades = pd.DataFrame({'Side': ['LONG', 'SHORT', 'LONG'], 'Gross %': [1.0, -0.5, 0.05],
                               'Net %': [0.9, -0.6, -0.05]})
        s = fb.summarize(trades)
        self.assertEqual(s['Trades'], 3)
        self.assertAlmostEqual(s['Direction right'], 2 / 3)
        self.assertAlmostEqual(s['Win rate after fees'], 1 / 3)
        self.assertAlmostEqual(s['Profit factor'], 0.9 / 0.65)
        self.assertAlmostEqual(s['Long win rate'], 0.5)


class BacktestTests(unittest.TestCase):
    def test_flow_that_leads_price_is_found_and_random_flow_is_not(self):
        rng = np.random.default_rng(3)
        n = 3000
        share = rng.uniform(0.35, 0.65, n)
        # Price drifts in the direction of the last hour's buying: the rule should be right well above 50%.
        hour = pd.Series(share).rolling(12).mean().fillna(0.5).to_numpy()
        returns = (hour - 0.5) * 0.01 + rng.normal(0, 0.0005, n)
        leading = klines(100 * np.exp(np.cumsum(np.r_[0, returns[:-1]])), share)
        table, _, _ = fb.backtest(leading, holds={'1h': fb.HOLDS['1h']}, close_gap=2.0, now=NOW)
        self.assertGreater(table.loc[0, 'Direction right'], 0.7)

        random_walk = klines(100 * np.exp(np.cumsum(rng.normal(0, 0.003, n))), share)
        table, trades, info = fb.backtest(random_walk, holds={'1h': fb.HOLDS['1h']}, close_gap=2.0, now=NOW)
        self.assertAlmostEqual(table.loc[0, 'Direction right'], 0.5, delta=0.1)
        self.assertEqual(list(table['Rule']), list(fb.VARIANTS))
        self.assertIn('1h', trades)
        self.assertEqual(info['candles'], n)


if __name__ == '__main__':
    unittest.main()
