import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import volatility_quantum_scan as vq

MODULE_PATH = Path(__file__).resolve().parents[1] / 'dashboard.py'


def candles(seed, bars=400, drift=0.0, freq='1h'):
    rng = np.random.default_rng(seed)
    returns = drift + rng.normal(0, 0.01, bars) * np.linspace(0.5, 1.5, bars)
    close = 100 * np.exp(np.cumsum(returns))
    index = pd.date_range('2026-09-01', periods=bars, freq=freq, tz='UTC')
    frame = pd.DataFrame({'open': close, 'high': close * 1.01, 'low': close * 0.99, 'close': close,
                          'volume': 1.0}, index=index)
    frame['ema_50'] = frame['close'].ewm(span=50, adjust=False).mean()
    frame['rsi'] = vq._rsi(frame['close'])
    return frame


class QuantumStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('quant_app_vq', MODULE_PATH)
        cls.app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.app)

    def test_prediction_matches_the_tab_signal(self):
        for seed in range(25):
            for drift in (-0.002, 0.0, 0.002):
                frame = candles(seed, drift=drift)
                with self.subTest(seed=seed, drift=drift):
                    self.assertEqual(vq.quantum_state(frame)['Prediction'],
                                     self.app.get_quantum_signal_for_candle(frame))

    def test_prediction_follows_outlook_trend_and_momentum(self):
        for seed in range(25):
            state = vq.quantum_state(candles(seed, drift=0.001))
            expected = 'CONSOLIDATE / CHOP'
            if state['Volatility Outlook'] == 'INCREASE':
                if state['Price Trend'] == 'BULLISH' and state['Momentum'] == 'POSITIVE':
                    expected = 'MOVE UP'
                elif state['Price Trend'] == 'BEARISH' and state['Momentum'] == 'NEGATIVE':
                    expected = 'MOVE DOWN'
            self.assertEqual(state['Prediction'], expected)
            self.assertTrue(0 < state['Vol Percentile'] <= 100)

    def test_outlook_reverts_toward_the_median(self):
        self.assertEqual(vq.volatility_outlook([0.5, 0.6, 0.7, 0.2]), ('INCREASE', 0.55))
        self.assertEqual(vq.volatility_outlook([0.5, 0.6, 0.7, 0.9])[0], 'DECREASE')
        self.assertEqual(vq.volatility_outlook([0.5, 0.7, 0.6])[0], 'NEUTRAL')  # sitting on the median
        self.assertEqual(vq.volatility_outlook([np.nan, 0.3, 0.1])[0], 'INCREASE')  # NaNs ignored
        self.assertEqual(vq.volatility_outlook([np.nan])[0], 'NEUTRAL')
        # Only the last OUTLOOK_LOOKBACK readings count: an old high regime drops out.
        old_high = [1.0] * 5000 + [0.2] * vq.OUTLOOK_LOOKBACK
        self.assertEqual(vq.volatility_outlook(old_high[:-1] + [0.3])[0], 'DECREASE')

    def test_state_reports_the_median_it_compared_against(self):
        for seed in range(10):
            state = vq.quantum_state(candles(seed))
            expected = ('INCREASE' if state['Volatility'] < state['Vol Median']
                        else 'DECREASE' if state['Volatility'] > state['Vol Median'] else 'NEUTRAL')
            self.assertEqual(state['Volatility Outlook'], expected)

    def test_surface_returns_the_same_outlook(self):
        frame = candles(4)
        fig, data = self.app.plot_volatility_surface(frame.copy(), 'TEST')
        self.assertEqual(data['vol_outlook'], vq.quantum_state(frame)['Volatility Outlook'])
        self.app.plt.close(fig)

    def test_surface_without_data_still_returns_two_values(self):
        # Yahoo rate-limits Streamlit Cloud; the page unpacks (fig, data) and crashed on a single None.
        for frame in (None, pd.DataFrame(), candles(5, bars=30)):
            fig, data = self.app.plot_volatility_surface(frame, 'TEST')
            self.assertIsNone(fig)
            self.assertIsNone(data)

    def test_too_little_history_returns_none(self):
        self.assertIsNone(vq.quantum_state(candles(1, bars=40)))
        self.assertIsNone(vq.quantum_state(None))

    def test_expected_daily_move_uses_daily_returns_or_scales_short_history(self):
        hourly = candles(2, bars=1000)
        daily = np.log(hourly['close'].resample('1D').last().dropna()).diff().dropna()
        self.assertAlmostEqual(vq.expected_daily_move(hourly), daily.tail(20).std())
        quarter_hour = candles(3, bars=300, freq='15min')  # about 3 days: scaled from the 15m returns
        move = vq.expected_daily_move(quarter_hour)
        bar_std = np.log(quarter_hour['close']).diff().dropna().tail(200).std()
        self.assertAlmostEqual(move, bar_std * np.sqrt(96))


class ScanTests(unittest.TestCase):
    def test_scan_keeps_volume_rank_skips_short_or_failed_symbols_and_ranks_moves_first(self):
        data = {f'C{i}/USDT': candles(i, drift=0.003 if i % 2 else -0.003) for i in range(8)}
        data['SHORT/USDT'] = candles(99, bars=50)

        def fetch(symbol, **_):
            if symbol == 'BROKEN/USDT':
                raise RuntimeError('down')
            return data[symbol]

        symbols = ['BROKEN/USDT', *data]
        table, failed, histories = vq.scan_volatility_quantum(symbols, fetch, '1h', workers=2)
        self.assertEqual(failed, ['BROKEN/USDT', 'SHORT/USDT'])
        self.assertEqual(len(table), 8)
        self.assertEqual(list(table['Volume Rank']), list(range(2, 10)))
        self.assertNotIn('SHORT/USDT', histories)
        ranked = vq.rank_quantum_scan(table)
        moves = ranked['Prediction'].isin(['MOVE UP', 'MOVE DOWN']).to_list()
        self.assertEqual(moves, sorted(moves, reverse=True))  # every MOVE row before every CHOP row
        self.assertTrue((ranked['Range Low'] < ranked['Price']).all())
        self.assertTrue((ranked['Range High'] > ranked['Price']).all())

    def test_empty_scan(self):
        table, failed, _ = vq.scan_volatility_quantum(['X/USDT'], lambda *a, **k: None, '1h')
        self.assertTrue(table.empty)
        self.assertEqual(failed, ['X/USDT'])
        self.assertTrue(vq.rank_quantum_scan(table).empty)


if __name__ == '__main__':
    unittest.main()
