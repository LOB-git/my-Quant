import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import alert_runner
import zone_flow_tracker as zft
from dashboard import calculate_zone_battle_scores
from qqq_nq_signals import demand_supply_score, historical_z_components


def candles(bars=700, freq='5min', seed=3, start='2026-11-10 00:00'):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, bars)))
    open_ = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({'open': open_, 'close': close, 'high': np.maximum(open_, close) * 1.0006,
                       'low': np.minimum(open_, close) * 0.9994, 'volume': rng.uniform(50, 150, bars)},
                      index=pd.date_range(start, periods=bars, freq=freq))
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(), (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    df['atr14'] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df['z_score'] = (df.close - df.close.rolling(20).mean()) / df.close.rolling(20).std()
    df['momentum'] = df.close.pct_change(30)
    return df


def after(df, tf='5m'):
    return pd.Timestamp(df.index[-1], tz='UTC') + zft.TF_DELTA[tf] * 2


class FlowTotalsTests(unittest.TestCase):
    def test_day_grouped_totals_are_the_highest_rise_and_drop_so_far_today(self):
        df = candles()
        totals = zft.flow_totals(df, '5m')
        ch = historical_z_components(df).diff()
        score = pd.Series(np.asarray(demand_supply_score(ch['Flow (z)'], ch['Momentum (z)'], ch['Volatility (z)'],
                                                         df['close'].diff()), dtype=float), index=df.index)
        for t in df.index[[150, 300, 450, 699]]:
            today = df.index[(df.index.normalize() == t.normalize()) & (df.index <= t)]
            flow = ch['Flow (z)'].loc[today]
            if (flow > 0).any():
                self.assertAlmostEqual(totals.at[t, 'Rise Total'], score.loc[flow.idxmax()])
            if (flow < 0).any():
                self.assertAlmostEqual(totals.at[t, 'Drop Total'], -score.loc[flow.idxmin()])

    def test_higher_timeframes_use_the_trailing_24_candles(self):
        df = candles(freq='4h', bars=300)
        totals = zft.flow_totals(df, '4h')
        ch = historical_z_components(df).diff()
        score = np.asarray(demand_supply_score(ch['Flow (z)'], ch['Momentum (z)'], ch['Volatility (z)'],
                                               df['close'].diff()), dtype=float)
        i = 250
        window = ch['Flow (z)'].iloc[i - 23:i + 1]
        self.assertAlmostEqual(totals['Rise Total'].iloc[i], score[i - 23 + int(np.argmax(window.to_numpy()))])

    def test_no_look_ahead(self):
        df = candles()
        changed = df.copy()
        changed.iloc[-30:, changed.columns.get_loc('close')] *= 1.2
        a, b = zft.flow_totals(df, '5m'), zft.flow_totals(changed, '5m')
        pd.testing.assert_frame_equal(a.iloc[:600], b.iloc[:600])


class SignalTests(unittest.TestCase):
    def fake_calc(self, df, result, at):
        frame = pd.DataFrame([{'Zone Entry': df.index[at - 9], 'Verdict Time': df.index[at], 'Zone Type': 'Bullish',
                               'Zone Bottom': 1.0, 'Zone Top': 2.0, 'Battle Score': 1.5 if 'Buyer' in result else -1.5,
                               'Result': result}])
        return lambda *a, **k: (frame, df)

    def test_buyer_victory_needs_the_rise_to_win(self):
        df = candles()
        totals = zft.flow_totals(df, '5m')
        at = int(np.flatnonzero((totals['Rise Total'] > totals['Drop Total']).to_numpy())[-5])
        sig = zft.confirmed_signals(df, '5m', self.fake_calc(df, 'Strong Buyer Victory', at), 3)
        self.assertEqual(list(sig['Side']), ['BUY'])
        self.assertEqual(sig['Signal Time (UTC)'].iloc[0], (df.index[at] + pd.Timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M'))
        self.assertEqual(sig['Fill Price'].iloc[0], df['open'].iloc[at + 1])
        self.assertTrue(zft.confirmed_signals(df, '5m', self.fake_calc(df, 'Strong Seller Victory', at), 3).empty)

    def test_signal_on_the_newest_candle_has_a_pending_fill(self):
        df = candles()
        totals = zft.flow_totals(df, '5m')
        at = len(df) - 1
        result = 'Strong Buyer Victory' if totals['Rise Total'].iloc[at] > totals['Drop Total'].iloc[at] else 'Strong Seller Victory'
        sig = zft.confirmed_signals(df, '5m', self.fake_calc(df, result, at), 3)
        self.assertEqual(sig['Fill Time (UTC)'].iloc[0], '')
        self.assertTrue(np.isnan(sig['Fill Price'].iloc[0]))

    def test_real_zone_battles_end_to_end(self):
        df = candles(bars=1500)
        sig = zft.confirmed_signals(df, '5m', calculate_zone_battle_scores, 3, now=after(df))
        self.assertFalse(sig.empty)
        self.assertTrue(set(sig['Side']) <= {'BUY', 'SELL'})
        buys = sig[sig['Side'] == 'BUY']
        self.assertTrue((buys['Rise Total'] > buys['Drop Total']).all())


class TradeTests(unittest.TestCase):
    def signals(self, rows):
        return pd.DataFrame([{'Signal Time (UTC)': t, 'Side': s, 'Fill Time (UTC)': t, 'Fill Price': p} for t, s, p in rows])

    def test_flip_trades_and_open_position(self):
        sig = self.signals([('2026-11-14 10:00', 'BUY', 100.0), ('2026-11-14 10:30', 'BUY', 101.0),
                            ('2026-11-14 11:00', 'SELL', 110.0), ('2026-11-14 12:00', 'BUY', 99.0)])
        trades, open_pos = zft.trades_from_signals(sig, last_price=104.0)
        self.assertEqual(list(trades['Side']), ['LONG', 'SHORT'])          # repeated BUY ignored
        self.assertAlmostEqual(trades['Net %'].iloc[0], (110 / 100 - 1 - 0.001) * 100)
        self.assertAlmostEqual(trades['Net %'].iloc[1], (-(99 / 110 - 1) - 0.001) * 100)
        self.assertEqual(open_pos['Side'], 'LONG')
        self.assertAlmostEqual(open_pos['Unrealized %'], (104 / 99 - 1 - 0.001) * 100)
        stats = zft.summarize(trades)
        self.assertEqual(stats['Trades'], 2)
        self.assertEqual(stats['Win %'], 100.0)

    def test_pending_fills_are_skipped(self):
        sig = self.signals([('2026-11-14 10:00', 'BUY', np.nan)])
        trades, open_pos = zft.trades_from_signals(sig, 100.0)
        self.assertTrue(trades.empty)
        self.assertIsNone(open_pos)


class ForwardLogTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.log = os.path.join(self.dir.name, 'log.csv')
        self.state = os.path.join(self.dir.name, 'state.json')

    def tearDown(self):
        self.dir.cleanup()

    def sig(self, t, side='BUY', fill=100.0):
        row = {c: np.nan for c in zft.LOG_COLUMNS[:-1]}
        row.update({'Timeframe': '1h', 'Signal Time (UTC)': t, 'Side': side,
                    'Fill Time (UTC)': '' if np.isnan(fill) else t, 'Fill Price': fill})
        return row

    def test_only_signals_after_tracking_began_are_logged(self):
        now = pd.Timestamp('2026-11-14 12:00', tz='UTC')
        signals = pd.DataFrame([self.sig('2026-11-14 11:00'), self.sig('2026-11-14 12:00', 'SELL')])
        added = zft.record('BTC/USDT', '1h', signals, 100.0, now, self.log, self.state)
        self.assertEqual(added, 1)                                  # 11:00 was before tracking began
        log = zft.load_log(self.log)
        self.assertEqual(list(log['Signal Time (UTC)']), ['2026-11-14 12:00'])
        self.assertEqual(zft.load_state(self.state)['BTC/USDT|1h']['started'], '2026-11-14 12:00')

    def test_repeat_updates_do_not_duplicate_and_fill_pending_prices(self):
        t0 = pd.Timestamp('2026-11-14 12:00', tz='UTC')
        zft.record('BTC/USDT', '1h', pd.DataFrame([self.sig('2026-11-14 13:00', fill=np.nan)]), 100.0, t0, self.log, self.state)
        later = pd.DataFrame([self.sig('2026-11-14 13:00', fill=101.5), self.sig('2026-11-14 15:00', 'SELL', 99.0)])
        added = zft.record('BTC/USDT', '1h', later, 99.5, t0 + pd.Timedelta(hours=4), self.log, self.state)
        self.assertEqual(added, 1)
        log = zft.load_log(self.log)
        self.assertEqual(len(log), 2)
        self.assertEqual(log['Fill Price'].iloc[0], 101.5)
        self.assertEqual(log['Recorded (UTC)'].iloc[0], '2026-11-14 12:00')   # first recording time kept
        overview = zft.forward_overview(self.log, self.state, ['BTC/USDT'], ['1h', '4h'])
        one = overview[overview['Timeframe'] == '1h'].iloc[0]
        self.assertEqual((one['Signals'], one['Closed Trades'], one['Position']), (2, 1, 'SHORT'))
        self.assertEqual(overview[overview['Timeframe'] == '4h'].iloc[0]['Tracking Since (UTC)'], 'not started')

    def test_update_all_reports_each_pair(self):
        df = candles(bars=1500)
        calls = []

        def fetch(symbol, **kw):
            calls.append((symbol, kw.get('market'), kw['timeframe']))
            return df

        results = zft.update_all(fetch, calculate_zone_battle_scores, ['BTC/USDT', 'BTC/USD'], ['5m'],
                                 after(df), self.log, self.state)
        self.assertEqual(set(results), {('BTC/USDT', '5m'), ('BTC/USD', '5m')})
        self.assertTrue(all(r.endswith('signal(s)') for r in results.values()))
        self.assertEqual(calls, [('BTC/USDT', 'usdm', '5m'), ('BTC/USD', None, '5m')])


class RealFlowTests(unittest.TestCase):
    def with_taker(self, buy_share=None, seed=5):
        df = candles(bars=1500, seed=seed)
        rng = np.random.default_rng(seed + 1)
        share = rng.uniform(0.3, 0.7, len(df)) if buy_share is None else buy_share
        df['taker_buy'] = df['volume'] * share
        return df

    def test_zone_battle_flow_z_uses_taker_volume_when_asked(self):
        df = self.with_taker()
        proxy, data_p = calculate_zone_battle_scores(df, 10, 10 ** 6, fvg_candles=3, now=after(df))
        real, data_r = calculate_zone_battle_scores(df, 10, 10 ** 6, fvg_candles=3, now=after(df), flow='real')
        buy, sell = df['taker_buy'], df['volume'] - df['taker_buy']
        expected = (buy.rolling(20).sum() - sell.rolling(20).sum()) / df['volume'].rolling(20).sum()
        np.testing.assert_allclose(data_r['Money Flow Signal'].dropna(), expected.dropna())
        self.assertFalse(np.allclose(data_p['Flow Z'].fillna(0), data_r['Flow Z'].fillna(0)))
        self.assertEqual(list(proxy['Zone Entry']), list(real['Zone Entry']))   # same zones, different Fz
        with self.assertRaises(ValueError):
            calculate_zone_battle_scores(df.drop(columns='taker_buy'), fvg_candles=3, flow='real')

    def test_flow_totals_real_uses_taker_flow_z(self):
        df = self.with_taker()
        real = zft.flow_totals(df, '1h', 'real')
        proxy = zft.flow_totals(df, '1h', 'proxy')
        self.assertFalse(np.allclose(real.fillna(0).to_numpy(), proxy.fillna(0).to_numpy()))
        z = zft.real_flow_z(df)
        self.assertAlmostEqual(z.iloc[-1], ((lambda f: (f - f.rolling(50).mean()) / f.rolling(50).std())(
            (2 * df['taker_buy'] - df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum())).iloc[-1])

    def test_real_flow_strategy_is_fetched_from_taker_klines(self):
        df = self.with_taker()
        raw = df[['open', 'high', 'low', 'close', 'volume', 'taker_buy']]
        hist = zft.real_flow_history('BTC/USDT', '5m', fetch_klines=lambda *a: raw)
        pd.testing.assert_series_equal(hist['atr14'], df['atr14'])
        self.assertIn('BTC/USDT (real flow)', zft.DEFAULT_SYMBOLS)
        self.assertEqual(zft.SOURCES['BTC/USDT (real flow)']['flow'], 'real')
        self.assertIn('BTC/USDT (real flow)', alert_runner.DEFAULT_CONFIG['groups']['zone_flow_tracker']['symbols'])
        with mock.patch.object(zft, 'real_flow_history', lambda symbol, tf: hist) as _:
            calls = []
            sig = zft.confirmed_signals(hist, '5m', lambda *a, **k: calls.append(k) or calculate_zone_battle_scores(*a, **k), 3,
                                        after(df), 'real')
        self.assertEqual(calls[0]['flow'], 'real')
        self.assertTrue(set(sig['Side']) <= {'BUY', 'SELL'})


class RunnerTests(unittest.TestCase):
    def test_tracker_runs_every_five_minutes_without_telegram(self):
        config = {'interval_seconds': 300, 'groups': {
            'zone_flow_tracker': dict(alert_runner.DEFAULT_CONFIG['groups']['zone_flow_tracker'])}}
        calls, last_run = [], {}
        for seconds in (0, 299, 300):
            with mock.patch.object(alert_runner, 'save_zone_flow_tracker', lambda g, d, dry: calls.append(seconds)), \
                    mock.patch.object(alert_runner, 'save_state', lambda state: None), \
                    mock.patch.object(alert_runner.time, 'monotonic', lambda: seconds):
                alert_runner.run_cycle(config, None, None, '', '', {}, last_run)
        self.assertEqual(calls, [0, 300])
        self.assertEqual(alert_runner.DEFAULT_CONFIG['groups']['zone_flow_tracker']['timeframes'],
                         ['5m', '15m', '1h', '4h', '1d'])

    def test_tracker_is_the_first_tab(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / 'dashboard.py').read_text(encoding='utf-8')
        tabs = source[source.index('= st.tabs(['):]
        self.assertIn('"🎯 Zone Flow Tracker"', tabs.splitlines()[1])


if __name__ == '__main__':
    unittest.main()
