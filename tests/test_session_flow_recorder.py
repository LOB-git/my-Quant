import os
import tempfile
import unittest
from datetime import date
from unittest import mock

import numpy as np
import pandas as pd

import alert_runner
import session_flow_panel as sfp
import session_flow_recorder as sfr
from historical_flow_analysis import highest_flow_zone
from qqq_nq_signals import historical_z_components


def candles(seed=0, start='2026-11-10 00:00', bars=1000, freq='5min'):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, bars)))
    open_ = np.r_[close[0], close[:-1]]
    index = pd.date_range(start, periods=bars, freq=freq)  # UTC-naive like fetch_and_analyze
    df = pd.DataFrame({'open': open_, 'close': close, 'high': np.maximum(open_, close) * 1.001,
                       'low': np.minimum(open_, close) * 0.999, 'volume': rng.uniform(50, 150, bars)}, index=index)
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(), (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    df['atr14'] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df['z_score'] = (df.close - df.close.rolling(20).mean()) / df.close.rolling(20).std()
    df['momentum'] = df.close.pct_change(30)
    return df


class SessionClockTests(unittest.TestCase):
    def test_winter_sessions_in_utc(self):
        w = sfr.session_windows(date(2026, 11, 13))  # New York on EST (UTC-5)
        self.assertEqual(w['Asia'], (pd.Timestamp('2026-11-12 22:00', tz='UTC'), pd.Timestamp('2026-11-13 08:00', tz='UTC')))
        self.assertEqual(w['London'][0], pd.Timestamp('2026-11-13 08:00', tz='UTC'))
        self.assertEqual(w['New York'], (pd.Timestamp('2026-11-13 13:00', tz='UTC'), pd.Timestamp('2026-11-13 22:00', tz='UTC')))
        self.assertEqual(w['Day'], (w['Asia'][0], w['New York'][1]))

    def test_summer_sessions_follow_daylight_saving(self):
        w = sfr.session_windows(date(2026, 7, 15))  # EDT (UTC-4)
        self.assertEqual(w['Asia'][0], pd.Timestamp('2026-07-14 21:00', tz='UTC'))
        self.assertEqual(w['New York'][1], pd.Timestamp('2026-07-15 21:00', tz='UTC'))

    def test_trading_day_rolls_at_the_new_york_close(self):
        self.assertEqual(sfr.trading_day('2026-11-13 21:59'), date(2026, 11, 13))
        self.assertEqual(sfr.trading_day('2026-11-13 22:00'), date(2026, 11, 14))
        day, name, start, end = sfr.current_session(pd.Timestamp('2026-11-13 23:30', tz='UTC'))
        self.assertEqual((day, name), (date(2026, 11, 14), 'Asia'))
        self.assertEqual(sfr.current_session(pd.Timestamp('2026-11-13 10:00', tz='UTC'))[1], 'London')
        self.assertEqual(sfr.current_session(pd.Timestamp('2026-11-13 15:00', tz='UTC'))[1], 'New York')


class SessionRowTests(unittest.TestCase):
    def setUp(self):
        self.df = candles()
        self.now = pd.Timestamp('2026-11-13 10:02', tz='UTC')   # inside London, day 2026-11-13
        self.rows = {r['Session']: r for r in sfr.session_rows(self.df, 'BTC/USD', date(2026, 11, 13), '5m', self.now)}

    def test_matches_the_us_indices_highest_flow_zone(self):
        data = self.df.copy()
        data.index = data.index.tz_localize('UTC')
        data = data.loc[data.index + pd.Timedelta(minutes=5) <= self.now]
        changes = historical_z_components(data).diff()
        start, end = sfr.session_windows(date(2026, 11, 13))['Asia']
        window = (data.index >= start) & (data.index < end)
        events = pd.DataFrame({
            'Flow Z Change': changes['Flow (z)'], 'Momentum Z Change': changes['Momentum (z)'],
            'Volatility Z Change': changes['Volatility (z)'], 'Price Change': data['close'].diff(),
        })[window]
        events = events[events['Flow Z Change'].notna() & events['Flow Z Change'].ne(0)]
        rise, drop = highest_flow_zone(events, 'rise'), highest_flow_zone(events, 'drop')
        row = self.rows['Asia']
        for field in ('Flow Contribution', 'Momentum Contribution', 'Volatility Contribution', 'Direction Score',
                      'Flow Z Change', 'Price Change'):
            self.assertAlmostEqual(row[f'Rise {field}'], rise[field])
            self.assertAlmostEqual(row[f'Drop {field}'], drop[field])
        self.assertEqual(row['Rise Time (UTC)'], rise.name.strftime('%Y-%m-%d %H:%M'))

    def test_bias_compares_flow_plus_momentum(self):
        row = self.rows['Asia']
        rise = row['Rise Flow Contribution'] + row['Rise Momentum Contribution']
        drop = -(row['Drop Flow Contribution'] + row['Drop Momentum Contribution'])
        self.assertAlmostEqual(row['Margin'], rise - drop)
        self.assertEqual(row['Bias'], 'BUY' if rise > drop else 'SELL')

    def test_status_and_closed_candles_only(self):
        self.assertEqual(self.rows['Asia']['Status'], 'Complete')
        self.assertEqual(self.rows['London']['Status'], 'Live')
        self.assertNotIn('New York', self.rows)              # not started yet
        self.assertEqual(self.rows['London']['Bars'], 24)    # 08:00..09:55 closed by 10:02; 10:00 still forming
        self.assertEqual(self.rows['Day']['Bars'], self.rows['Asia']['Bars'] + self.rows['London']['Bars'])

    def test_missing_columns_or_empty(self):
        self.assertEqual(sfr.session_rows(pd.DataFrame(), 'X', date(2026, 11, 13)), [])
        self.assertEqual(sfr.session_rows(self.df[['open', 'close']], 'X', date(2026, 11, 13)), [])


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, 'log.csv')

    def tearDown(self):
        self.dir.cleanup()

    def test_live_rows_are_replaced_not_duplicated(self):
        df = candles()
        first = sfr.record_sessions(['BTC/USD', 'ETH/USDT'], lambda s: df, '5m',
                                    pd.Timestamp('2026-11-13 09:00', tz='UTC'), self.path)
        self.assertTrue(all(r.endswith('row(s)') for r in first.values()))
        before = sfr.load_history(self.path)
        sfr.record_sessions(['BTC/USD', 'ETH/USDT'], lambda s: df, '5m',
                            pd.Timestamp('2026-11-13 10:30', tz='UTC'), self.path)
        after = sfr.load_history(self.path)
        self.assertEqual(len(after), len(before))
        self.assertFalse(after.duplicated(sfr.KEY).any())
        london = after[(after['Trading Day'] == '2026-11-13') & (after['Session'] == 'London') & (after['Symbol'] == 'BTC/USD')]
        self.assertEqual(int(london['Bars'].iloc[0]), 30)   # 08:00..10:25 closed by 10:30 (was 12 at 09:00)
        self.assertEqual(list(after.columns), sfr.COLUMNS)

    def test_failures_are_reported_per_symbol(self):
        def fetch(symbol):
            if symbol == 'BAD/USDT':
                raise RuntimeError('down')
            return candles()
        results = sfr.record_sessions(['BTC/USD', 'BAD/USDT'], fetch, '5m',
                                      pd.Timestamp('2026-11-13 09:00', tz='UTC'), self.path)
        self.assertTrue(results['BAD/USDT'].startswith('failed'))
        self.assertEqual(set(sfr.load_history(self.path)['Symbol']), {'BTC/USD'})


class PanelHelperTests(unittest.TestCase):
    def test_symbols_and_routing(self):
        self.assertEqual(sfp.session_symbols(['BTC/USD'], 3, lambda n: ['BTC/USDT', 'ETH/USDT', 'BTC/USDT'][:n]),
                         ['BTC/USD', 'BTC/USDT', 'ETH/USDT'])
        self.assertEqual(sfp.session_symbols(['BTC/USD'], 3, lambda n: 1 / 0), ['BTC/USD'])
        calls = []
        fetch = sfp.session_fetcher(lambda symbol, **kw: calls.append((symbol, kw)), '5m')
        fetch('BTC/USD')
        fetch('ETH/USDT')
        self.assertNotIn('market', calls[0][1])
        self.assertEqual(calls[1][1]['market'], 'usdm')


class RunnerTests(unittest.TestCase):
    def test_session_flow_runs_every_five_minutes_without_telegram(self):
        config = {'interval_seconds': 300, 'groups': {'session_flow': dict(
            alert_runner.DEFAULT_CONFIG['groups']['session_flow'])}}
        calls, last_run = [], {}
        for seconds in (0, 299, 300):
            with mock.patch.object(alert_runner, 'save_session_flow', lambda g, d, dry: calls.append(seconds)), \
                    mock.patch.object(alert_runner, 'save_state', lambda state: None), \
                    mock.patch.object(alert_runner.time, 'monotonic', lambda: seconds):
                alert_runner.run_cycle(config, None, None, '', '', {}, last_run)
        self.assertEqual(calls, [0, 300])


if __name__ == '__main__':
    unittest.main()
