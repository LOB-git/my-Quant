import unittest

import numpy as np
import pandas as pd

import iv_history as h

DAY0 = pd.Timestamp('2026-01-01', tz='UTC')


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return {'result': self.payload}


def hourly_prices(days, seed=0):
    """Hourly OHLC for `days` whole UTC days starting DAY0."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(DAY0, periods=days * 24, freq='h')
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, len(index))))
    open_ = np.r_[100.0, close[:-1]]
    return pd.DataFrame({'open': open_, 'high': np.maximum(open_, close) * 1.001,
                         'low': np.minimum(open_, close) * 0.999, 'close': close}, index=index)


def fake_get(hourly):
    calls = []

    def get(url, params, timeout):
        calls.append(params)
        start = pd.Timestamp(params['start_timestamp'], unit='ms', tz='UTC')
        end = pd.Timestamp(params['end_timestamp'], unit='ms', tz='UTC')
        part = hourly[(hourly.index >= start) & (hourly.index <= end)]
        return FakeResponse({'ticks': [int(t.value // 1_000_000) for t in part.index],
                             **{k: part[k].tolist() for k in ('open', 'high', 'low', 'close')}, 'status': 'ok'})
    return get, calls


def dvol_series(days, start=0.40, step=0.001):
    index = pd.date_range(DAY0, periods=days, freq='D')
    series = pd.Series(start + step * np.arange(days), index=index)
    series.attrs['open'] = {day: value - 0.05 for day, value in series.items()}   # 00:00 opens differ from closes
    return series


class DailyTests(unittest.TestCase):
    def test_hourly_candles_group_into_utc_days(self):
        hourly = hourly_prices(3)
        daily = h.utc_daily(hourly)
        first = hourly.loc[hourly.index < DAY0 + pd.Timedelta(days=1)]
        self.assertEqual(list(daily['hours']), [24, 24, 24])
        self.assertAlmostEqual(daily['open'].iloc[0], first['open'].iloc[0])
        self.assertAlmostEqual(daily['close'].iloc[0], first['close'].iloc[-1])
        self.assertAlmostEqual(daily['high'].iloc[0], first['high'].max())

    def test_a_day_without_its_00_00_candle_has_no_open(self):
        daily = h.utc_daily(hourly_prices(2).iloc[3:])
        self.assertTrue(np.isnan(daily['open'].iloc[0]))
        self.assertFalse(np.isnan(daily['open'].iloc[1]))


class TableTests(unittest.TestCase):
    def setUp(self):
        self.hourly = hourly_prices(40)
        self.dvol = dvol_series(40)
        self.table = h.history_table(self.dvol, h.utc_daily(self.hourly)).set_index('Date')

    def test_range_uses_dvol_at_midnight_and_the_00_00_open(self):
        day = DAY0 + pd.Timedelta(days=25)
        row = self.table.loc[day]
        iv = self.dvol.attrs['open'][day]
        self.assertAlmostEqual(row['DVOL @00:00'], iv)
        self.assertAlmostEqual(row['DVOL close'], self.dvol[day])
        self.assertAlmostEqual(row['Daily move'], iv / np.sqrt(365))
        opening = self.hourly.loc[day, 'open']
        self.assertAlmostEqual(row['Open 00:00'], opening)
        self.assertAlmostEqual(row['Range high'], opening * (1 + iv / np.sqrt(365)))
        self.assertAlmostEqual(row['7-day high'], opening * (1 + iv * np.sqrt(7 / 365)))
        sigma = (row['Close'] / opening - 1) / row['Daily move']
        self.assertAlmostEqual(row['Close (× move)'], sigma)
        self.assertEqual(row['Result'], 'inside' if abs(sigma) <= 1 else 'closed above' if sigma > 0 else 'closed below')

    def test_realised_vol_only_uses_days_before(self):
        day = DAY0 + pd.Timedelta(days=30)
        closes = h.utc_daily(self.hourly)['close']
        before = closes[closes.index < day].tail(21)
        expected = np.log(before).diff().dropna().std() * np.sqrt(365)
        self.assertAlmostEqual(self.table.loc[day, 'Realised 20d'], expected)
        self.assertTrue(np.isnan(self.table.loc[DAY0 + pd.Timedelta(days=5), 'Realised 20d']))

    def test_seven_day_outcome_is_the_close_six_days_later(self):
        day = DAY0 + pd.Timedelta(days=20)
        closes = h.utc_daily(self.hourly)['close']
        self.assertAlmostEqual(self.table.loc[day, '7-day close'], closes[day + pd.Timedelta(days=6)])
        self.assertTrue(np.isnan(self.table.loc[DAY0 + pd.Timedelta(days=36), '7-day close']))

    def test_a_crash_day_closes_below(self):
        hourly = hourly_prices(25)
        crash = DAY0 + pd.Timedelta(days=24)
        hourly.loc[hourly.index >= crash + pd.Timedelta(hours=1), ['close', 'low']] = hourly.loc[crash, 'open'] * 0.85
        row = h.history_table(dvol_series(25), h.utc_daily(hourly)).set_index('Date').loc[crash]
        self.assertEqual(row['Result'], 'closed below')
        self.assertFalse(row['Wicks inside'])
        self.assertLess(row['Close (× move)'], -2)


class LoadDayTests(unittest.TestCase):
    def test_load_day_pages_and_keeps_30_days_ending_on_the_date(self):
        hourly = hourly_prices(120)
        get, calls = fake_get(hourly)
        day = DAY0 + pd.Timedelta(days=90)
        table, row = h.load_day('BTC', day.date(), dvol_series(120), get, now=DAY0 + pd.Timedelta(days=200))
        self.assertEqual(len(table), h.TABLE_DAYS)
        self.assertEqual(table['Date'].max(), day)
        self.assertEqual(row['Date'], day)
        self.assertGreaterEqual(len(calls), 2)                    # more than 50 days -> paged
        self.assertTrue(all(c['resolution'] == '60' and c['instrument_name'] == 'BTC-PERPETUAL' for c in calls))
        self.assertEqual(row['Result'] != 'in progress', True)

    def test_today_is_in_progress_with_realised_vol(self):
        hourly = hourly_prices(60)
        today = DAY0 + pd.Timedelta(days=59)
        now = today + pd.Timedelta(hours=10, minutes=20)
        get, _ = fake_get(hourly)
        table, row = h.load_day('ETH', today, dvol_series(60), get, now=now)
        self.assertEqual(row['Result'], 'in progress')
        self.assertIsNone(row['Wicks inside'])
        self.assertTrue(np.isfinite(row['Realised 20d']))
        self.assertAlmostEqual(row['Close'], hourly.loc[today + pd.Timedelta(hours=9), 'close'])  # last closed hour

    def test_no_data_gives_none(self):
        get, _ = fake_get(hourly_prices(1).iloc[:0])
        table, row = h.load_day('BTC', DAY0, dvol_series(5), get, now=DAY0 + pd.Timedelta(days=3))
        self.assertTrue(table.empty)
        self.assertIsNone(row)

    def test_percentile_and_utc_day(self):
        dvol = dvol_series(400)
        self.assertEqual(h.dvol_percentile(dvol, dvol.index[-1]), 100.0)
        self.assertEqual(h.utc_day('2026-03-05'), pd.Timestamp('2026-03-05', tz='UTC'))
        self.assertEqual(h.utc_day(pd.Timestamp('2026-03-05 23:00', tz='America/New_York')),
                         pd.Timestamp('2026-03-06', tz='UTC'))


if __name__ == '__main__':
    unittest.main()
