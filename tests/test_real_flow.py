import unittest

import numpy as np
import pandas as pd

import real_flow as rf
import real_flow_panel as rfp

T0 = 1_794_600_000_000  # ms, a 5-minute boundary


def kline(i, open_, close, volume, buy_share, price=None, step=300_000):
    quote = volume * (price or close)
    return [T0 + i * step, str(open_), str(max(open_, close) + 1), str(min(open_, close) - 1), str(close),
            str(volume), T0 + (i + 1) * step - 1, str(quote), 100 + i, str(volume * buy_share),
            str(quote * buy_share), '0']


def rows(n=120, seed=0):
    rng = np.random.default_rng(seed)
    price, out = 100.0, []
    for i in range(n):
        close = price * (1 + rng.normal(0, 0.002))
        out.append(kline(i, price, close, rng.uniform(10, 20), rng.uniform(0.3, 0.7)))
        price = close
    return out


class RealFlowTests(unittest.TestCase):
    def test_market_id(self):
        self.assertEqual(rf.market_id('BTC/USDT'), 'BTCUSDT')
        self.assertEqual(rf.market_id('eth/usdt:usdt'), 'ETHUSDT')

    def test_buy_sell_delta_and_cvd(self):
        frame = rf.klines_frame([kline(0, 100, 101, 10, 0.7, price=100), kline(1, 101, 100, 10, 0.2, price=100)])
        flow = rf.flow_frame(frame)
        np.testing.assert_allclose(flow['buy_usdt'], [700, 200])
        np.testing.assert_allclose(flow['sell_usdt'], [300, 800])
        np.testing.assert_allclose(flow['delta_usdt'], [400, -600])
        np.testing.assert_allclose(flow['cvd_usdt'], [400, -200])
        self.assertEqual(frame['close_time'].iloc[0], pd.Timestamp(T0 + 300_000, unit='ms'))

    def test_real_flow_z_uses_taker_delta_not_candle_colour(self):
        # Green candles that are mostly SOLD by takers: proxy says buying, real flow says selling.
        data = [kline(i, 100, 100.5 if i >= 70 else 99.5 + (i % 2), 10, 0.2 if i >= 70 else 0.5) for i in range(100)]
        flow = rf.flow_frame(rf.klines_frame(data))
        self.assertLess(flow['real_flow_z'].iloc[-1], 0)
        self.assertGreater(flow['proxy_flow_z'].iloc[-1], 0)

    def test_summary_uses_closed_candles_and_windows(self):
        frame = rf.klines_frame(rows(400))
        last_open = frame.index[-1]
        now = last_open + pd.Timedelta(minutes=2)            # the newest candle is still forming
        s = rf.summary(frame, now)
        self.assertEqual(s['as_of'], last_open)              # = close time of the previous candle
        closed = frame.iloc[:-1]
        flow = rf.flow_frame(closed)
        last_hour = flow[flow.index >= last_open - pd.Timedelta(hours=1)]
        self.assertAlmostEqual(s['net_1h'], last_hour['delta_usdt'].sum())
        self.assertEqual(len(last_hour), 12)
        self.assertTrue(0 <= s['buy_share_24h'] <= 1)
        self.assertTrue(0 <= s['agreement'] <= 1)
        self.assertIsNone(rf.summary(frame.iloc[:1], now))

    def test_5m_and_15m_windows(self):
        frame = rf.klines_frame(rows(400))
        last_open = frame.index[-1]
        s = rf.summary(frame, last_open + pd.Timedelta(minutes=2))
        flow = rf.flow_frame(frame.iloc[:-1])
        self.assertAlmostEqual(s['net_5m'], flow['delta_usdt'].iloc[-1])
        self.assertAlmostEqual(s['net_15m'], flow['delta_usdt'].iloc[-3:].sum())
        last = flow.iloc[-1]
        self.assertAlmostEqual(s['buy_share_5m'], last['buy_usdt'] / (last['buy_usdt'] + last['sell_usdt']))
        self.assertEqual(list(rf.WINDOWS), ['5m', '15m', '1h', '4h', '24h'])

    def test_short_windows_come_from_5m_candles_on_bigger_charts(self):
        own, five = {'name': 'chart'}, {'name': '5m'}
        calls = []

        def loader(symbol, interval, limit):
            calls.append((symbol, interval))
            return 'klines'
        original = rf.summary
        rf.summary = lambda klines: five
        try:
            sources = rfp.window_sources(own, 'BTC/USDT', '15m', loader)
            self.assertEqual({k: v['name'] for k, v in sources.items()},
                             {'5m': '5m', '15m': 'chart', '1h': 'chart', '4h': 'chart', '24h': 'chart'})
            sources = rfp.window_sources(own, 'BTC/USDT', '4h', loader)
            self.assertEqual([k for k, v in sources.items() if v is five], ['5m', '15m', '1h'])
            calls.clear()
            sources = rfp.window_sources(own, 'BTC/USDT', '5m', loader)
            self.assertTrue(all(v is own for v in sources.values()))
            self.assertEqual(calls, [])                       # 5m chart: no extra download
        finally:
            rf.summary = original

    def test_money_format_and_scan_sorting(self):
        self.assertEqual(rfp.money(-12_345_678), '−$12.3M')
        self.assertEqual(rfp.money(2.5e9), '+$2.50B')
        self.assertEqual(rfp.money(np.nan), '—')
        fake = {'A/USDT': -1.0, 'B/USDT': 5.0, 'C/USDT': None}

        def fake_row(symbol):
            return None if fake[symbol] is None else {'Symbol': symbol, 'Net Flow 24h': fake[symbol]}
        original = rfp.scan_row
        rfp.scan_row = fake_row
        try:
            table = rfp.scan(list(fake), workers=2)
        finally:
            rfp.scan_row = original
        self.assertEqual(list(table['Symbol']), ['B/USDT', 'A/USDT'])


class PriceCvdCrossTests(unittest.TestCase):
    def frame(self, price, cvd):
        idx = pd.date_range('2026-01-01', periods=len(price), freq='5min')
        return pd.DataFrame({'close': price, 'cvd_usdt': cvd}, index=idx)

    def test_gap_matches_chart_scaling_and_finds_crosses(self):
        n = 30
        # price dips while CVD makes highs (price drawn below CVD), then price makes highs while CVD falls (above)
        price = np.r_[np.linspace(100, 120, 15), np.linspace(119, 115, 5), np.linspace(118, 130, 10)]
        cvd = np.r_[np.linspace(0, 50, 20), np.linspace(48, 10, 10)]
        g = rf.price_cvd_gap(self.frame(price, cvd), window=10)
        i = 15
        p, c = price[i - 9:i + 1], cvd[i - 9:i + 1]
        expected = (price[i] - p.min()) / (p.max() - p.min()) - (cvd[i] - c.min()) / (c.max() - c.min())
        self.assertAlmostEqual(g['gap'].iloc[i], expected)
        self.assertTrue(g['gap'].iloc[:9].isna().all())   # needs a full window
        self.assertTrue(g['cross'].any())
        first = g.index[g['cross']][0]
        self.assertLess(g['gap'].shift().loc[first] * g['gap'].loc[first], 0)
        self.assertEqual(g.loc[first, 'cross_dir'], 1.0)                 # price went above CVD
        self.assertLess(g['gap'].iloc[19], 0)
        self.assertTrue((g.loc[~g['cross'], 'cross_dir'] == 0).all())

    def test_gap_is_causal(self):
        rng = np.random.default_rng(1)
        price, cvd = 100 + rng.normal(0, 1, 300).cumsum(), rng.normal(0, 1, 300).cumsum()
        full = rf.price_cvd_gap(self.frame(price, cvd))
        cut = rf.price_cvd_gap(self.frame(price[:250], cvd[:250]))
        pd.testing.assert_frame_equal(full.iloc[:250], cut)

    def test_cross_watch_window(self):
        idx = pd.date_range('2026-01-01', periods=6, freq='5min')
        gaps = pd.DataFrame({'gap': [0.3, 0.2, -0.1, -0.2, -0.3, -0.4],
                             'cross': [False, False, True, False, False, False],
                             'cross_dir': [0, 0, -1.0, 0, 0, 0]}, index=idx)
        w = rf.cross_watch(gaps.iloc[:4], watch=3)          # cross 1 candle ago
        self.assertTrue(w['active'])
        self.assertEqual((w['candles_since'], w['remaining'], w['last_dir']), (1, 2, -1.0))
        self.assertEqual(w['last_cross'], idx[2])
        w = rf.cross_watch(gaps, watch=3)                   # 3 candles ago: watch over
        self.assertFalse(w['active'])
        self.assertEqual(w['candles_since'], 3)
        self.assertFalse(rf.cross_watch(gaps.iloc[:2])['active'])
        self.assertIsNone(rf.cross_watch(gaps.iloc[:2])['last_cross'])

    def test_summary_and_chart_include_crosses(self):
        frame = rf.klines_frame(rows(400))
        s = rf.summary(frame, frame.index[-1] + pd.Timedelta(minutes=2))
        self.assertIn('gaps', s)
        self.assertEqual(len(s['gaps']), len(s['flow']))
        self.assertIn('active', s['cross_watch'])
        fig = rfp.flow_chart(s, 'BTC/USDT', '5m')
        names = [t.name for t in fig.data]
        if s['gaps'].reindex(s['flow'].tail(200).index)['cross'].any():
            self.assertIn('Price–CVD cross', names)


class FakeExchange:
    """Serves raw klines in pages of `page` rows from a fixed list, like Binance's startTime paging."""

    def __init__(self, data, page=50):
        self.data, self.page, self.calls = data, page, []

    def fapiPublicGetKlines(self, params):
        self.calls.append(params)
        rows = [r for r in self.data if r[0] >= params['startTime']]
        return rows[:min(self.page, params['limit'])]


def history_like_dashboard(data):
    """What fetch_and_analyze returns for these klines: OHLCV + factor columns, UTC-naive index, no taker volume."""
    frame = rf.klines_frame(data)
    hist = frame[['open', 'high', 'low', 'close', 'volume']].copy()
    tr = pd.concat([hist.high - hist.low, (hist.high - hist.close.shift()).abs(),
                    (hist.low - hist.close.shift()).abs()], axis=1).max(axis=1)
    hist['atr14'] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    hist['z_score'] = (hist.close - hist.close.rolling(20).mean()) / hist.close.rolling(20).std()
    hist['momentum'] = hist.close.pct_change(30)
    return hist, frame


class DerivativeRealFlowTests(unittest.TestCase):
    def test_attach_taker_volume_pages_and_aligns(self):
        data = rows(300)
        hist, frame = history_like_dashboard(data)
        ex = FakeExchange(data, page=70)
        out = rf.attach_taker_volume(hist, 'BTC/USDT', '5m', exchange=ex)
        np.testing.assert_allclose(out['taker_buy'], frame['taker_buy'])
        self.assertGreater(len(ex.calls), 3)                       # paged
        self.assertEqual(ex.calls[0]['symbol'], 'BTCUSDT')
        self.assertNotIn('taker_buy', hist)                        # input untouched
        aware = hist.copy()
        aware.index = aware.index.tz_localize('UTC')
        np.testing.assert_allclose(rf.attach_taker_volume(aware, 'BTC/USDT', '5m', exchange=FakeExchange(data))['taker_buy'],
                                   frame['taker_buy'])
        with self.assertRaises(RuntimeError):
            rf.attach_taker_volume(hist, 'BTC/USDT', '5m', exchange=FakeExchange([]))

    def test_z_components_and_highest_rise_drop_use_real_flow(self):
        from historical_flow_analysis import analyze_ghana_flow_events, highest_flow_zone
        from qqq_nq_signals import historical_z_components
        data = rows(600)
        hist, frame = history_like_dashboard(data)
        hist = rf.attach_taker_volume(hist, 'BTC/USDT', '5m', exchange=FakeExchange(data, 1500))
        real = historical_z_components(hist, 'real')
        proxy = historical_z_components(hist)
        # Same 20/50 formula on taker volume in base units (the USDT panel uses quote units: close, not equal).
        buy, vol = frame['taker_buy'], frame['volume']
        flow = (buy.rolling(20).sum() - (vol - buy).rolling(20).sum()) / vol.rolling(20).sum()
        expected = (flow - flow.rolling(50).mean()) / flow.rolling(50).std()
        np.testing.assert_allclose(real['Flow (z)'].dropna(), expected.dropna())
        self.assertGreater(np.corrcoef(real['Flow (z)'].dropna(), rf.flow_frame(frame)['real_flow_z'].dropna())[0, 1], 0.99)
        pd.testing.assert_series_equal(real['Momentum (z)'], proxy['Momentum (z)'])
        self.assertFalse(np.allclose(real['Flow (z)'].fillna(0), proxy['Flow (z)'].fillna(0)))
        day = hist.index[-1].date()
        _, _, real_events = analyze_ghana_flow_events(hist, 'BTC/USDT', day, flow='real')
        _, _, proxy_events = analyze_ghana_flow_events(hist, 'BTC/USDT', day)
        self.assertFalse(real_events.empty)
        self.assertNotEqual(list(real_events['Flow Z Change'].round(6)), list(proxy_events['Flow Z Change'].round(6)))
        rise = highest_flow_zone(real_events, 'rise')
        self.assertAlmostEqual(rise['Flow Z Change'], real_events['Flow Z Change'].max())
        with self.assertRaises(ValueError):
            historical_z_components(hist.drop(columns='taker_buy'), 'real')

    def test_quant_verdict_accepts_real_flow(self):
        import alert_quant_verdict as aq
        data = rows(600)
        hist, _ = history_like_dashboard(data)
        hist = rf.attach_taker_volume(hist, 'BTC/USDT', '5m', exchange=FakeExchange(data, 1500))
        _, real_score = aq.scored_history(hist, 'real')
        _, proxy_score = aq.scored_history(hist)
        self.assertFalse(np.allclose(np.nan_to_num(real_score), np.nan_to_num(proxy_score)))


if __name__ == '__main__':
    unittest.main()
