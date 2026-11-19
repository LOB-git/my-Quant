import unittest

import numpy as np
import pandas as pd

import binance_futures as bf
import derivative_crypto_panel as panel


def candles(count=120, start='2026-01-01', freq='1h', price=100.0, volume=10.0):
    index = pd.date_range(start, periods=count, freq=freq, name='date')
    close = price + np.sin(np.arange(count) / 3)
    frame = pd.DataFrame({
        'open': close - 0.2 * np.where(np.arange(count) % 2, 1, -1), 'close': close,
        'high': close + 1, 'low': close - 1, 'volume': volume,
    }, index=index)
    frame['z_score'] = np.cos(np.arange(count) / 5) * 2
    frame['momentum'] = np.sin(np.arange(count) / 7) / 10
    frame['vol_ma'] = volume
    frame['cmf'] = np.sin(np.arange(count) / 4) / 5
    frame['rsi'] = 55.0
    frame['ema_50'] = price
    frame['atr14'] = 2.0
    return frame


class FakeExchange:
    """Serves a fixed series in pages smaller than the requested limit."""

    def __init__(self, bars, page=1000):
        self.bars, self.page, self.calls = bars, page, 0

    def parse_timeframe(self, _):
        return 3600

    def milliseconds(self):
        return self.bars[-1][0]

    def parse8601(self, text):
        return int(pd.Timestamp(text).value // 1_000_000)

    def fetch_ohlcv(self, market, timeframe, since=None, limit=None):
        self.calls += 1
        self.market = market
        return [bar for bar in self.bars if bar[0] >= since][:self.page]


def hourly_bars(count, start='2026-01-01'):
    first = int(pd.Timestamp(start).value // 1_000_000)
    return [[first + i * 3_600_000, 1.0, 2.0, 0.5, 1.5, 3.0] for i in range(count)]


class SymbolAndCandleTests(unittest.TestCase):
    def test_every_symbol_form_resolves_to_the_perpetual(self):
        for form in ['BTC/USDT', 'BTCUSDT', 'btc-usdt', 'BTC/USDT:USDT']:
            self.assertEqual(bf.perpetual_symbol(form), 'BTC/USDT:USDT')

    def test_non_usdt_symbol_is_rejected(self):
        with self.assertRaises(ValueError):
            bf.perpetual_symbol('SPY')

    def test_paging_survives_pages_smaller_than_requested(self):
        exchange = FakeExchange(hourly_bars(1500), page=1000)
        frame = bf.fetch_usdm_candles('BTC/USDT', limit=1400, exchange=exchange)
        self.assertEqual(len(frame), 1400)
        self.assertGreater(exchange.calls, 1)
        self.assertEqual(exchange.market, 'BTC/USDT:USDT')
        self.assertTrue(frame.index.is_monotonic_increasing)

    def test_start_date_returns_all_history_without_duplicates(self):
        exchange = FakeExchange(hourly_bars(2500), page=1000)
        frame = bf.fetch_usdm_candles('ETHUSDT', start_date='2026-01-01', exchange=exchange)
        self.assertEqual(len(frame), 2500)
        self.assertFalse(frame.index.has_duplicates)


class MarketFilterTests(unittest.TestCase):
    def market(self, **changes):
        base = {'symbol': 'BTC/USDT:USDT', 'base': 'BTC', 'quote': 'USDT', 'swap': True, 'linear': True,
                'active': True, 'info': {'underlyingType': 'COIN'}}
        return {**base, **changes}

    def test_crypto_perpetual_is_kept(self):
        self.assertTrue(bf.is_crypto_perpetual(self.market()))

    def test_equities_dated_futures_and_stablecoins_are_excluded(self):
        self.assertFalse(bf.is_crypto_perpetual(self.market(info={'underlyingType': 'EQUITY'})))
        self.assertFalse(bf.is_crypto_perpetual(self.market(symbol='BTC/USDT:USDT-261225')))
        self.assertFalse(bf.is_crypto_perpetual(self.market(base='USDC')))
        self.assertFalse(bf.is_crypto_perpetual(self.market(swap=False)))


class OptionsGexTests(unittest.TestCase):
    def test_puts_are_negative_and_strikes_are_summed(self):
        contracts = [
            {'symbol': 'C1', 'strikePrice': '100', 'side': 'CALL', 'unit': 1},
            {'symbol': 'P1', 'strikePrice': '100', 'side': 'PUT', 'unit': 1},
            {'symbol': 'C2', 'strikePrice': '110', 'side': 'CALL', 'unit': 2},
            {'symbol': 'NO_OI', 'strikePrice': '120', 'side': 'CALL', 'unit': 1},
        ]
        marks = [{'symbol': 'C1', 'gamma': '0.5'}, {'symbol': 'P1', 'gamma': '0.5'},
                 {'symbol': 'C2', 'gamma': '0.1'}, {'symbol': 'NO_OI', 'gamma': '0.9'}]
        interest = [{'symbol': 'C1', 'sumOpenInterest': '10'}, {'symbol': 'P1', 'sumOpenInterest': '4'},
                    {'symbol': 'C2', 'sumOpenInterest': '5'}]
        profile = bf.options_gex_profile(contracts, marks, interest)
        self.assertAlmostEqual(profile.loc[100.0], 3.0)
        self.assertAlmostEqual(profile.loc[110.0], 1.0)
        self.assertNotIn(120.0, profile.index)

    def test_chain_with_no_gamma_counts_as_empty(self):
        contract = [{'symbol': 'C1', 'strikePrice': '100', 'side': 'CALL', 'unit': 1}]
        self.assertTrue(bf.options_gex_profile(
            contract, [{'symbol': 'C1', 'gamma': '0'}], [{'symbol': 'C1', 'sumOpenInterest': '10'}],
        ).empty)


class PanelCalculationTests(unittest.TestCase):
    def test_price_format_keeps_low_priced_assets_readable(self):
        self.assertEqual(panel.format_price(76000.5), '$76,000.50')
        self.assertEqual(panel.format_price(0.1234), '$0.1234')
        self.assertEqual(panel.format_price(0.000012), '$0.00001200')
        self.assertEqual(panel.format_price(np.nan), 'n/a')

    def test_overview_daily_volume_uses_a_full_24_hours_in_usdt(self):
        frame = candles(price=100.0, volume=10.0)
        row = panel.build_overview_row('BTC/USDT', frame, '1h')
        expected = (frame['close'] * frame['volume']).tail(24).sum()
        self.assertAlmostEqual(row['Est. Daily Volume'], expected)

    def test_threshold_alerts_are_not_repeated_for_the_same_candle(self):
        row = {'Symbol': 'BTC/USDT', 'Candle Time': pd.Timestamp('2026-01-01'), 'Price': 1.0,
               'Z-Score': -1.5, 'Z-Score Change': 0.1}
        seen = set()
        first = panel.new_threshold_alerts([row], 1.0, '1h', seen)
        self.assertEqual(first[0]['Alert'], 'NEGATIVE Z-SCORE')
        self.assertEqual(panel.new_threshold_alerts([row], 1.0, '1h', seen), [])
        self.assertEqual(panel.new_threshold_alerts([{**row, 'Z-Score': 0.5}], 1.0, '1h', set()), [])

    def test_order_flow_is_comparable_across_price_scales(self):
        frame = candles()
        cheap = frame.copy()
        for column in ('open', 'high', 'low', 'close'):
            cheap[column] = frame[column] / 1000
        cheap['volume'] = frame['volume'] * 1000
        high, low = panel.order_flow_summary(frame), panel.order_flow_summary(cheap)
        self.assertAlmostEqual(high['Net Flow'], low['Net Flow'], places=6)
        self.assertLessEqual(abs(high['Money Flow Signal']), 1.0)

    def test_volume_delta_totals_match(self):
        frame = panel.with_volume_delta(candles())
        self.assertAlmostEqual(frame['Volume Delta'].sum(), frame['Cumulative Volume Delta'].iloc[-1])
        self.assertTrue(((frame['Buy Volume'] == 0) | (frame['Sell Volume'] == 0)).all())

    def test_qs_ranks_stronger_asset_first_using_the_us_weights(self):
        factors = pd.DataFrame({
            'symbol': ['A', 'B', 'C'], 'momentum': [0.3, 0.0, -0.3],
            'money_flow_signal': [0.5, 0.0, -0.5], 'atr14': [0.02, 0.02, 0.02], 'z_score_raw': [1.0, 0.0, -1.0],
        })
        scored = panel.add_qs_scores(factors)
        self.assertEqual(scored['symbol'].tolist(), ['A', 'B', 'C'])
        top = scored.iloc[0]
        expected = 0.5 * top['z_momentum'] + 0.3 * top['z_flow'] + 0.1 * top['z_trend']
        self.assertAlmostEqual(top['qs_score'], expected)

    def test_qs_volatility_is_comparable_across_price_scales(self):
        self.assertAlmostEqual(
            panel.qs_factor_row('A', candles(price=100.0))['atr14'] * 100,
            panel.qs_factor_row('B', candles(price=100000.0))['atr14'] * 100000, delta=0.5,
        )

    def test_engine_wording_never_reaches_the_screen(self):
        self.assertEqual(panel.relabel_text('Positive QQQ/NQ and negative VIX daily alerts'),
                         'Positive BTC/ETH and negative BTCDOM daily alerts')
        self.assertIn('Refresh Derivative Overview', panel.relabel_text('Refresh Indices Data'))

    def test_only_pair_alerts_are_mapped_to_engine_symbols(self):
        state = {panel.ALERTS_KEY: [{'Symbol': 'BTC/USDT'}, {'Symbol': 'SOL/USDT'}, {'Symbol': 'BTCDOM/USDT'}],
                 panel.ALERT_TF_KEY: '1h', panel.ALERT_THRESHOLD_KEY: 1.0}
        inputs = panel.pair_alert_inputs(state)
        self.assertEqual([row['Symbol'] for row in inputs['daily_alerts']], ['QQQ', '^VIX'])

    def test_pair_signals_fail_closed_without_daily_alerts(self):
        history = {}
        for name, seed in [('BTC', 1), ('ETH', 2)]:
            rng = np.random.default_rng(seed)
            frame = candles(300)
            frame['close'] = 100 + np.cumsum(rng.normal(0, 0.5, 300))
            frame['open'] = frame['close'] + rng.normal(0, 0.3, 300)
            frame['volume'] = rng.uniform(5, 15, 300)
            frame['momentum'] = frame['close'].pct_change(30)
            frame['atr14'] = rng.uniform(1, 3, 300)
            frame['z_score'] = rng.normal(0, 1, 300)
            history[name] = frame
        as_of = pd.Timestamp('2026-02-01', tz='UTC')
        inputs = {'daily_alerts': [], 'alert_timeframe': '1h', 'alert_threshold': 1.0}
        signals = panel.build_pair_signals(history['BTC'], history['ETH'], '1h', 1.0, as_of, inputs)
        self.assertIn('BTC Score', signals.columns)
        self.assertIn('BTCDOM Daily Z', signals.columns)
        # Without stored alerts the daily filter must fail closed.
        self.assertFalse(signals['Signal'].isin(['BUY', 'SELL']).any())
        self.assertFalse(signals['Reason'].str.contains(r'\bQQQ\b|\bNQ\b|\bVIX\b').any())

    def test_costs_scale_with_the_median_test_window_price(self):
        slippage, fees = panel.cost_in_price_terms(candles(price=2000.0), '2026-01-01', 1.0, 10.0)
        self.assertAlmostEqual(slippage, 0.2, delta=0.01)
        self.assertAlmostEqual(fees, 2.0, delta=0.05)


if __name__ == '__main__':
    unittest.main()
