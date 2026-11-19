import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import dex_candles as dex
import meme_coin_panel as panel


NOW = pd.Timestamp('2026-03-01 12:30', tz='UTC')
ASSET = 'solana:POOL:TOKEN'


def api_rows(count=1000, price=2e-6, end=pd.Timestamp('2026-03-01 12:00'), seconds=3600):
    """Newest-first rows in the provider's shape, with USD volume."""
    rng = np.random.default_rng(7)
    closes = price * np.exp(np.cumsum(rng.normal(0, 0.03, count)))
    stamp = int(end.timestamp())
    rows = []
    for offset in range(count):
        close = closes[offset]
        opened = close * (1 + rng.normal(0, 0.01))
        rows.append([
            stamp - offset * seconds, opened, max(opened, close) * 1.01,
            min(opened, close) * 0.99, close, float(rng.uniform(1000, 9000)),
        ])
    return rows


class FakeResponse:
    def __init__(self, rows=None, status=200):
        self.status_code = status
        self.rows = rows

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f'{self.status_code}')

    def json(self):
        return {'data': {'attributes': {'ohlcv_list': self.rows}}}


class FakeSession:
    """Serves newest-first pages like the API, refusing history past `oldest_allowed`."""

    def __init__(self, rows, responses=None, oldest_allowed=None):
        self.rows, self.responses, self.calls = rows, list(responses or []), []
        self.oldest_allowed = oldest_allowed

    def get(self, url, params, headers, timeout):
        self.calls.append((url, dict(params)))
        if self.responses:
            return self.responses.pop(0)
        before = params.get('before_timestamp')
        if before is not None and self.oldest_allowed is not None and before < self.oldest_allowed:
            return FakeResponse(status=401)
        available = [row for row in self.rows if before is None or row[0] < before]
        return FakeResponse(available[:params['limit']])


def no_wait(*_):
    return None


def fetch(session, **options):
    return dex.fetch_dex_candles(
        ASSET, session=session, sleep=no_wait, throttle=no_wait, now=NOW, **options,
    )


class CandleShapeTests(unittest.TestCase):
    def test_usd_volume_becomes_coin_units_so_notional_is_usd_again(self):
        rows = [[int(pd.Timestamp('2026-03-01 11:00').timestamp()), 2e-6, 3e-6, 1e-6, 2.5e-6, 5000.0]]
        frame = dex.candles_from_rows(rows, '1h', NOW)
        self.assertAlmostEqual(frame['volume'].iloc[0], 5000.0 / 2.5e-6)
        self.assertAlmostEqual((frame['volume'] * frame['close']).iloc[0], 5000.0)

    def test_quiet_periods_become_flat_zero_volume_candles(self):
        hour = 3600
        first = int(pd.Timestamp('2026-03-01 08:00').timestamp())
        rows = [[first, 1.0, 1.2, 0.9, 1.1, 10.0], [first + 2 * hour, 1.1, 1.3, 1.0, 1.2, 10.0]]
        frame = dex.candles_from_rows(rows, '1h', NOW)
        self.assertEqual(frame.index[-1], pd.Timestamp('2026-03-01 12:00'))
        self.assertEqual(len(frame), 5)
        quiet = frame.loc['2026-03-01 09:00']
        self.assertEqual(quiet['volume'], 0)
        self.assertTrue(quiet[['open', 'high', 'low', 'close']].eq(1.1).all())
        self.assertEqual(int(quiet['timestamp']), first + hour)

    def test_index_is_utc_naive_ascending_and_unbroken(self):
        frame = dex.candles_from_rows(api_rows(50), '1h', NOW)
        self.assertIsNone(frame.index.tz)
        self.assertTrue(frame.index.is_monotonic_increasing)
        self.assertTrue((frame.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all())

    def test_empty_and_unusable_rows_give_an_empty_frame(self):
        self.assertTrue(dex.candles_from_rows([], '1h', NOW).empty)
        self.assertTrue(dex.candles_from_rows([[1, 0, 0, 0, 0, 5]], '1h', NOW).empty)


class FetchTests(unittest.TestCase):
    def test_requests_the_coin_side_of_the_pool_in_usd(self):
        session = FakeSession(api_rows(100))
        fetch(session)
        url, params = session.calls[0]
        self.assertIn('/networks/solana/pools/POOL/ohlcv/hour', url)
        self.assertEqual(params['token'], 'TOKEN')
        self.assertEqual(params['currency'], 'usd')

    def test_timeframes_map_to_period_and_aggregate(self):
        for timeframe, period, aggregate in [('5m', 'minute', 5), ('4h', 'hour', 4), ('1d', 'day', 1)]:
            session = FakeSession(api_rows(10))
            fetch(session, timeframe=timeframe)
            self.assertIn(f'/ohlcv/{period}', session.calls[0][0])
            self.assertEqual(session.calls[0][1]['aggregate'], aggregate)

    def test_default_uses_one_page_and_limit_is_applied(self):
        session = FakeSession(api_rows(1500))
        self.assertEqual(len(fetch(session)), 1000)
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(len(fetch(FakeSession(api_rows(1500)), limit=200)), 200)

    def test_start_date_pages_back_until_covered(self):
        session = FakeSession(api_rows(2500))
        frame = fetch(session, start_date='2026-01-15')
        self.assertGreater(len(session.calls), 1)
        self.assertEqual(frame.index[0], pd.Timestamp('2026-01-15'))
        self.assertFalse(frame.index.has_duplicates)

    def test_history_refused_past_the_public_limit_keeps_what_loaded(self):
        rows = api_rows(2500)
        session = FakeSession(rows, oldest_allowed=rows[1500][0])
        frame = fetch(session, start_date='2025-01-01')
        self.assertGreaterEqual(len(frame), 1000)

    def test_a_short_page_spends_no_second_call(self):
        session = FakeSession(api_rows(100))
        fetch(session, start_date='2020-01-01')
        self.assertEqual(len(session.calls), 1)

    def test_rate_limit_is_waited_out(self):
        session = FakeSession(api_rows(100), responses=[FakeResponse(status=429), FakeResponse(status=429)])
        self.assertFalse(fetch(session).empty)
        self.assertEqual(len(session.calls), 3)

    def test_permanent_rate_limit_and_missing_pool_report_a_reason(self):
        limited = [FakeResponse(status=429)] * (dex.RATE_LIMIT_RETRIES + 1)
        with self.assertRaises(RuntimeError):
            fetch(FakeSession([], responses=limited))
        self.assertIn('rate limit', dex.last_errors[ASSET])
        with self.assertRaises(RuntimeError):
            fetch(FakeSession([], responses=[FakeResponse(status=404)]))
        self.assertIn('does not list', dex.last_errors[ASSET])

    def test_success_clears_an_earlier_failure(self):
        dex.last_errors[ASSET] = 'old'
        fetch(FakeSession(api_rows(10)))
        self.assertNotIn(ASSET, dex.last_errors)

    def test_rate_limiter_blocks_only_past_the_window(self):
        dex._call_times.clear()
        moments, waits = [100.0], []
        for _ in range(dex.CALLS_PER_WINDOW):
            dex.wait_for_rate_limit(clock=lambda: moments[0], sleep=waits.append)
        self.assertEqual(waits, [])
        dex.wait_for_rate_limit(clock=lambda: moments[0], sleep=waits.append)
        self.assertEqual(waits, [dex.WINDOW_SECONDS])
        dex._call_times.clear()

    def test_asset_ids_translate_chains_and_keep_colons_in_tokens(self):
        self.assertEqual(dex.asset_id('ethereum', '0xP', '0xT'), 'eth:0xP:0xT')
        self.assertEqual(dex.asset_id('solana', 'P', 'T'), 'solana:P:T')
        self.assertEqual(dex.parse_asset_id('sui-network:0xP:0x2::sui::SUI'),
                         ['sui-network', '0xP', '0x2::sui::SUI'])
        with self.assertRaises(ValueError):
            dex.parse_asset_id('SPY')


def overview_frame():
    return pd.DataFrame({
        'Asset': ['PEPE/WETH', 'PEPE/WETH', 'WIF/SOL', 'USDC/SOL', 'BONK/SOL', 'NOADDR/SOL'],
        'Chain': ['ethereum', 'base', 'solana', 'solana', 'solana', 'solana'],
        'Pair Address': ['p1', 'p2', 'p3', 'p4', 'p5', ''],
        'Token Address': ['t1', 't2', 't3', 't4', 't5', 't6'],
        'Liquidity USD': [50.0, 900.0, 300.0, 9999.0, 700.0, 8000.0],
        'Volume USD': [1.0, 2.0, 30.0, 500.0, 4.0, 90.0],
    })


class BasketTests(unittest.TestCase):
    def test_ranks_by_liquidity_and_excludes_stablecoins_and_blank_addresses(self):
        basket = panel.build_basket(overview_frame(), 3)
        self.assertEqual(list(basket), ['PEPE/WETH', 'BONK/SOL', 'WIF/SOL'])
        self.assertEqual(basket['PEPE/WETH'], 'base:p2:t2')
        self.assertEqual(basket['WIF/SOL'], 'solana:p3:t3')

    def test_same_symbol_on_two_chains_gets_distinct_labels_and_ids(self):
        basket = panel.build_basket(overview_frame(), 10)
        self.assertEqual(basket['PEPE/WETH (ethereum)'], 'eth:p1:t1')
        self.assertEqual(basket['PEPE/WETH (base)'], 'base:p2:t2')

    def test_top_coins_rank_by_volume(self):
        self.assertEqual(list(panel.top_by_volume(overview_frame(), 2)), ['WIF/SOL', 'BONK/SOL'])

    def test_empty_scan_gives_an_empty_basket(self):
        self.assertEqual(panel.build_basket(pd.DataFrame(), 5), {})

    def test_basket_holds_until_refresh_or_source_change(self):
        state = {}
        first = panel.resolve_basket(overview_frame(), 2, 'a', False, state)
        changed = overview_frame().assign(**{'Liquidity USD': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        self.assertEqual(panel.resolve_basket(changed, 2, 'a', False, state), first)
        self.assertNotEqual(panel.resolve_basket(changed, 2, 'a', True, state), first)
        self.assertEqual(list(panel.resolve_basket(overview_frame(), 2, 'b', False, state))[0], 'PEPE/WETH')

    def test_failed_refresh_keeps_the_previous_basket(self):
        state = {}
        first = panel.resolve_basket(overview_frame(), 2, 'a', False, state)
        self.assertEqual(panel.resolve_basket(pd.DataFrame(), 2, 'a', True, state), first)


class FormattingTests(unittest.TestCase):
    def test_tiny_prices_keep_significant_figures(self):
        self.assertEqual(panel.format_meme_price(0.0000050980), '$0.000005098')
        self.assertEqual(panel.format_meme_price(1.234e-9), '$0.000000001234')
        self.assertEqual(panel.format_meme_price(0.1234), '$0.1234')
        self.assertEqual(panel.format_meme_price(2.5), '$2.5000')
        self.assertEqual(panel.format_meme_price(76000.5), '$76,000.50')
        self.assertEqual(panel.format_meme_price(0), '$0.00')
        self.assertEqual(panel.format_meme_price(np.nan), 'n/a')

    def test_price_changes_are_signed(self):
        self.assertEqual(panel.format_price_change(-0.000002), '-$0.000002000')
        self.assertEqual(panel.format_price_change(0.000002), '+$0.000002000')


class CalculationTests(unittest.TestCase):
    def test_daily_flow_timeframes_agree_on_the_day_total(self):
        frame = dex.candles_from_rows(api_rows(600, seconds=300), '5m', NOW)
        row = panel.daily_flow_row('X', frame, NOW)
        today = frame.loc[frame.index.normalize() == pd.Timestamp('2026-03-01')]
        total = (today['volume'] * today['close']).sum()
        for interval in panel.FLOW_FREQUENCY:
            self.assertAlmostEqual(row[f'{interval} Inflow'] + row[f'{interval} Outflow'], total, places=4)

    def test_daily_flow_excludes_yesterday(self):
        index = pd.date_range('2026-02-28 23:50', periods=6, freq='5min', name='date')
        frame = pd.DataFrame({'open': 1.0, 'close': 2.0, 'high': 2.0, 'low': 1.0, 'volume': 1.0}, index=index)
        row = panel.daily_flow_row('X', frame, NOW)
        self.assertEqual(row['5m Inflow'], 8.0)  # Four bars after midnight, at $2 each.

    def test_gex_proxy_keeps_its_shape_at_tiny_prices(self):
        def proxy(price, volatility):
            # The dashboard proxy's width floor, which is what scaling must avoid.
            strikes = np.linspace(price * 0.8, price * 1.2, 21)
            scale = max(price * volatility * 0.75, 1.0)
            return pd.Series(np.exp(-((strikes - price) / scale) ** 2) * price, index=strikes)

        normal = panel.scaled_gex_proxy(proxy, 1000.0, 0.03)
        tiny = panel.scaled_gex_proxy(proxy, 2e-6, 0.03)
        self.assertAlmostEqual(tiny.index[10], 2e-6)
        np.testing.assert_allclose(tiny.to_numpy() / 2e-6, normal.to_numpy() / 1000.0)
        self.assertGreater(tiny.to_numpy().std() / tiny.to_numpy().mean(), 0.1)

    def test_engine_wording_never_reaches_the_screen(self):
        text = panel.relabel_text('Positive QQQ/NQ and negative VIX daily alerts', 'WIF/SOL', 'BONK/SOL')
        self.assertEqual(text, 'Positive WIF/SOL/BONK/SOL and negative BTCDOM daily alerts')
        self.assertIn('Refresh Meme Overview', panel.relabel_text('Refresh Indices Data', 'A', 'B'))

    def test_alert_mapping_uses_the_chosen_coins_and_drops_others(self):
        state = {
            panel.ALERTS_KEY: [{'Symbol': 'A'}, {'Symbol': 'OTHER'}, {'Symbol': 'B'}, {'Symbol': panel.GAUGE}],
            panel.ALERT_TF_KEY: '1h', panel.ALERT_THRESHOLD_KEY: 1.0,
        }
        rows = panel.pair_alert_inputs(state, 'A', 'B')['daily_alerts']
        self.assertEqual([row['Symbol'] for row in rows], ['QQQ', 'NQ=F', '^VIX'])

    def test_pair_signals_fail_closed_without_daily_alerts(self):
        history = {}
        for name, seed in [('A', 1), ('B', 2)]:
            rng = np.random.default_rng(seed)
            index = pd.date_range('2026-01-01', periods=300, freq='1h', name='date')
            close = 1e-6 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
            history[name] = pd.DataFrame({
                'open': close * (1 + rng.normal(0, 0.01, 300)), 'close': close, 'high': close * 1.02,
                'low': close * 0.98, 'volume': rng.uniform(1e9, 2e9, 300),
                'momentum': pd.Series(close).pct_change(30).to_numpy(),
                'atr14': close * 0.02, 'z_score': rng.normal(0, 1, 300),
            }, index=index)
        inputs = {'daily_alerts': [], 'alert_timeframe': '1h', 'alert_threshold': 1.0}
        signals = panel.build_pair_signals(
            history['A'], history['B'], 'A', 'B', '1h', 1.0, pd.Timestamp('2026-02-01', tz='UTC'), inputs,
        )
        self.assertIn('A Score', signals.columns)
        self.assertIn('BTCDOM Daily Z', signals.columns)
        self.assertFalse(signals['Signal'].isin(['BUY', 'SELL']).any())
        self.assertFalse(signals['Reason'].str.contains(r'\bQQQ\b|\bNQ\b|\bVIX\b').any())

    def test_load_candles_rejects_young_pools_and_reports_why(self):
        young = pd.DataFrame({'close': [1.0] * 10})
        deps = SimpleNamespace(fetch_candles=lambda asset, **_: young, fetch_gauge=None)
        frame, reason = panel.load_candles(deps, ASSET, '1h')
        self.assertIsNone(frame)
        self.assertIn('only 10 1h candles', reason)
        dex.last_errors[ASSET] = 'RuntimeError: rate limit'
        deps = SimpleNamespace(fetch_candles=lambda asset, **_: None, fetch_gauge=None)
        self.assertIn('rate limit', panel.load_candles(deps, ASSET, '1h')[1])
        dex.last_errors.pop(ASSET)

    def test_gauge_comes_from_binance_and_coins_from_the_dex(self):
        sources = []
        frame = pd.DataFrame({'close': [1.0] * 100})

        def source(name):
            def fetch_from(asset, **_):
                sources.append(name)
                return frame
            return fetch_from

        deps = SimpleNamespace(fetch_candles=source('dex'), fetch_gauge=source('binance'))
        panel.load_candles(deps, panel.GAUGE, '1h')
        panel.load_candles(deps, ASSET, '1h')
        self.assertEqual(sources, ['binance', 'dex'])


def run_tab():
    """Render every section against synthetic indicator candles at meme-coin prices."""
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    import meme_coin_panel as panel
    import tests.test_meme_coin_panel as fixtures
    from tests.test_meme_coin_panel import overview_frame, fake_zone_battle, fake_zones

    def indicator_candles(*args, **kwargs):
        return fixtures.indicator_candles(*args, **kwargs)

    deps = SimpleNamespace(
        fetch_candles=lambda asset, timeframe='1h', **_: indicator_candles(timeframe, seed=len(asset)),
        fetch_gauge=lambda asset, timeframe='1h', **_: indicator_candles(timeframe, seed=3, price=1000.0),
        calculate_liquidity_zones=fake_zones,
        calculate_zone_battle_scores=fake_zone_battle,
        build_daily_alert_history=lambda *a, **k: [],
        build_gex_proxy=lambda price, volatility: pd.Series(
            np.linspace(1, -1, 21) * price, index=np.linspace(price * 0.8, price * 1.2, 21),
        ),
        summarize_gex=lambda profile, price: {
            'total_gex': float(profile.sum()), 'zero_gamma_level': float(profile.index[10]),
            'gex_profile': profile, 'current_price': price,
        },
        format_large_number=lambda value: f'${value:,.0f}',
        color_metrics=lambda value: '',
        send_telegram_alert=lambda *a: None,
        tg_token='', tg_chat_id='',
    )
    panel.render_meme_sections(deps, overview_frame(), source_key='test')


def indicator_candles(timeframe='1h', seed=1, price=2e-6, count=400):
    rng = np.random.default_rng(seed)
    frequency = dex.BAR_FREQUENCY[timeframe]
    index = pd.date_range(end=pd.Timestamp.now(tz='UTC').tz_localize(None).floor(frequency),
                          periods=count, freq=frequency, name='date')
    close = price * np.exp(np.cumsum(rng.normal(0, 0.02, count)))
    frame = pd.DataFrame({
        'open': close * (1 + rng.normal(0, 0.01, count)), 'close': close, 'high': close * 1.03,
        'low': close * 0.97, 'volume': rng.uniform(1e8, 2e8, count),
    }, index=index)
    frame['momentum'] = frame['close'].pct_change(30)
    frame['z_score'] = rng.normal(0, 1, count)
    frame['vol_ma'] = frame['volume'].rolling(20).mean()
    frame['cmf'] = rng.normal(0, 0.2, count)
    frame['rsi'] = 55.0
    frame['ema_50'] = frame['close'].ewm(span=50).mean()
    frame['atr14'] = close * 0.02
    frame['ob_bull'] = close * 0.9
    frame['bb_lower'] = close * 0.95
    frame['sma_200'] = frame['close'].rolling(200).mean()
    return frame


def fake_zone_battle(candles, observation_bars=10, max_zones=50):
    close = float(candles['close'].iloc[-1])
    return pd.DataFrame([{
        'Zone Formed': candles.index[-30], 'Zone Entry': candles.index[-20], 'Zone Type': 'Bullish',
        'Zone Bottom': close * 0.9, 'Zone Top': close * 1.1, 'Entry Price': close, 'Reaction Close': close,
        'Bars Observed': 10, 'Reaction Rz': 1.0, 'Flow Fz': 1.0, 'Momentum Mz': 1.0, 'Acceptance Az': 1.0,
        'Battle Score': 1.5, 'Result': 'Strong Buyer Victory',
    }]), candles


def fake_zones(candles, price_bins=24, top_zones=8):
    close = float(candles['close'].iloc[-1])
    return pd.DataFrame([{
        'Zone Bottom': close * 0.9, 'Zone Top': close * 1.1, 'Zone Midpoint': close, 'Liquidity Score': 90.0,
        'Location vs Price': 'At price', 'Flow Bias': 'Buying', 'Order Flow Imbalance': 0.2,
        'Volume Z-Score': 1.0, 'ATR Compression Z': 1.0, 'Swing Density Z': 1.0,
        'Time at Price Z': 1.0, 'VWAP Proximity Z': 1.0,
    }]), candles


class RenderTests(unittest.TestCase):
    def app(self):
        at = AppTest.from_function(run_tab, default_timeout=60)
        at.run()
        self.assertFalse(at.exception, at.exception)
        return at

    def test_nothing_loads_until_a_button_is_pressed(self):
        with mock.patch(f'{__name__}.indicator_candles', wraps=indicator_candles) as candles:
            at = self.app()
        candles.assert_not_called()
        self.assertTrue(any('Refresh Meme Overview' in info.value for info in at.info))

    def test_signals_and_backtest_run_end_to_end_at_meme_prices(self):
        at = self.app()
        # One BUY-side daily alert set per Ghana day lets base BUY signals through the filter.
        basket = list(panel.build_basket(overview_frame(), 5))
        alerts = []
        for day in pd.date_range(end=pd.Timestamp.now(tz='UTC').normalize(), periods=20, freq='1D'):
            for symbol, side, z in [(basket[0], 'POSITIVE Z-SCORE', 1.5), (basket[1], 'POSITIVE Z-SCORE', 1.5),
                                    (panel.GAUGE, 'NEGATIVE Z-SCORE', -1.5)]:
                alerts.append({'Ghana Date': day.date(), 'Ghana Time': day, 'Symbol': symbol, 'Alert': side,
                               'Price': 1.0, 'Z-Score': z, 'Momentum Z': z, 'Timeframe': '1h'})
        at.session_state[panel.ALERTS_KEY] = alerts
        at.session_state[panel.ALERT_TF_KEY] = '1h'
        at.session_state[panel.ALERT_THRESHOLD_KEY] = 1.0
        at.selectbox(key='mc_bt_target').select('Fixed reward / risk').run()
        at.button(key='mc_pair_generate').click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(any('Signal Time (Ghana)' in frame.value.columns for frame in at.dataframe))
        at.button(key='mc_bt_run').click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual([error.value for error in at.error], [])
        result = at.session_state['mc_bt_result']['result']
        self.assertGreater(result['summary']['Trades'], 0)
        self.assertLess(result['trades']['Entry Price'].max(), 1e-3)
        self.assertLess(result['trades']['Cost Points'].max(), 1e-4)

    def test_every_section_button_renders_without_error(self):
        at = self.app()
        for key in ['mc_overview_refresh', 'mc_gex_run', 'mc_flow_refresh', 'mc_daily_flow_refresh',
                    'mc_qs_refresh', 'mc_z_run', 'mc_pair_generate', 'mc_zone_run',
                    'mc_liquidity_run', 'mc_delta_refresh', 'mc_support_refresh', 'mc_top_refresh']:
            at.button(key=key).click().run()
            self.assertFalse(at.exception, f'{key}: {at.exception}')
        rendered = ' '.join(frame.value.to_string() for frame in at.dataframe)
        self.assertNotRegex(rendered, r'\bQQQ\b|\bNQ\b|\bVIX\b')

    def test_tiny_prices_never_display_as_zero(self):
        at = self.app()
        at.button(key='mc_support_refresh').click().run()
        self.assertFalse(at.exception, at.exception)
        table = next(frame.value for frame in at.dataframe if 'Current Price' in frame.value.columns)
        self.assertTrue(table['Current Price'].between(1e-9, 1e-3).all())


if __name__ == '__main__':
    unittest.main()
