import unittest

import numpy as np
import pandas as pd

import derivative_crypto_panel as panel


def scan_table():
    """A small scan table: one row per filter it is meant to fail, plus two matches."""
    rows = [
        # symbol,        verdict,   regime,           $ risk,     leverage, BT expectancy
        ('MATCH_A/USDT', 'LONG', 'Mean-Reverting', 10.0, 0.25, 0.10),
        ('MATCH_B/USDT', 'SHORT', 'Mean-Reverting', 9.9999999, 0.39, 0.30),
        ('FLAT/USDT', 'FLAT', 'Mean-Reverting', 0.0, 0.0, 0.50),
        ('NOEDGE/USDT', 'NO EDGE', 'Mean-Reverting', 0.0, 0.0, -0.10),
        ('TREND/USDT', 'LONG', 'Trending', 10.0, 0.20, 0.40),
        ('RANDOM/USDT', 'SHORT', 'Random Walk', 10.0, 0.20, 0.40),
        ('KELLY/USDT', 'LONG', 'Mean-Reverting', 7.42, 0.20, 0.40),   # half-Kelly risked less
        ('CENT/USDT', 'LONG', 'Mean-Reverting', 10.01, 0.20, 0.40),   # one cent off
        ('LEV_EQ/USDT', 'SHORT', 'Mean-Reverting', 10.0, 0.40, 0.40),  # not strictly below
        ('LEV_HI/USDT', 'LONG', 'Mean-Reverting', 10.0, 1.20, 0.40),
        ('NAN/USDT', 'LONG', 'Mean-Reverting', np.nan, np.nan, 0.40),
    ]
    return pd.DataFrame(rows, columns=['Symbol', 'Verdict', 'Regime', '$ Risk', 'Leverage', 'BT Expectancy R'])


class RiskLeverageWatchlistTests(unittest.TestCase):
    def test_keeps_only_rows_passing_all_four_filters(self):
        matches, _ = panel.risk_leverage_funnel(scan_table(), risk_usd=10.0, max_leverage=0.4)
        self.assertEqual(set(matches['Symbol']), {'MATCH_A/USDT', 'MATCH_B/USDT'})

    def test_ranked_by_backtest_expectancy(self):
        matches, _ = panel.risk_leverage_funnel(scan_table())
        self.assertEqual(list(matches['Symbol']), ['MATCH_B/USDT', 'MATCH_A/USDT'])

    def test_funnel_counts_each_step_in_order(self):
        _, funnel = panel.risk_leverage_funnel(scan_table(), risk_usd=10.0, max_leverage=0.4)
        self.assertEqual([count for _, count in funnel], [9, 7, 4, 2])
        self.assertEqual(funnel[2][0], '$ Risk = $10.00')
        self.assertEqual(funnel[3][0], 'Leverage < 0.40x')

    def test_leverage_limit_is_adjustable(self):
        matches, _ = panel.risk_leverage_funnel(scan_table(), max_leverage=2.0)
        self.assertEqual(set(matches['Symbol']), {'MATCH_A/USDT', 'MATCH_B/USDT', 'LEV_EQ/USDT', 'LEV_HI/USDT'})
        matches, _ = panel.risk_leverage_funnel(scan_table(), max_leverage=0.3)
        self.assertEqual(set(matches['Symbol']), {'MATCH_A/USDT'})

    def test_risk_amount_is_adjustable(self):
        matches, _ = panel.risk_leverage_funnel(scan_table(), risk_usd=7.42, max_leverage=0.4)
        self.assertEqual(list(matches['Symbol']), ['KELLY/USDT'])

    def test_missing_columns_or_empty_table_return_nothing(self):
        for table in (pd.DataFrame(), scan_table().drop(columns='Leverage')):
            matches, funnel = panel.risk_leverage_funnel(table)
            self.assertTrue(matches.empty)
            self.assertEqual(funnel, [])

    def test_real_quant_rows_carry_the_filtered_columns(self):
        """quant_edge_row must keep producing the columns the watch list reads."""
        from quant_edge import quant_edge_row
        index = pd.date_range('2026-01-01', periods=400, freq='1h')
        close = 100 + np.sin(np.arange(400) / 4) * 3
        candles = pd.DataFrame({
            'open': close, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': 10.0,
            'z_score': np.sin(np.arange(400) / 4) * 2, 'cmf': 0.0, 'atr14': 2.0,
        }, index=index)
        settings = dict(equity=1000.0, risk_pct=1.0, atr_multiple=1.5, reward_risk=2.0,
                        threshold=0.25, max_hold=48, fee_bps=10.0)
        row = quant_edge_row('TEST/USDT', candles, 24, settings)
        for column in ('Verdict', 'Regime', '$ Risk', 'Leverage'):
            self.assertIn(column, row)


class WatchlistFlowDropdownTests(unittest.TestCase):
    def test_every_watch_list_asset_is_selectable_once_in_table_order(self):
        matches, _ = panel.risk_leverage_funnel(scan_table())
        doubled = pd.concat([matches, matches.head(1)])
        options = panel.watchlist_flow_options(doubled)
        self.assertEqual(list(options.values()), list(matches['Symbol']))
        for label, symbol in options.items():
            verdict = matches.loc[matches['Symbol'] == symbol, 'Verdict'].iloc[0]
            self.assertEqual(label, f'{symbol} — {verdict}')
        self.assertEqual(panel.watchlist_flow_options(matches.iloc[0:0]), {})
        self.assertEqual(panel.watchlist_flow_options(None), {})

    def test_dropdown_renders_real_flow_for_the_selected_asset(self):
        """Run the watch list in a real Streamlit script with taker klines faked: the chosen asset is loaded."""
        from streamlit.testing.v1 import AppTest

        def app():
            import numpy as np
            import pandas as pd
            import derivative_crypto_panel as panel
            import real_flow_panel
            from tests.test_real_flow import rows
            import real_flow as rf
            asked = []

            def fake_load(symbol, interval, limit=500):
                asked.append((symbol, interval))
                return rf.klines_frame(rows(400))
            real_flow_panel.load = fake_load
            matches = pd.DataFrame({'Symbol': ['AAA/USDT', 'BBB/USDT'], 'Verdict': ['LONG', 'SHORT'],
                                    'Entry': [1.0, 2.0], 'Stop': [0.9, 2.2], 'Target': [1.2, 1.6],
                                    '$ Risk': [10.0, 10.0], 'Leverage': [0.2, 0.3]})
            panel.render_watchlist_flow(matches, 'test', '15m')
            import streamlit as st
            st.text(repr(asked))

        at = AppTest.from_function(app, default_timeout=60).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.selectbox[0].options, ['AAA/USDT — LONG', 'BBB/USDT — SHORT'])
        self.assertEqual(at.radio[0].value, '15m')
        # 15m chart: the 5m net-flow window is read from 5m candles.
        self.assertEqual(at.text[-1].value, "[('AAA/USDT', '15m'), ('AAA/USDT', '5m')]")
        at.selectbox[0].select('BBB/USDT — SHORT').run()
        self.assertFalse(at.exception)
        self.assertEqual(at.text[-1].value, "[('BBB/USDT', '15m'), ('BBB/USDT', '5m')]")
        self.assertTrue(any('Watch-list plan for BBB/USDT' in str(c.value) for c in at.caption))


if __name__ == '__main__':
    unittest.main()
