import unittest

import numpy as np
import pandas as pd

import forex_quant_scan_panel as fx
import quant_edge as qe
from tests.test_quant_edge import SETTINGS, ohlc_from_close


def flat(price, count=5):
    return pd.DataFrame({'close': [price] * count})


class ForexConversionTests(unittest.TestCase):
    def test_quote_currencies_convert_to_usd(self):
        rates = fx.usd_per_quote({
            'EURUSD=X': flat(1.10), 'USDJPY=X': flat(160.0), 'EURJPY=X': flat(176.0),
            'GBPUSD=X': flat(1.25), 'EURGBP=X': flat(0.88),
        })
        self.assertEqual(rates['EURUSD=X'], 1.0)
        self.assertAlmostEqual(rates['USDJPY=X'], 1 / 160)
        self.assertAlmostEqual(rates['EURJPY=X'], 1 / 160)  # Uses the USDJPY rate.
        self.assertAlmostEqual(rates['EURGBP=X'], 1.25)     # Uses GBPUSD.

    def test_missing_conversion_is_nan_not_guessed(self):
        self.assertTrue(np.isnan(fx.usd_per_quote({'EURJPY=X': flat(176.0)})['EURJPY=X']))

    def test_helper_pairs_are_added_for_crosses(self):
        needed = fx.required_symbols(['EURJPY=X', 'EURGBP=X', 'USDCAD=X'])
        self.assertIn('USDJPY=X', needed)
        self.assertIn('GBPUSD=X', needed)
        self.assertEqual(needed[:3], ['EURJPY=X', 'EURGBP=X', 'USDCAD=X'])

    def test_jpy_pair_risks_the_right_dollar_amount(self):
        rng = np.random.default_rng(12)  # A series whose rule passes the gate, so a trade is sized.
        candles = ohlc_from_close(160 * np.exp(np.cumsum(rng.normal(0, 0.001, 900))))
        q = 1 / float(candles['close'].iloc[-1])
        row = qe.quant_edge_row('USDJPY=X', candles, None, {**SETTINGS, 'threshold': 0.0}, q)
        self.assertIn(row['Verdict'], ('LONG', 'SHORT'))
        self.assertAlmostEqual(row['$ Risk'], SETTINGS['equity'] * SETTINGS['risk_pct'] / 100)
        self.assertAlmostEqual(row['$ Risk'], row['Size (units)'] * abs(row['Entry'] - row['Stop']) * q)
        self.assertLessEqual(row['Leverage'], qe.MAX_LEVERAGE + 1e-9)

    def test_no_rate_blocks_the_trade(self):
        rng = np.random.default_rng(4)
        candles = ohlc_from_close(176 * np.exp(np.cumsum(rng.normal(0, 0.001, 900))))
        row = qe.quant_edge_row('EURJPY=X', candles, None, {**SETTINGS, 'threshold': 0.0}, np.nan)
        self.assertNotIn(row['Verdict'], ('LONG', 'SHORT'))
        self.assertEqual(row['Size (units)'], 0.0)


class ForexDisplayTests(unittest.TestCase):
    def test_pips_lots_and_pip_value(self):
        table = pd.DataFrame({
            'Symbol': ['EURUSD=X', 'USDJPY=X', 'XAUUSD'], 'Entry': [1.1000, 160.00, 4000.0],
            'Stop': [1.0980, 160.50, 3990.0], 'Size (units)': [50_000.0, 100_000.0, 10.0],
            'Quote to Account': [1.0, 1 / 160, 1.0],
        })
        result = fx.add_fx_columns(table).set_index('Symbol')
        self.assertAlmostEqual(result.loc['EURUSD', 'Stop (pips)'], 20.0)
        self.assertAlmostEqual(result.loc['USDJPY', 'Stop (pips)'], 50.0)
        self.assertAlmostEqual(result.loc['EURUSD', 'Lots'], 0.5)
        self.assertAlmostEqual(result.loc['EURUSD', 'Micro Lots'], 50.0)
        self.assertAlmostEqual(result.loc['EURUSD', 'Pip $ / Lot'], 10.0)
        self.assertAlmostEqual(result.loc['USDJPY', 'Pip $ / Lot'], 1000 / 160)
        self.assertAlmostEqual(result.loc['XAUUSD', 'Lots'], 0.1)
        self.assertAlmostEqual(result.loc['XAUUSD', 'Stop (pips)'], 100.0)

    def test_usd_exposure_reads_pair_direction(self):
        table = pd.DataFrame({'Symbol': ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'EURGBP=X'],
                              'Verdict': ['LONG', 'LONG', 'SHORT', 'LONG']})
        # Long EURUSD sells USD; long USDJPY buys USD; short GBPUSD buys USD; EURGBP has no USD.
        self.assertEqual(fx.usd_exposure(table), (2, 1))


if __name__ == '__main__':
    unittest.main()
