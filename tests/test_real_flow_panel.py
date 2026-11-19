import unittest
from unittest import mock

import real_flow_panel as rfp


class ScanTests(unittest.TestCase):
    def test_scan_with_no_data_returns_an_empty_table(self):
        # Binance answers 451 on Streamlit Cloud, so every row fails; sorting the empty frame used to raise KeyError.
        with mock.patch.object(rfp, 'scan_row', return_value=None):
            self.assertTrue(rfp.scan(['BTC/USDT', 'ETH/USDT'], workers=2).empty)

    def test_scan_sorts_by_24h_net_flow(self):
        rows = {'A': {'Symbol': 'A', 'Net Flow 24h': 1.0}, 'B': {'Symbol': 'B', 'Net Flow 24h': 5.0}}
        with mock.patch.object(rfp, 'scan_row', side_effect=rows.get):
            self.assertEqual(list(rfp.scan(['A', 'B'], workers=1)['Symbol']), ['B', 'A'])


if __name__ == '__main__':
    unittest.main()
