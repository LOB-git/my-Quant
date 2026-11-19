import unittest

import numpy as np
import pandas as pd

import alert_quant_verdict as aq
from historical_flow_analysis import analyze_ghana_flow_events
from tests.test_quant_edge import ohlc_from_close

SETTINGS = {'min_score': 30.0, 'confirm': True, 'min_q': 0.10, 'max_hold': 48, 'atr_multiple': 1.5,
            'reward_risk': 2.0, 'fee_bps': 10.0, 'equity': 1000.0, 'risk_pct': 1.0}


def history(seed=3, bars=1500):
    rng = np.random.default_rng(seed)
    frame = ohlc_from_close(100 * np.exp(np.cumsum(rng.normal(0, 0.01, bars))))
    frame['volume'] = rng.uniform(5, 15, bars)
    frame['momentum'] = frame['close'].pct_change(30).fillna(0)
    return frame


class AlertVerdictTests(unittest.TestCase):
    def setUp(self):
        self.history = history()
        self.day = pd.Timestamp(self.history.index[-30]).date()
        _, _, self.alerts = analyze_ghana_flow_events(self.history, 'X/USDT', self.day)

    def test_scores_match_the_alert_table(self):
        _, score = aq.scored_history(self.history)
        times = pd.to_datetime(self.alerts['Ghana Time'], utc=True)
        np.testing.assert_allclose(score.loc[times].to_numpy(), self.alerts['Direction Score'].to_numpy())

    def test_every_alert_gets_a_verdict_and_trades_follow_the_score(self):
        table, summary = aq.evaluate_alerts(self.history, self.alerts, self.day, SETTINGS)
        self.assertEqual(len(table), len(self.alerts))
        self.assertTrue(set(table['Verdict']) <= {'BUY', 'SELL', 'WEAK', 'CONFLICT', 'UNCONFIRMED', 'NO EDGE'})
        for row in table.to_dict('records'):
            if row['Verdict'] == 'BUY':
                self.assertGreater(row['Direction Score'], 0)
                self.assertLess(row['Stop'], row['Entry'])
            if row['Verdict'] == 'SELL':
                self.assertLess(row['Direction Score'], 0)
                self.assertGreater(row['Stop'], row['Entry'])
            if row['Verdict'] in ('BUY', 'SELL'):
                self.assertLessEqual(row['$ Risk'], 10.0 + 1e-6)
            else:
                self.assertEqual(row['Size (units)'], 0.0)
        self.assertEqual(summary['train_bars'],
                         int(np.argmax(self.history.index.date == self.day)))

    def test_the_selected_day_never_judges_itself(self):
        _, before = aq.evaluate_alerts(self.history, self.alerts, self.day, SETTINGS)
        tampered = self.history.copy()
        later = tampered.index.date >= self.day
        tampered.loc[later, ['open', 'high', 'low', 'close']] *= 3  # Rewrite the day and after.
        _, after = aq.evaluate_alerts(tampered, self.alerts, self.day, SETTINGS)
        self.assertEqual(before['raw'], after['raw'])
        self.assertEqual(before['confirmed'], after['confirmed'])
        self.assertEqual(before['hurst'], after['hurst'])

    def test_too_little_history_is_refused(self):
        day = pd.Timestamp(self.history.index[100]).date()
        _, _, alerts = analyze_ghana_flow_events(self.history, 'X/USDT', day)
        table, summary = aq.evaluate_alerts(self.history, alerts, day, SETTINGS)
        self.assertTrue(table.empty)
        self.assertIn('needed', summary['error'])

    def test_trade_outcome_hits_target_stop_and_stays_open(self):
        frame = pd.DataFrame({'close': [100, 100, 100.0], 'high': [100, 103.5, 100.0],
                              'low': [100, 99.5, 100.0], 'atr14': 1.0})
        self.assertEqual(aq.trade_outcome(frame, 0, 1, 1.5, 2.0, 10, 0), (2.0, 'Target hit'))
        self.assertEqual(aq.trade_outcome(frame, 0, -1, 1.5, 2.0, 10, 0), (-1.0, 'Stop hit'))
        self.assertEqual(aq.trade_outcome(frame, 2, 1, 1.5, 2.0, 10, 0)[1], 'Still open')

    def test_confirmation_only_removes_trades(self):
        _, score = aq.scored_history(self.history)
        q = pd.Series(np.random.default_rng(0).uniform(-1, 1, len(score)), index=score.index)
        raw = aq.alert_direction(score, q, 30, 0.1, confirm=False)
        confirmed = aq.alert_direction(score, q, 30, 0.1, confirm=True)
        self.assertTrue(confirmed.notna().sum() < raw.notna().sum())
        both = confirmed.notna()
        self.assertTrue((confirmed[both] == raw[both]).all())


if __name__ == '__main__':
    unittest.main()
