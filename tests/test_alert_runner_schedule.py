import unittest
from unittest import mock

import alert_runner


def config():
    return {'interval_seconds': 300, 'groups': {
        'derivatives': {'enabled': True},
        'watchlist': {'enabled': True, 'every_seconds': 3600},
        'us_indices': {'enabled': False},
    }}


class RunCycleScheduleTests(unittest.TestCase):
    def run_at(self, seconds, last_run, token='t', chat='c'):
        calls = []
        with mock.patch.object(alert_runner, 'check_group', lambda name, *a: calls.append(name) or 0), \
                mock.patch.object(alert_runner, 'save_watchlist', lambda *a, **k: calls.append('watchlist')), \
                mock.patch.object(alert_runner, 'save_state', lambda state: None), \
                mock.patch.object(alert_runner.time, 'monotonic', lambda: seconds):
            alert_runner.run_cycle(config(), None, None, token, chat, {}, last_run)
        return calls

    def test_watchlist_runs_hourly_while_alerts_run_every_cycle(self):
        last_run = {}
        self.assertEqual(self.run_at(0, last_run), ['derivatives', 'watchlist'])
        self.assertEqual(self.run_at(300, last_run), ['derivatives'])
        self.assertEqual(self.run_at(3599, last_run), ['derivatives'])
        self.assertEqual(self.run_at(3600, last_run), ['derivatives', 'watchlist'])

    def test_watchlist_runs_without_telegram_details(self):
        self.assertEqual(self.run_at(0, {}, token='', chat=''), ['watchlist'])


if __name__ == '__main__':
    unittest.main()
