import unittest
from contextlib import contextmanager

import dashboard


class FakeTab:
    def __init__(self, is_open):
        self.open, self.entered = is_open, 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        return False


@contextmanager
def lazy(value):
    old = dashboard.LAZY_TABS
    dashboard.LAZY_TABS = value
    try:
        yield
    finally:
        dashboard.LAZY_TABS = old


class LazyTabTests(unittest.TestCase):
    def run_body(self, tab):
        ran = 0
        for _ in dashboard.tab_body(tab):
            ran += 1
        return ran

    def test_online_only_the_open_tab_runs(self):
        with lazy(True):
            shown, hidden = FakeTab(True), FakeTab(False)
            self.assertEqual((self.run_body(shown), shown.entered), (1, 1))
            self.assertEqual((self.run_body(hidden), hidden.entered), (0, 0))

    def test_on_a_pc_every_tab_runs(self):
        with lazy(False):
            for tab in (FakeTab(True), FakeTab(False), FakeTab(None)):
                self.assertEqual((self.run_body(tab), tab.entered), (1, 1))

    def test_on_a_pc_tabs_are_not_rerun_widgets(self):
        # Locally the tabs must keep their old behaviour, so st.tabs gets no extra options.
        if not dashboard.LAZY_TABS:
            self.assertEqual(dashboard.MAIN_TAB_OPTIONS, {})


if __name__ == '__main__':
    unittest.main()
