import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import zone_progress_panel as zpp
from dashboard import calculate_zone_battle_scores

OBS = 10


def candles(bars=600, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, bars)))
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.date_range('2026-11-10 00:00', periods=bars, freq='5min')  # UTC-naive like fetch_and_analyze
    df = pd.DataFrame({'open': open_, 'close': close, 'high': np.maximum(open_, close) * 1.0005,
                       'low': np.minimum(open_, close) * 0.9995, 'volume': rng.uniform(50, 150, bars)}, index=idx)
    df['momentum'] = df['close'].pct_change(30)
    return df


def after_close(df):
    """A moment when every candle in df has closed."""
    return pd.Timestamp(df.index[-1], tz='UTC') + pd.Timedelta(minutes=10)


class InProgressTests(unittest.TestCase):
    def setUp(self):
        self.df = candles()
        self.full, _ = calculate_zone_battle_scores(self.df, OBS, max_zones=10 ** 6, fvg_candles=3,
                                                    now=after_close(self.df))

    def cut_inside_a_battle(self):
        """Cut the data 4 closed candles into a known complete battle."""
        battle = self.full.sort_values('Zone Entry').iloc[len(self.full) // 2]
        entry = self.df.index.get_loc(battle['Zone Entry'])
        return battle, entry, self.df.iloc[:entry + 4]

    def test_unfinished_battle_is_in_progress_with_a_countdown(self):
        battle, entry, cut = self.cut_inside_a_battle()
        done, _, progress = calculate_zone_battle_scores(cut, OBS, max_zones=10 ** 6, fvg_candles=3,
                                                         include_in_progress=True, now=after_close(cut))
        self.assertNotIn(battle['Zone Entry'], set(done['Zone Entry']))
        row = progress[progress['Zone Entry'] == battle['Zone Entry']].iloc[0]
        self.assertEqual((row['Candles Closed'], row['Candles Left'], row['Candles Needed']), (4, 6, OBS))
        self.assertFalse(row['Includes Live Candle'])
        self.assertEqual(row['Verdict Due'], pd.Timestamp(battle['Zone Entry'], tz='UTC') + pd.Timedelta(minutes=5 * OBS))

    def test_running_score_uses_the_battle_formula_on_the_candles_so_far(self):
        battle, entry, cut = self.cut_inside_a_battle()
        _, data, progress = calculate_zone_battle_scores(cut, OBS, max_zones=10 ** 6, fvg_candles=3,
                                                         include_in_progress=True, now=after_close(cut))
        row = progress[progress['Zone Entry'] == battle['Zone Entry']].iloc[0]
        reaction = data.iloc[entry:]
        mid = (row['Zone Bottom'] + row['Zone Top']) / 2
        rz = (reaction['close'].iloc[-1] - mid) / data['ATR15'].iloc[entry]
        az = np.mean(np.where(reaction['close'] > row['Zone Top'], 1.0,
                              np.where(reaction['close'] < row['Zone Bottom'], -1.0, 0.0)))
        fz = reaction['Flow Z'].iloc[-1] if pd.notna(reaction['Flow Z'].iloc[-1]) else 0.0
        mz = reaction['Momentum Z'].iloc[-1] if pd.notna(reaction['Momentum Z'].iloc[-1]) else 0.0
        self.assertAlmostEqual(row['Running Score'], 0.4 * rz + 0.3 * fz + 0.2 * mz + 0.1 * az)
        # Once all candles are in, the in-progress zone becomes the finished battle with its final score.
        self.assertNotAlmostEqual(row['Running Score'], battle['Battle Score'], places=6)

    def test_a_forming_last_candle_does_not_complete_a_battle(self):
        battle = self.full.sort_values('Zone Entry').iloc[len(self.full) // 2]
        end = self.df.index.get_loc(battle['Verdict Time'])
        cut = self.df.iloc[:end + 1]                      # the battle's last candle is the newest one
        closed_now = after_close(cut)
        forming_now = pd.Timestamp(cut.index[-1], tz='UTC') + pd.Timedelta(minutes=2)
        done, _, _ = calculate_zone_battle_scores(cut, OBS, max_zones=10 ** 6, fvg_candles=3,
                                                  include_in_progress=True, now=closed_now)
        self.assertIn(battle['Zone Entry'], set(done['Zone Entry']))
        done, _, progress = calculate_zone_battle_scores(cut, OBS, max_zones=10 ** 6, fvg_candles=3,
                                                         include_in_progress=True, now=forming_now)
        self.assertNotIn(battle['Zone Entry'], set(done['Zone Entry']))
        row = progress[progress['Zone Entry'] == battle['Zone Entry']].iloc[0]
        self.assertEqual((row['Candles Closed'], row['Candles Left']), (OBS - 1, 1))
        self.assertTrue(row['Includes Live Candle'])

    def test_default_call_is_unchanged(self):
        out = calculate_zone_battle_scores(self.df, OBS, max_zones=10 ** 6, fvg_candles=3, now=after_close(self.df))
        self.assertEqual(len(out), 2)
        pd.testing.assert_frame_equal(out[0], self.full)


class PanelTests(unittest.TestCase):
    def test_due_in(self):
        now = pd.Timestamp('2026-11-14 10:00', tz='UTC')
        self.assertEqual(zpp.due_in(pd.Timestamp('2026-11-14 10:25', tz='UTC'), now), '25m')
        self.assertEqual(zpp.due_in(pd.Timestamp('2026-11-14 11:05', tz='UTC'), now), '1h 05m')
        self.assertEqual(zpp.due_in(pd.Timestamp('2026-11-14 09:59', tz='UTC'), now), 'due now')
        self.assertEqual(zpp.due_in(pd.NaT, now), '—')

    def test_markers_and_table(self):
        df = candles()
        cut = df.iloc[:-3]
        _, _, progress = calculate_zone_battle_scores(cut, OBS, max_zones=10 ** 6, fvg_candles=3,
                                                      include_in_progress=True, now=after_close(cut))
        self.assertFalse(progress.empty)
        fig = zpp.add_in_progress_markers(go.Figure(), progress)
        self.assertEqual(fig.data[0].name, 'In progress (provisional)')
        self.assertEqual(len(fig.data[0].x), len(progress))
        table = zpp.progress_table(progress, now=after_close(cut))
        self.assertEqual(list(table['Progress']), [f'{c} / {OBS}' for c in progress['Candles Closed']])
        self.assertIs(zpp.add_in_progress_markers(fig, pd.DataFrame()), fig)


if __name__ == '__main__':
    unittest.main()
