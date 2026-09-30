import numpy as np

from src.audio import benchmark as bm


def test_gaps_satisfy_constraints():
    lo, hi = bm._ms(bm.MIN_GAP_MS), bm._ms(bm.MAX_GAP_MS)
    sep, edge = bm._ms(bm.MIN_SEP_MS), bm._ms(bm.EDGE_MS)
    for clip_id in ["piano_01", "music_03", "speech_05"]:
        for level in bm.GAP_LEVELS_MS:
            gaps = bm.make_gaps(level, clip_id)
            assert sum(l for _, l in gaps) == bm._ms(level)          # exact total
            assert all(lo <= l <= hi for _, l in gaps)                # gap length range
            assert gaps[0][0] >= edge                                 # edge margin
            assert gaps[-1][0] + gaps[-1][1] <= bm.N_SAMPLES - edge
            for (s1, l1), (s2, _) in zip(gaps, gaps[1:]):
                assert s2 - (s1 + l1) >= sep                          # separation


def test_masks_are_deterministic_and_match_total():
    a = bm.make_gaps(600, "speech_01")
    assert a == bm.make_gaps(600, "speech_01")
    assert a != bm.make_gaps(600, "speech_02")
    mask = bm.mask_from_gaps(a)
    assert (mask == 0).sum() == bm._ms(600)


def test_quick_levels_subset_of_full():
    assert set(bm.QUICK_LEVELS_MS) <= set(bm.GAP_LEVELS_MS)
