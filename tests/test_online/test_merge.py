from textClust.online.fade import ExpFade
from textClust.online.merge import SumMerger
from textClust.online.microcluster import Microcluster


def test_sum_merger_combines_weights_counts_and_terms():
    mc1 = Microcluster(id=1, tf={"a": 1.0, "b": 2.0}, current_time=0, n=2, weight=1.5)
    mc2 = Microcluster(id=2, tf={"b": 1.0, "c": 3.0}, current_time=0, n=3, weight=2.0)
    merger = SumMerger()

    merged = merger(mc1, mc2, current_time=10)

    assert merged.id == 1
    assert merged.weight == 3.5
    assert merged.n == 5
    assert merged.last_update_time == 10
    assert merged.tf == {"a": 1.0, "b": 3.0, "c": 3.0}


def test_sum_merger_applies_fader_before_merging():
    mc1 = Microcluster(id=1, tf={"a": 2.0}, current_time=0, n=1, weight=2.0)
    mc2 = Microcluster(id=2, tf={"a": 1.0, "b": 3.0}, current_time=0, n=2, weight=4.0)
    fader = ExpFade(decay_rate=1, term_fading=True, min_weight=0.1)
    merger = SumMerger()

    merged = merger(mc1, mc2, current_time=1, fader=fader)

    assert mc1.last_update_time == 1
    assert mc2.last_update_time == 1
    assert merged.weight == 3.0
    assert merged.n == 3
    assert merged.last_update_time == 1
    assert merged.tf == {"a": 1.5, "b": 1.5}
