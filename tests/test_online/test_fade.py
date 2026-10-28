from textClust.online.fade import ExpFade
from textClust.online.cluster import Microcluster


def test_exp_fade_updates_weight_and_time_without_term_fading():
    mc = Microcluster(id=1, embedding={"a": 1.0}, current_time=0, n=1, weight=4.0)
    fade = ExpFade(decay_rate=1, term_fading=False, min_weight=0.5)

    updated, expired, expired_terms = fade(mc, current_time=1)

    assert updated.weight == 2.0
    assert updated.last_update_time == 1
    assert updated.embedding == {"a": 1.0}
    assert expired is False
    assert expired_terms == []


def test_exp_fade_expires_by_weight():
    mc = Microcluster(id=1, embedding={"a": 1.0}, current_time=0, n=1, weight=1.0)
    fade = ExpFade(decay_rate=2, term_fading=False, min_weight=0.5)

    updated, expired, expired_terms = fade(mc, current_time=2)

    assert updated.weight == 1.0 / 16
    assert expired is True
    assert expired_terms == []


def test_exp_fade_term_fading_can_expire_on_empty_terms():
    mc = Microcluster(id=1, embedding={"a": 0.8, "b": 0.4}, current_time=0, n=1, weight=2.0)
    fade = ExpFade(decay_rate=1, term_fading=True, min_weight=0.5)

    updated, expired, expired_terms = fade(mc, current_time=1)

    assert updated.weight == 1.0
    assert updated.embedding == {"a": 0.4, "b": 0.2}
    assert expired is True
    assert expired_terms == ["a", "b"]


def test_exp_fade_keeps_terms_at_min_weight():
    mc = Microcluster(id=1, embedding={"a": 1.0, "b": 0.4}, current_time=0, n=1, weight=2.0)
    fade = ExpFade(decay_rate=1, term_fading=True, min_weight=0.5)

    updated, expired, expired_terms = fade(mc, current_time=1)

    assert updated.embedding == {"a": 0.5, "b": 0.2}
    assert expired is False
    assert expired_terms == ["b"]


def test_exp_fade_term_fading_keeps_terms_above_min_weight():
    mc = Microcluster(id=1, embedding={"a": 2.0, "b": 1.2}, current_time=0, n=1, weight=2.0)
    fade = ExpFade(decay_rate=1, term_fading=True, min_weight=0.5)

    updated, expired, expired_terms = fade(mc, current_time=1)

    assert updated.weight == 1.0
    assert updated.embedding == {"a": 1.0, "b": 0.6}
    assert expired is False
    assert expired_terms == []


def test_exp_fade_reports_expired_terms_without_removing_them():
    mc = Microcluster(id=1, embedding={"keep": 2.0, "drop": 0.5}, current_time=0, n=1, weight=2.0)
    fade = ExpFade(decay_rate=1, term_fading=True, min_weight=0.5)

    updated, expired, expired_terms = fade(mc, current_time=1)

    assert updated.embedding == {"keep": 1.0, "drop": 0.25}
    assert expired is False
    assert expired_terms == ["drop"]
