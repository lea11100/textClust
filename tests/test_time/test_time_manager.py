import pytest

from textClust.time.manager import FixedIntervalTimeManager, TimeManager


def test_time_manager_initial_state_and_update_cycle():
    manager = TimeManager()

    assert manager.get_current_time() == 0

    manager.update(42)

    assert manager.get_current_time() == 42


def test_time_manager_base_methods_raise_not_implemented():
    manager = TimeManager()

    with pytest.raises(NotImplementedError, match=r"time_to_clean\(\) must be implemented by subclass"):
        manager.time_to_clean()

    with pytest.raises(NotImplementedError, match=r"time_to_cluster\(\) must be implemented by subclass"):
        manager.time_to_cluster()


def test_fixed_interval_time_manager_uses_default_intervals():
    manager = FixedIntervalTimeManager()

    manager.update(60)
    assert manager.time_to_clean() is True
    assert manager.time_to_cluster() is False

    manager.update(300)
    assert manager.time_to_clean() is True
    assert manager.time_to_cluster() is True


def test_fixed_interval_time_manager_respects_custom_intervals():
    manager = FixedIntervalTimeManager(clean_interval=7, cluster_interval=11)

    manager.update(14)
    assert manager.time_to_clean() is True
    assert manager.time_to_cluster() is False

    manager.update(22)
    assert manager.time_to_clean() is False
    assert manager.time_to_cluster() is True

    manager.update(77)
    assert manager.time_to_clean() is True
    assert manager.time_to_cluster() is True


def test_fixed_interval_time_manager_triggers_at_zero_time():
    manager = FixedIntervalTimeManager(clean_interval=5, cluster_interval=9)

    assert manager.get_current_time() == 0
    assert manager.time_to_clean() is True
    assert manager.time_to_cluster() is True
