from textClust.online.cluster import Microcluster


def test_microcluster_initialization_sets_attributes():
    tf = {"term": 1.0}
    embedding = [0.1, 0.2]

    mc = Microcluster(id=10, tf=tf, current_time=5.0, n=3, embedding=embedding, weight=2.5)

    assert mc.id == 10
    assert mc.tf == {"term": 1.0}
    assert mc.last_update_time == 5.0
    assert mc.n == 3
    assert mc.embedding == [0.1, 0.2]
    assert mc.weight == 2.5


def test_microcluster_equality_true_for_same_values_without_embedding():
    mc1 = Microcluster(id=1, tf={"a": 1.0}, current_time=0.0, n=2, embedding=None, weight=1.0)
    mc2 = Microcluster(id=1, tf={"a": 1.0}, current_time=0.0, n=2, embedding=None, weight=1.0)

    assert mc1 == mc2


def test_microcluster_equality_false_for_non_microcluster():
    mc = Microcluster(id=1, tf={"a": 1.0}, current_time=0.0, n=1)

    assert (mc == {"id": 1}) is False


def test_microcluster_equality_false_when_core_fields_differ():
    base = Microcluster(id=1, tf={"a": 1.0}, current_time=0.0, n=2, weight=1.0)

    assert base != Microcluster(id=2, tf={"a": 1.0}, current_time=0.0, n=2, weight=1.0)
    assert base != Microcluster(id=1, tf={"a": 2.0}, current_time=0.0, n=2, weight=1.0)
    assert base != Microcluster(id=1, tf={"a": 1.0}, current_time=1.0, n=2, weight=1.0)
    assert base != Microcluster(id=1, tf={"a": 1.0}, current_time=0.0, n=3, weight=1.0)
    assert base != Microcluster(id=1, tf={"a": 1.0}, current_time=0.0, n=2, weight=2.0)


def test_microcluster_equality_embedding_logic():
    with_embedding_1 = Microcluster(
        id=1,
        tf={"a": 1.0},
        current_time=0.0,
        n=2,
        embedding=[0.5, 0.7],
        weight=1.0,
    )
    with_embedding_2 = Microcluster(
        id=1,
        tf={"a": 1.0},
        current_time=0.0,
        n=2,
        embedding=[0.5, 0.7],
        weight=1.0,
    )
    different_embedding = Microcluster(
        id=1,
        tf={"a": 1.0},
        current_time=0.0,
        n=2,
        embedding=[0.6, 0.7],
        weight=1.0,
    )
    no_embedding = Microcluster(id=1, tf={"a": 1.0}, current_time=0.0, n=2, embedding=None, weight=1.0)

    assert with_embedding_1 == with_embedding_2
    assert with_embedding_1 != different_embedding
    assert with_embedding_1 != no_embedding
