import torch

from textClust.online.cluster import Microcluster


def test_microcluster_initialization_sets_attributes():
    embedding = {"term": 1.0}
    mc = Microcluster(id=10, current_time=5.0, n=3, embedding=embedding, weight=2.5)

    assert mc.id == 10
    assert mc.embedding is embedding
    assert mc.last_update_time == 5.0
    assert mc.n == 3
    assert mc.weight == 2.5


def test_microcluster_equality_true_for_same_dictionary_values():
    mc1 = Microcluster(id=1, embedding={"a": 1.0}, current_time=0.0, n=2, weight=1.0)
    mc2 = Microcluster(id=1, embedding={"a": 1.0}, current_time=0.0, n=2, weight=1.0)

    assert mc1 == mc2


def test_microcluster_equality_false_for_non_microcluster():
    mc = Microcluster(id=1, embedding={"a": 1.0}, current_time=0.0, n=1)

    assert (mc == {"id": 1}) is False


def test_microcluster_equality_false_when_core_fields_differ():
    base = Microcluster(id=1, embedding={"a": 1.0}, current_time=0.0, n=2, weight=1.0)

    assert base != Microcluster(id=2, embedding={"a": 1.0}, current_time=0.0, n=2, weight=1.0)
    assert base != Microcluster(id=1, embedding={"a": 2.0}, current_time=0.0, n=2, weight=1.0)
    assert base != Microcluster(id=1, embedding={"a": 1.0}, current_time=1.0, n=2, weight=1.0)
    assert base != Microcluster(id=1, embedding={"a": 1.0}, current_time=0.0, n=3, weight=1.0)
    assert base != Microcluster(id=1, embedding={"a": 1.0}, current_time=0.0, n=2, weight=2.0)


def test_microcluster_equality_tensor_embedding_logic():
    mc1 = Microcluster(id=1, embedding=torch.tensor([0.5, 0.7]), current_time=0.0, n=2)
    mc2 = Microcluster(id=1, embedding=torch.tensor([0.5, 0.7]), current_time=0.0, n=2)
    different = Microcluster(id=1, embedding=torch.tensor([0.6, 0.7]), current_time=0.0, n=2)
    dictionary = Microcluster(id=1, embedding={"a": 1.0}, current_time=0.0, n=2)

    assert mc1 == mc2
    assert mc1 != different
    assert mc1 != dictionary
