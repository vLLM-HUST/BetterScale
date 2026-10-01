"""CPU checks for interval accounting used in the D6 evidence report."""
from analyze_d_cluster import distribution
from analyze_d_timeline import union_ns, category

def test_union_does_not_add_overlapping_streams():
    assert union_ns([(0, 10), (5, 15), (3, 4), (20, 22)]) == 17
    assert union_ns([]) == 0
    assert union_ns([(5, 5), (4, 2)]) == 0

def test_distribution_preserves_units_and_empty_case():
    assert distribution([]) == {"n": 0}
    assert distribution([10, 20, 30])["median"] == 20
    assert distribution([10, 20, 30])["p90"] == 28

def test_hccl_not_compute_and_owned_attention_named():
    assert category("hcom_allReduce_") == "communication"
    assert category("GroupedMatmul") == "matmul"
    assert category("betterscale_context_parallel") == "attention"
