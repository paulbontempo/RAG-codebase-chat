from codebase_chat_tool.retrieval.hybrid import reciprocal_rank_fusion


def test_fusion_favors_items_ranked_high_in_both_lists():
    dense = ["a", "b", "c"]
    bm25 = ["b", "a", "c"]
    fused = reciprocal_rank_fusion([dense, bm25])
    fused_ids = [cid for cid, _score in fused]
    # 'a' and 'b' both appear near the top of both lists, so they should
    # outrank 'c', which is last in both.
    assert fused_ids[:2] == sorted(["a", "b"], key=lambda x: fused_ids.index(x))
    assert fused_ids[-1] == "c"


def test_fusion_includes_items_present_in_only_one_list():
    dense = ["a", "b"]
    bm25 = ["c"]
    fused = reciprocal_rank_fusion([dense, bm25])
    fused_ids = {cid for cid, _score in fused}
    assert fused_ids == {"a", "b", "c"}


def test_fusion_scores_are_deterministic_and_symmetric_in_input_order():
    dense = ["a", "b"]
    bm25 = ["b", "a"]
    fused_1 = dict(reciprocal_rank_fusion([dense, bm25]))
    fused_2 = dict(reciprocal_rank_fusion([bm25, dense]))
    assert fused_1 == fused_2


def test_empty_lists_produce_empty_fusion():
    assert reciprocal_rank_fusion([[], []]) == []
