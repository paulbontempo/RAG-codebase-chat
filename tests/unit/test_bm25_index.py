from codebase_chat_tool.retrieval.bm25_index import build_bm25_index, tokenize


def test_tokenize_splits_snake_case():
    tokens = tokenize("def get_user_by_id(user_id):")
    assert "get" in tokens
    assert "user" in tokens
    assert "by" in tokens
    assert "id" in tokens


def test_tokenize_splits_camel_case():
    tokens = tokenize("class UserService:")
    assert "userservice" in tokens
    assert "user" in tokens
    assert "service" in tokens


def test_tokenize_lowercases_and_ignores_punctuation():
    tokens = tokenize("self._users.get(entity_id)")
    assert all(t == t.lower() for t in tokens)
    assert "users" in tokens
    assert "entity" in tokens


def test_bm25_search_ranks_matching_chunk_first():
    chunk_ids = ["a", "b", "c"]
    texts = [
        "def retry(times): pass",
        "def unrelated_function(): return 42",
        "class UserService: pass",
    ]
    index = build_bm25_index(chunk_ids, texts)
    results = index.search("retry logic", top_k=3)
    assert results[0][0] == "a"


def test_bm25_search_excludes_zero_score_results():
    chunk_ids = ["a", "b"]
    texts = ["def retry(times): pass", "completely unrelated text about widgets"]
    index = build_bm25_index(chunk_ids, texts)
    results = index.search("xyzxyz_no_match", top_k=5)
    assert results == []
