import math

import numpy as np
import pytest

from spacebridge import Index, LexicalIndex, merge_pool
from spacebridge.lexical import tokens

TITLES = ["Apollo program", "", "Moon landing", "Cooking", "Apollo Moon", "Rivers"]
TEXTS = [
    "the apollo program reached the moon in 1969",
    "a recipe for bread and butter with bread flour",
    "the first moon landing was in 1969",
    "bread needs flour water and salt",
    "apollo moon apollo moon apollo moon apollo moon",
    "the nile is long",
]


def small_index() -> Index:
    return Index(np.zeros((len(TEXTS), 2), dtype=np.float32), TEXTS, TITLES)


def test_tokens_lowercase_keep_alphanumerics_and_drop_stop_words():
    assert tokens("The Apollo-11 landing, in 1969!") == ["apollo", "11", "landing", "1969"]
    assert tokens("What is it") == []
    assert tokens("中文查询") == []  # the tokenizer is English only, which is why auto stays off for other languages


def test_scores_follow_the_bm25_formula():
    lex = LexicalIndex.build(small_index())
    k1, b, N = 0.9, 0.4, len(TEXTS)
    docs = [tokens(f"{t} {x}") for t, x in zip(TITLES, TEXTS)]
    avgdl = sum(map(len, docs)) / N
    df = sum("bread" in d for d in docs)
    idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
    for row, doc in enumerate(docs):
        tf = doc.count("bread")
        want = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avgdl)) if tf else 0.0
        got = float(lex.X[row, lex.vocab["bread"]]) * float(lex.idf[lex.vocab["bread"]])
        assert got == pytest.approx(want, rel=1e-5)


def test_top_ranks_by_score_and_reads_title_and_text():
    lex = LexicalIndex.build(small_index())
    assert lex.top("bread flour", 2).tolist() == [1, 3]
    assert 0 in lex.top("apollo program", 1).tolist()  # "program" is in row 0 only, and "Apollo" also in its title
    assert lex.top("cooking", 5).tolist() == [3]  # title only


def test_top_with_no_match_is_empty_and_honours_k_and_exclude():
    lex = LexicalIndex.build(small_index())
    assert lex.top("zebra", 5).tolist() == []
    assert lex.top("the", 5).tolist() == []  # a stop word
    assert lex.top("moon", 0).tolist() == []
    assert len(lex.top("moon", 2)) == 2
    assert 4 not in lex.top("apollo moon", 6, exclude=4).tolist()
    assert not {0, 4} & set(lex.top("apollo moon", 6, exclude=np.array([0, 4])).tolist())
    np.testing.assert_array_equal(lex.top("moon", 6, exclude=-1), lex.top("moon", 6))  # no row, as the benchmark's self_rows


def test_ties_keep_row_order_whatever_k_is():
    titles = [""] * 40
    texts = ["same words here"] * 40  # every row ties
    lex = LexicalIndex.build(Index(np.zeros((40, 2)), texts, titles))
    assert lex.top("words", 40).tolist() == list(range(40))
    for k in (1, 7, 39):
        assert lex.top("words", k).tolist() == list(range(k))


def test_save_and_load_keep_the_results(tmp_path):
    lex = LexicalIndex.build(small_index())
    lex.save(tmp_path / "lex")
    back = LexicalIndex.load(tmp_path / "lex")
    assert len(back) == len(lex) and back.vocab == lex.vocab
    for q in ("bread flour", "apollo moon", "zebra", "1969"):
        np.testing.assert_array_equal(back.top(q, 5), lex.top(q, 5))


def test_load_refuses_another_kind_of_file_and_a_cut_short_save(tmp_path):
    from spacebridge import storage

    storage.save_dir(tmp_path / "other", "fitted-bridge", {"a": np.zeros(1)}, {})
    with pytest.raises(ValueError, match="not a lexical-index"):
        LexicalIndex.load(tmp_path / "other")
    lex = LexicalIndex.build(small_index())
    lex.save(tmp_path / "lex")
    (tmp_path / "lex" / "manifest.json").unlink()
    with pytest.raises(ValueError, match="did not finish"):
        LexicalIndex.load(tmp_path / "lex")


def test_merge_pool_takes_all_of_lexical_then_the_dense_head_with_no_repeats():
    dense = np.arange(10, 30)
    assert merge_pool(dense, np.array([1, 2]), 6).tolist() == [10, 11, 12, 13, 1, 2]
    # A row in both lists is counted once, so more of the dense list comes in: that's why dense is asked for the whole budget.
    assert merge_pool(dense, np.array([11, 2]), 6).tolist() == [10, 12, 13, 14, 11, 2]
    assert merge_pool(dense, np.array([], dtype=np.int64), 3).tolist() == [10, 11, 12]
    assert merge_pool(dense, None, 3).tolist() == [10, 11, 12]
    assert merge_pool(dense, np.array([5, -1, 6]), 4).tolist() == [10, 11, 5, 6]  # -1: no result
    assert merge_pool(np.array([3, -1, 4]), np.array([9]), 3).tolist() == [3, 4, 9]
    assert merge_pool(np.array([3]), np.array([9]), 5).tolist() == [3, 9]  # the lists ran out
    merged = merge_pool(dense, np.array([11, 2, 12]), 8)
    assert len(merged) == len(set(merged.tolist())) == 8
