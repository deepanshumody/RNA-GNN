"""Tests for structure-level (grouped) splitting.

Pure-Python (no heavy deps), so these run everywhere including CI. They guard
the core anti-leakage property: no structure's candidate sites may appear in
more than one split.
"""
import rna_splits


def _split_of(train, val, test):
    out = {}
    for name, idxs in (("train", train), ("val", val), ("test", test)):
        for i in idxs:
            out[i] = name
    return out


def test_no_group_spans_splits_and_partition_is_complete():
    groups = [f"pdb{i // 10}" for i in range(200)]  # 20 groups, 10 samples each
    train, val, test = rna_splits.group_train_val_test_split(groups, (0.8, 0.1, 0.1), seed=0)

    # Disjoint and complete.
    assert sorted(train + val + test) == list(range(200))
    assert not (set(train) & set(val))
    assert not (set(val) & set(test))
    assert not (set(train) & set(test))

    # Every group lands entirely in one split.
    split_of = _split_of(train, val, test)
    group_splits = {}
    for i, g in enumerate(groups):
        group_splits.setdefault(g, set()).add(split_of[i])
    for g, s in group_splits.items():
        assert len(s) == 1, f"group {g} leaked across splits: {s}"


def test_fractions_roughly_honored():
    groups = [f"p{i // 5}" for i in range(500)]  # 100 groups of 5
    train, val, test = rna_splits.group_train_val_test_split(groups, (0.8, 0.1, 0.1), seed=1)
    assert abs(len(train) / 500 - 0.8) < 0.06
    assert abs(len(val) / 500 - 0.1) < 0.06
    assert abs(len(test) / 500 - 0.1) < 0.06


def test_deterministic_for_a_seed():
    groups = [f"p{i // 5}" for i in range(100)]
    assert rna_splits.group_train_val_test_split(groups, seed=7) == \
        rna_splits.group_train_val_test_split(groups, seed=7)


def test_split_out_named_holdout():
    groups = ["A", "A", "B", "C", "B"]
    kept, held = rna_splits.split_out_groups(groups, ["B"])
    assert held == [2, 4]
    assert kept == [0, 1, 3]
