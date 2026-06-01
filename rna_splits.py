"""Structure-level (grouped) train/val/test splitting.

Candidate binding sites taken from the *same* RNA structure are highly
correlated: two grid points a few angstrom apart share almost all of their
neighbouring atoms, so their graphs are near-duplicates. Splitting candidate
sites at random therefore leaks near-identical graphs across train/val/test and
inflates the reported metrics. These helpers keep every site from a given
structure (its PDB id) inside a single split, which is the honest way to
estimate generalisation to *unseen structures*.
"""
import random


def group_train_val_test_split(groups, fractions=(0.8, 0.1, 0.1), seed=42):
    """Split sample indices into train/val/test without splitting any group.

    Args:
        groups: sequence of group labels, one per sample (e.g. the PDB id each
            candidate site comes from).
        fractions: target ``(train, val, test)`` fractions of the *samples*.
        seed: RNG seed controlling which whole groups land in which split.

    Returns:
        ``(train_idx, val_idx, test_idx)`` — three sorted lists of integer
        indices whose union is ``range(len(groups))`` and where all samples
        sharing a group fall in exactly one split.
    """
    if len(fractions) != 3:
        raise ValueError("fractions must be (train, val, test)")
    total = float(sum(fractions))
    if total <= 0:
        raise ValueError("fractions must sum to a positive number")
    fractions = [f / total for f in fractions]

    # Indices grouped by label, in first-appearance order for determinism.
    by_group = {}
    for idx, g in enumerate(groups):
        by_group.setdefault(g, []).append(idx)

    unique_groups = list(by_group.keys())
    random.Random(seed).shuffle(unique_groups)

    n_samples = len(groups)
    targets = [f * n_samples for f in fractions]
    buckets = ([], [], [])
    counts = [0.0, 0.0, 0.0]

    for g in unique_groups:
        members = by_group[g]
        # Assign this whole group to the split that is currently furthest below
        # its target (by sample count); skip splits whose target is 0 so a 0.0
        # fraction stays empty.
        choice = max(
            (i for i in range(3) if targets[i] > 0),
            key=lambda i: targets[i] - counts[i],
        )
        buckets[choice].extend(members)
        counts[choice] += len(members)

    return tuple(sorted(b) for b in buckets)


def split_out_groups(groups, holdout):
    """Partition indices into ``(kept, heldout)`` by group membership.

    ``holdout`` is an iterable of group labels to pull out entirely — used to
    keep a named evaluation structure (e.g. a specific PDB) completely out of
    the training corpus instead of leaking its sites back in.
    """
    holdout = set(holdout)
    kept, held = [], []
    for idx, g in enumerate(groups):
        (held if g in holdout else kept).append(idx)
    return kept, held
