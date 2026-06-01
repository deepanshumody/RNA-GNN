"""Shared atom featurisation for RNA-GNN dataset construction.

Single source of truth for the per-atom feature vector, replacing four
copy-pasted copies in the dataset-creation scripts. It also fixes a latent bug:
magnesium — the ligand template atom — was silently encoded as *hydrogen*
because ``'Mg'`` was missing from the symbol vocabulary and
``one_of_k_encoding_unk`` folds unknown symbols onto the last entry. ``'Mg'`` is
now an explicit symbol and a dedicated ``'X'`` catch-all absorbs any genuinely
unknown element instead of corrupting a real one.

Note: extending the vocabulary widens the feature vector (was 28 per atom block,
now ``N_ATOM_FEATURES``). Downstream models read the width from the data rather
than hard-coding it, so regenerated datasets train without further changes.
"""
import numpy as np

# 'X' is an explicit catch-all so an unexpected element no longer masquerades as
# hydrogen (or, now, as magnesium).
ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "B", "H", "Mg", "X"]
DEGREES = [0, 1, 2, 3, 4, 5]
NUM_HS = [0, 1, 2, 3, 4]
IMPLICIT_VALENCES = [0, 1, 2, 3, 4, 5]

# symbol + degree + numHs + implicit-valence one-hots, plus the aromaticity flag.
N_ATOM_FEATURES = len(ATOM_SYMBOLS) + len(DEGREES) + len(NUM_HS) + len(IMPLICIT_VALENCES) + 1


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Map inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_feature(m, atom_i):
    """Per-atom one-hot feature vector of length :data:`N_ATOM_FEATURES`."""
    atom = m.GetAtomWithIdx(atom_i)
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), ATOM_SYMBOLS)
        + one_of_k_encoding_unk(atom.GetDegree(), DEGREES)
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), NUM_HS)
        + one_of_k_encoding_unk(atom.GetImplicitValence(), IMPLICIT_VALENCES)
        + [atom.GetIsAromatic()]
    )


def get_atom_feature(m, is_ligand=True):
    """Stack per-atom features and pad the opposite (ligand/receptor) channel.

    The receptor and ligand share one feature matrix: receptor atoms occupy the
    first :data:`N_ATOM_FEATURES` columns and ligand atoms the next
    :data:`N_ATOM_FEATURES`, with the other half zero-padded.
    """
    n = m.GetNumAtoms()
    feats = np.array([atom_feature(m, i) for i in range(n)])
    pad = np.zeros((n, N_ATOM_FEATURES))
    return np.concatenate([feats, pad], 1) if is_ligand else np.concatenate([pad, feats], 1)
