"""Tests for the shared atom featuriser, in particular the magnesium fix.

The pure-encoding checks need only numpy, so they run in CI and permanently
guard against the old bug where 'Mg' was silently encoded as hydrogen.
"""
import os
import sys

# dataset_creation/ is a script directory (not a package); put it on the path.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset_creation")
)

import atom_features as af  # noqa: E402


def test_magnesium_is_its_own_symbol_not_hydrogen():
    enc = af.one_of_k_encoding_unk("Mg", af.ATOM_SYMBOLS)
    hot = [i for i, v in enumerate(enc) if v]
    assert hot == [af.ATOM_SYMBOLS.index("Mg")]
    assert af.ATOM_SYMBOLS.index("Mg") != af.ATOM_SYMBOLS.index("H")


def test_unknown_element_folds_to_catch_all_not_a_real_atom():
    enc = af.one_of_k_encoding_unk("Uup", af.ATOM_SYMBOLS)
    hot = [i for i, v in enumerate(enc) if v]
    assert hot == [af.ATOM_SYMBOLS.index("X")]  # not 'H', not 'Mg'


def test_known_atom_encodes_in_place():
    enc = af.one_of_k_encoding_unk("C", af.ATOM_SYMBOLS)
    assert enc[0] is True or enc[0] == 1
    assert sum(bool(v) for v in enc) == 1


def test_feature_width_constant_matches_vocabulary():
    expected = (
        len(af.ATOM_SYMBOLS) + len(af.DEGREES) + len(af.NUM_HS) + len(af.IMPLICIT_VALENCES) + 1
    )
    assert af.N_ATOM_FEATURES == expected


def test_get_atom_feature_shape_and_channels_with_rdkit():
    Chem = __import__("pytest").importorskip("rdkit.Chem", reason="rdkit not installed")
    mol = Chem.MolFromSmiles("CCO")  # sanitized: implicit valences are computed
    assert mol is not None

    ligand = af.get_atom_feature(mol, is_ligand=True)
    receptor = af.get_atom_feature(mol, is_ligand=False)
    n = mol.GetNumAtoms()

    # Receptor + ligand channels, each N_ATOM_FEATURES wide.
    assert ligand.shape == (n, 2 * af.N_ATOM_FEATURES)
    assert receptor.shape == (n, 2 * af.N_ATOM_FEATURES)
    # Ligand atoms occupy the first half (receptor half zero); receptor atoms the
    # second half (ligand half zero).
    assert ligand[:, af.N_ATOM_FEATURES:].sum() == 0
    assert receptor[:, : af.N_ATOM_FEATURES].sum() == 0
    # The first carbon must light up the 'C' symbol column.
    assert ligand[0, af.ATOM_SYMBOLS.index("C")] == 1
