"""Tests for createpredictedionpdb.write_merged_pdb_with_hetatms.

This is a pure, dependency-light function (numpy only), so it runs everywhere
including lightweight CI. It guards the strict PDB column formatting that the
demo relies on when merging predicted ions back into a structure.
"""
import numpy as np
from createpredictedionpdb import write_merged_pdb_with_hetatms

CLEAN_PDB = """\
ATOM      1  P    A A   1      10.000  20.000  30.000  1.00  0.00           P
ATOM      2  O5'  A A   1      11.000  21.000  31.000  1.00  0.00           O
END
"""


def test_merge_appends_correctly_formatted_hetatms(tmp_path):
    clean = tmp_path / "clean.pdb"
    clean.write_text(CLEAN_PDB)
    out = tmp_path / "merged.pdb"

    coords = np.array([[1.234, 5.678, 9.012], [-4.5, 0.0, 12.345]])
    write_merged_pdb_with_hetatms(str(clean), str(out), coords, res_name="MG")

    lines = out.read_text().splitlines()
    atom_lines = [ln for ln in lines if ln.startswith("ATOM")]
    het_lines = [ln for ln in lines if ln.startswith("HETATM")]

    # Original atoms preserved; one HETATM appended per predicted coordinate.
    assert len(atom_lines) == 2
    assert len(het_lines) == len(coords)
    # File terminates with an END record.
    assert lines[-1].strip() == "END"

    for ln, (x, y, z) in zip(het_lines, coords):
        assert ln[0:6] == "HETATM"            # record name, cols 1-6
        assert ln[17:20].strip() == "MG"      # residue name, cols 18-20
        # Coordinates land in the strict PDB columns 31-54 as %8.3f.
        assert float(ln[30:38]) == round(x, 3)
        assert float(ln[38:46]) == round(y, 3)
        assert float(ln[46:54]) == round(z, 3)


def test_serial_numbers_continue_from_max(tmp_path):
    clean = tmp_path / "clean.pdb"
    clean.write_text(CLEAN_PDB)
    out = tmp_path / "merged.pdb"

    write_merged_pdb_with_hetatms(str(clean), str(out), np.array([[0.0, 0.0, 0.0]]))

    het = [ln for ln in out.read_text().splitlines() if ln.startswith("HETATM")][0]
    # Max serial in the clean file is 2, so the first predicted ion is 3.
    assert int(het[6:11]) == 3
