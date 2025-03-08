import numpy as np

def write_merged_pdb_with_hetatms(
    pdb_clean_path: str,
    output_pdb_path: str,
    fixed_coords: np.ndarray,
    chain_id: str = "A",
    res_name: str = "MG"
):
    """
    Merge predicted HETATM coordinates (held in-memory via 'fixed_coords')
    into a cleaned PDB (with Mg removed). The merged PDB has newly added
    HETATM lines for the predicted coordinates, each labeled (by default)
    as 'MG' in chain A. Adjust as needed.

    Args:
        pdb_clean_path (str): Path to the cleaned PDB file (Mg HETATM removed).
        output_pdb_path (str): Path to write the merged PDB file.
        fixed_coords (np.ndarray): shape (N, 3) array of predicted coordinates (floats).
        chain_id (str): Chain identifier for these new HETATM lines.
        res_name (str): Residue name for these new HETATMs. Default 'MG'.
    """

    # Keep a reference for typical PDB columns, for clarity:
    #  COLUMNS   DATA  TYPE    FIELD          DEFINITION
    #  ------------------------------------------------------
    #  1 -  6    Record name   "HETATM"
    #  7 - 11    Integer       Atom serial number (right-justified)
    # 12         Blank
    # 13 - 16    Atom name     (e.g., "MG"  or " CA ")
    # 17         AltLoc        (usually blank if not used)
    # 18 - 20    Residue name  (right-justified)
    # 21         Blank
    # 22         Chain ID
    # 23 - 26    Residue sequence number (right-justified)
    # 27         iCode         (Insertion code, usually blank)
    # 28 - 30    Blank
    # 31 - 38    Real(8.3)     x coordinate
    # 39 - 46    Real(8.3)     y coordinate
    # 47 - 54    Real(8.3)     z coordinate
    # 55 - 60    Real(6.2)     Occupancy
    # 61 - 66    Real(6.2)     Temperature factor
    # 67 - 76    Blank (for Segment ID in old formats)
    # 77 - 78    LString(2)    Element symbol (right-justified)
    # 79 - 80    LString(2)    Charge (optional; often blank)

    # 1) Read the cleaned PDB, gather lines, and find the maximum atom serial number
    clean_lines = []
    max_serial = 0

    with open(pdb_clean_path, "r") as f_in:
        for line in f_in:
            # Collect the lines we want to keep
            if line.startswith(("ATOM", "HETATM")):
                # Attempt to parse the atom serial number
                try:
                    serial_str = line[6:11].strip()  # columns 7-11 are [6:11] in Python indexing
                    serial_num = int(serial_str)
                    max_serial = max(max_serial, serial_num)
                except ValueError:
                    pass

                clean_lines.append(line.rstrip('\n'))

            elif not line.startswith(("END", "CONECT", "MASTER", "TER")):
                # Keep header, SEQRES, etc.
                clean_lines.append(line.rstrip('\n'))

    # 2) Build new HETATM lines from fixed_coords
    new_hetatm_lines = []
    current_serial = max_serial

    # We will default occupancy=1.00, tempFactor=0.00, element same as resName if 'MG'
    occupancy = 1.00
    temp_factor = 0.00
    element_symbol = res_name.strip().upper()  # e.g. 'MG'

    for i, (x, y, z) in enumerate(fixed_coords, start=1):
        current_serial += 1

        # Construct a new HETATM line with strict PDB formatting:
        #
        # HETATM vs ATOM (cols 1-6)  -> "HETATM"
        # Atom serial (cols 7-11)   -> right-justified integer
        # Space (col 12)
        # Atom name (cols 13-16)    -> e.g. "MG"
        # AltLoc (col 17)          -> usually blank
        # ResName (cols 18-20)     -> right-justified
        # Space (col 21)
        # ChainID (col 22)         -> 'A' (default)
        # ResSeq (cols 23-26)      -> integer, right-justified
        # iCode (col 27)           -> blank
        # 3 spaces (cols 28-30)
        # x coord (cols 31-38)     -> float(8.3)
        # y coord (cols 39-46)     -> float(8.3)
        # z coord (cols 47-54)     -> float(8.3)
        # Occupancy (cols 55-60)   -> float(6.2)
        # TempFactor (cols 61-66)  -> float(6.2)
        # 10 blanks (cols 67-76)
        # Element (cols 77-78)     -> right-justified
        # Charge (cols 79-80)      -> blank
        #
        # We'll place the atom name as "MG" in columns 13-16 with trailing spaces if needed.

        line = (
            f"HETATM"                     # cols 1-6
            f"{current_serial:5d}"       # cols 7-11
            f" "                         # col 12
            f"{res_name:<4s}"           # cols 13-16; e.g. "MG  "
            f" "                         # col 17 altLoc
            f"{res_name:>3s}"           # cols 18-20
            f" "                         # col 21
            f"{chain_id:1s}"            # col 22
            f"{i:4d}"                   # cols 23-26 (resSeq)
            f"    "                      # cols 27-30 (iCode + spaces)
            f"{x:8.3f}"                 # cols 31-38
            f"{y:8.3f}"                 # cols 39-46
            f"{z:8.3f}"                 # cols 47-54
            f"{occupancy:6.2f}"         # cols 55-60
            f"{temp_factor:6.2f}"       # cols 61-66
            f"          "               # cols 67-76 (blank)
            f"{element_symbol:>2s}"     # cols 77-78 (Element)
            "\n"
        )
        new_hetatm_lines.append(line)

    # 3) Write out the final merged file
    with open(output_pdb_path, "w") as f_out:
        # Write all lines from the cleaned PDB
        for line in clean_lines:
            f_out.write(line + "\n")

        # Append the newly created HETATM lines
        for het_line in new_hetatm_lines:
            f_out.write(het_line)

        # Ensure an END record at the bottom
        f_out.write("END\n")

    print(f"[+] Merged PDB with predicted HETATMs saved to: {output_pdb_path}")
