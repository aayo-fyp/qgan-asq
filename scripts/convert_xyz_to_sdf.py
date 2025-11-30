import os
import argparse
from rdkit import Chem


def fix_qm9_float(s):
    """
    Convert QM9 scientific notation:
        1.23*^4  →  1.23e4
        -6.1048*^-6 → -6.1048e-6
    """
    if "*^" in s:
        return s.replace("*^", "e")

    # handles "*^-6"
    if "*^-" in s:
        return s.replace("*^-", "e-")

    # handles "*^+6"
    if "*^+" in s:
        return s.replace("*^+", "e+")

    # handles "*-6" (rare variant)
    if "*-" in s:
        return s.replace("*-", "e-")

    # handles "*+6" 
    if "*+" in s:
        return s.replace("*+", "e+")

    return s


def xyz_to_mol(xyz_path):
    """Convert a QM9-style .xyz file into an RDKit Mol with cleaned floats."""
    try:
        with open(xyz_path, "r") as f:
            lines = f.readlines()

        natoms = int(lines[0].strip())
        atom_lines = lines[2:2 + natoms]

        mol = Chem.RWMol()
        conf = Chem.Conformer(natoms)

        for i, line in enumerate(atom_lines):
            parts = line.split()
            atom = parts[0]
            x = float(fix_qm9_float(parts[1]))
            y = float(fix_qm9_float(parts[2]))
            z = float(fix_qm9_float(parts[3]))

            idx = mol.AddAtom(Chem.Atom(atom))
            conf.SetAtomPosition(idx, (x, y, z))

        mol.AddConformer(conf)
        mol = mol.GetMol()
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
        return mol

    except Exception as e:
        print(f"Failed: {xyz_path} → {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    files = sorted([f for f in os.listdir(args.input) if f.endswith(".xyz")])
    print(f"Found {len(files)} .xyz files")

    writer = Chem.SDWriter(args.output)

    for i, fx in enumerate(files):
        xy = os.path.join(args.input, fx)
        mol = xyz_to_mol(xy)
        if mol:
            writer.write(mol)

        if i % 2000 == 0:
            print(f"{i}/{len(files)} processed...")

    writer.close()
    print("Done →", args.output)


if __name__ == "__main__":
    main()