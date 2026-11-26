"""The contacts of a ligand in every frame of a trajectory, found by ProLIF and written as one JSON file.

A SKETCH, not part of bindview: nothing here is tested or promised.

Runs where MDAnalysis and ProLIF are (an environment of its own: MDAnalysis is GPL). bindview reads no
trajectory; it reads what this writes (count.py, occupancy.py).

    python contacts.py topology.pdb trajectory.xtc LIGAND out.json [stride]

A contact is written as [frame, kind, chain, residue number, residue name, ligand atoms, partner atoms,
distance]: heavy atoms only, by their names in the topology. ``kind`` is ProLIF's name for it.
"""
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import MDAnalysis as mda
import prolif as plf
from rdkit import Chem

KINDS = ["HBDonor", "HBAcceptor", "Cationic", "Anionic", "CationPi", "PiCation", "PiStacking", "Hydrophobic"]

top, traj, ligand, out = sys.argv[1:5]
stride = int(sys.argv[5]) if len(sys.argv) > 5 else 1
t0 = time.time()
u = mda.Universe(top, traj)
u.trajectory[0]
if not hasattr(u.atoms, "bonds") or not len(u.atoms.bonds):
    u.atoms.guess_bonds()                                   # a topology without bonds: from the first frame's distances
lig = u.select_atoms(f"resname {ligand}")
rec = u.select_atoms(f"not resname {ligand}")
lmol, pmol = plf.Molecule.from_mda(lig), plf.Molecule.from_mda(rec)


def name(mol, i: int) -> str:
    return mol.GetAtomWithIdx(int(i)).GetMonomerInfo().GetName().strip()


def heavy(mol, indices) -> list:
    return [name(mol, i) for i in indices if mol.GetAtomWithIdx(int(i)).GetAtomicNum() > 1]


fp = plf.Fingerprint(KINDS)
frames = u.trajectory[::stride]
fp.run(frames, lig, rec, progress=False, n_jobs=1)
times = [float(ts.time) / 1000.0 for ts in u.trajectory[::stride]]
contacts = []
for k, frame in enumerate(sorted(fp.ifp)):
    for (_, res), found in fp.ifp[frame].items():
        for kind, hits in found.items():
            for h in hits:
                contacts.append([k, kind, res.chain or "", int(res.number), res.name,
                                 heavy(lmol, h["indices"]["ligand"]), heavy(pmol, h["parent_indices"]["protein"]),
                                 round(float(h.get("distance", float("nan"))), 2)])
json.dump({"topology": top.rsplit("/", 1)[-1], "trajectory": traj.rsplit("/", 1)[-1], "stride": stride,
           "frames": len(times), "time_ns": times, "kinds": KINDS,
           "ligand": {"resname": ligand, "smiles": Chem.MolToSmiles(Chem.RemoveHs(lmol)),
                      "charge": Chem.GetFormalCharge(lmol), "atoms": heavy(lmol, range(lmol.GetNumAtoms()))},
           "prolif": plf.__version__, "mdanalysis": mda.__version__, "contacts": contacts}, open(out, "w"))
print(f"{traj.rsplit('/', 1)[-1]}: {len(times)} frames, {len(contacts)} contacts, {time.time() - t0:.0f} s")
