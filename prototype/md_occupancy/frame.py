"""Frames of a trajectory written as PDB files, for the frames count.py names. A SKETCH; runs where MDAnalysis is.

    python frame.py topology.pdb trajectory.xtc out_prefix frame [frame ...]
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import MDAnalysis as mda

top, traj, prefix = sys.argv[1:4]
u = mda.Universe(top, traj)
for f in sorted({int(x) for x in sys.argv[4:]}):
    u.trajectory[f]
    u.atoms.write(f"{prefix}{f:05d}.pdb")
    print(f"{prefix.rsplit('/', 1)[-1]}{f:05d}.pdb at {u.trajectory.time / 1000:.0f} ns")
