"""What a trajectory's contacts say: the occupancy of each contact, the bound state of each frame, and the
frame to draw for the whole trajectory and for each state.

A SKETCH, not part of bindview: nothing here is tested or promised.

    python count.py topology.pdb LIGAND SMILES contacts.json out_plan.json ["what the simulation is"]

Runs in the core environment (bindview, NumPy): the contacts come from contacts.py. Writes a plan: for each
figure the frame to draw and the table of its contacts with their occupancy, and what a strip needs. A
drawn frame is named as a file beside the plan (``<plan>.frame_00000.pdb``), which frame.py writes;
occupancy.py reads the plan and draws.

The rules:

* One contact across frames is its kind, its residue and the ligand's atoms up to the symmetry of the
  ligand (``bindview.ensemble.symmetry_classes``); for a hydrogen bond also the partner's atom, the two free
  oxygens of a phosphate counted as one. A pi-cation contact is with a base, whichever ring; a salt bridge
  with a charged group; a hydrophobic contact with a residue.
* Its occupancy is the share of the frames that make it.
* The frame that is drawn has the largest sum, over all contacts, of the occupancy of those it makes and
  of one minus the occupancy of those it does not; of equal frames the earliest.
* A figure holds the drawn frame's contacts of an occupancy of 0.10 or more; a contact of 0.30 or more
  that the frame does not make is named beside it.
* A chain's pocket is occupied in a frame when the ligand makes two contacts that have a direction with
  the chain, one of them with a base. The state of a frame is the set of occupied chains.
* A trajectory whose commonest state holds less than 0.90 of the frames gets a figure for each state of
  0.10 or more, counted inside the state.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from bindview import Diagram, read_structure
from bindview.ensemble import symmetry_classes
from bindview.features import perceive_charged_groups
from bindview.residues import canonical_atom, charged_groups, is_backbone_atom

KIND = {"HBDonor": "hbond", "HBAcceptor": "hbond", "Cationic": "saltbridge", "Anionic": "saltbridge",
        "CationPi": "pi_cation", "PiCation": "pi_cation", "PiStacking": "pi_stacking", "Hydrophobic": "hydrophobic"}
DIRECTED = ("hbond", "saltbridge", "pi_cation")
DRAWN, NAMED, ONE_STATE, A_STATE, BIN_NS = 0.10, 0.30, 0.90, 0.10, 20.0     # rules 5, 7 and 8

topology, ligand, smiles, source, out = sys.argv[1:6]
data = json.load(open(source))
what = sys.argv[6] if len(sys.argv) > 6 else data["trajectory"]
st = read_structure(topology)
lig = Diagram.from_structure(st, ligand, smiles=smiles, interactions=[]).ligand      # (no table: none is looked for)
classes = symmetry_classes(lig)
group_of = {}
# (An atom can be in several charged groups, a nitrogen by itself and its guanidinium: the largest counts,
# so that a salt bridge of one residue is one row and not up to three.)
for g in sorted(perceive_charged_groups(lig), key=lambda g: len(g.atoms)):
    members = tuple(lig.names[i] for i in g.atoms)
    group_of.update({n: members for n in members})
by_class = {}
for atom_name, c in classes.items():
    by_class.setdefault(c, []).append(atom_name)
atoms_of = {(k.chain, k.number): [a.name for a in st.residue_atoms(k)] for k in st.residues()}


def read(raw):
    """A contact of ProLIF's as a key (what it is across frames, rule 2) and a record bindview takes."""
    _, kind, chain, number, resname, la, pa, distance = raw
    what = KIND[kind]
    rec = {"interaction_type": what, "residue_number": number, "residue_name": resname, "chain": chain,
           "distance": distance}
    if what == "hbond":
        partner = "OP" if canonical_atom(resname, pa[0]) in ("OP1", "OP2") else pa[0]
        rec.update(ligand_atom=la[0], protein_atom=pa[0], ligand_is_donor=kind == "HBDonor")
        return ("hbond", chain, number, classes[la[0]], partner), rec
    if what == "saltbridge":
        group = group_of.get(la[0], tuple(la))
        charged = [n for _, names in charged_groups(resname) for n in names if n in atoms_of[(chain, number)]]
        rec.update(ligand_atoms=list(group), protein_atoms=charged or list(pa))
        return ("saltbridge", chain, number, tuple(sorted({classes[n] for n in group}))), rec
    if what == "pi_cation":
        group = group_of.get(la[0], tuple(la)) if kind == "CationPi" else tuple(la)
        rec.update(ligand_atoms=list(group), protein_atoms=list(pa))
        return ("pi_cation", chain, number, tuple(sorted({classes[n] for n in group}))), rec
    if what == "pi_stacking":
        rec.update(ligand_atoms=list(la), protein_atoms=list(pa))
        return ("pi_stacking", chain, number, tuple(sorted({classes[n] for n in la}))), rec
    rec.update(ligand_atom=la[0], protein_atom=pa[0])
    return ("hydrophobic", chain, number), rec


n = data["frames"]
frames = [dict() for _ in range(n)]                      # per frame: key -> its record, the shortest of a key
for raw in data["contacts"]:
    key, rec = read(raw)
    held = frames[raw[0]].get(key)
    if held is None or rec["distance"] < held["distance"]:
        frames[raw[0]][key] = rec
keys = sorted({k for f in frames for k in f}, key=str)
index = {k: i for i, k in enumerate(keys)}
made = np.zeros((n, len(keys)), dtype=bool)
for f, found in enumerate(frames):
    made[f, [index[k] for k in found]] = True


def state_of(found) -> str:
    """Rule 6: a chain's pocket is occupied by two directed contacts with the chain, one of them with a base."""
    count, base = Counter(), Counter()
    for key, rec in found.items():
        if key[0] not in DIRECTED:
            continue
        count[key[1]] += 1
        with_base = key[0] == "pi_cation" or (key[0] == "hbond" and not is_backbone_atom(rec["residue_name"], rec["protein_atom"]))
        base[key[1]] += with_base
    return "+".join(sorted(c for c in count if count[c] >= 2 and base[c] >= 1)) or "none"


states = [state_of(f) for f in frames]


def label(key) -> str:
    resname = next(rec["residue_name"] for f in frames if key in f for rec in [f[key]])
    words = {"hbond": "hydrogen bond", "saltbridge": "salt bridge", "pi_cation": "pi-cation", "pi_stacking": "pi-stacking",
             "hydrophobic": "hydrophobic"}[key[0]]
    part = ""
    if key[0] == "hbond":
        part = (" to its phosphate" if key[4] == "OP" else f" to {key[4]}") + f" from {'/'.join(sorted(by_class[key[3]]))}"
    return f"{resname}{key[2]}({key[1]}) {words}{part}"


def figure(which: np.ndarray, name: str) -> dict:
    """Rules 3 to 5 on the frames ``which`` (a mask)."""
    occ = made[which].mean(axis=0)
    score = np.where(made, occ[None, :], 1.0 - occ[None, :]).sum(axis=1)
    score[~which] = -1.0
    best = int(np.argmax(score))                           # the earliest of equal frames
    records, names = [], []
    for key, rec in sorted(frames[best].items(), key=lambda kv: str(kv[0])):
        if occ[index[key]] >= DRAWN:
            records.append(dict(rec, occupancy=round(float(occ[index[key]]), 4)))
            names.append(label(key))                       # the record's name in ``occupancy``, record by record
    missing = [(label(k), round(float(occ[index[k]]), 3)) for k in keys
               if occ[index[k]] >= NAMED and k not in frames[best]]
    return {"name": name, "frames": int(which.sum()), "share": round(float(which.mean()), 4), "frame": best,
            "frame_file": f"{Path(out).stem}.frame_{best:05d}.pdb",
            "time_ns": data["time_ns"][best], "records": records, "labels": names, "not_drawn": missing,
            "left_out_below_0.10": sum(1 for k in frames[best] if occ[index[k]] < DRAWN),
            "occupancy": {label(k): round(float(occ[index[k]]), 3) for k in keys if occ[index[k]] >= DRAWN}}


count = Counter(states)
every = np.ones(n, dtype=bool)
figures = [figure(every, "all")]
if count.most_common(1)[0][1] / n < ONE_STATE:
    figures += [figure(np.array([s == state for s in states]), state)
                for state, c in sorted(count.items()) if c / n >= A_STATE]

# The strip (rule 8): the contacts of 0.10 or more, in bins of 20 ns, and the state of every frame.
t = np.array(data["time_ns"])
edges = np.arange(t[0], t[-1] + BIN_NS, BIN_NS)
which_bin = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 2)
occ = made.mean(axis=0)
rows = [k for k in keys if occ[index[k]] >= DRAWN]
strip = {"bin_ns": BIN_NS, "start_ns": float(t[0]), "bins": int(len(edges) - 1),
         "rows": [{"label": label(k), "kind": k[0], "chain": k[1], "number": k[2], "occupancy": round(float(occ[index[k]]), 3),
                   "share": [round(float(made[which_bin == b, index[k]].mean()), 3) if (which_bin == b).any() else 0.0
                             for b in range(len(edges) - 1)]} for k in rows],
         "states": sorted(count),
         "state_share": {s: [round(float(np.mean([states[i] == s for i in np.where(which_bin == b)[0]])), 3)
                             if (which_bin == b).any() else 0.0 for b in range(len(edges) - 1)] for s in sorted(count)}}
json.dump({"what": what, "ligand": ligand, "smiles": smiles,
           "source": source.rsplit("/", 1)[-1], "trajectory": data["trajectory"], "topology": data["topology"],
           "contacts_by": {"prolif": data["prolif"], "mdanalysis": data["mdanalysis"], "kinds": data["kinds"]},
           "frames": n,
           "contacts_found": len(keys), "states": {s: round(c / n, 4) for s, c in sorted(count.items())},
           "figures": figures, "strip": strip}, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"{data['trajectory']}: {n} frames, {len(keys)} different contacts, {len(rows)} at 0.10 or more")
print("   states:", {s: round(c / n, 3) for s, c in sorted(count.items())})
for fig in figures:
    print(f"   figure '{fig['name']}': {fig['frames']} frames, drawn frame {fig['frame']} ({fig['time_ns']:.0f} ns), "
          f"{len(fig['records'])} contacts drawn, {len(fig['not_drawn'])} named and not drawn, "
          f"{fig['left_out_below_0.10']} of the frame's left out as rare")
print("   frames to write (frame.py):", " ".join(str(f) for f in sorted({fig["frame"] for fig in figures})))
