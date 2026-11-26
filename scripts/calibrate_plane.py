#!/usr/bin/env python
"""What the two thresholds of the plane layout mean in terms of shape.

    python scripts/calibrate_plane.py                  # writes tests/data/plane-calibration.json
    python scripts/calibrate_plane.py --out rows.json --reps 50

An interface whose residues come in several stretches of the chain can be
drawn with each box where its residue lies on the plane through them. The
projection is drawn only if it keeps at least 90 % of neighbour relations and
is off by no more than 10 % in distances (:mod:`bindview.plane`). This script
says what those two numbers let through: it lays points at the spacing of
residue centres on surfaces of known shape -- a plane, spherical caps and
cylinders that turn by a stated angle from their middle to their rim --
roughens them, and measures the projection of each with the library's own
functions.

Nothing is read from disk and nothing is fitted. Every row has its own seed,
so every run gives the same numbers. The rows are written as JSON (one for
each shape, number of residues and turn); a short table of the sets of 20 and
30 residues is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bindview.plane import APART, ERROR_MAX, KEPT_MIN, NEAR, fidelity, project  # noqa: E402

#: Square ångström of surface per residue centre, and the least distance between two centres.
AREA, MIN_DISTANCE = 36.0, 4.3
#: Roughness: how far a centre lies off its surface (standard deviation, ångström), and how much of
#: that is along the surface.
ROUGHNESS, ALONG = 1.5, 0.35
SHAPES = ("cap", "cylinder")
SIZES = (6, 8, 12, 20, 30)
TURNS = (0, 30, 45, 60, 75, 90, 105, 120, 150, 180)
REPS = 300


def disc_points(n: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """``n`` points in a disc of ``n * AREA`` square ångström, no two closer than :data:`MIN_DISTANCE`."""
    rho = float(np.sqrt(n * AREA / np.pi))
    pts: list[np.ndarray] = []
    tries = 0
    while len(pts) < n:
        tries += 1
        if tries > 20000:
            pts, tries = [], 0
        r, phi = rho * np.sqrt(rng.random()), 2 * np.pi * rng.random()
        p = np.array([r * np.cos(phi), r * np.sin(phi)])
        if all(np.linalg.norm(p - q) >= MIN_DISTANCE for q in pts):
            pts.append(p)
    return np.array(pts), rho


def on_cap(uv: np.ndarray, rho: float, turn: float) -> np.ndarray:
    """The disc laid on a spherical cap whose surface turns by ``turn`` (radians) from its pole to its rim."""
    if turn < 1e-9:
        return np.column_stack([uv, np.zeros(len(uv))])
    radius = rho / turn
    r, phi = np.linalg.norm(uv, axis=1), np.arctan2(uv[:, 1], uv[:, 0])
    a = r / radius
    return np.column_stack([radius * np.sin(a) * np.cos(phi), radius * np.sin(a) * np.sin(phi),
                            radius * np.cos(a) - radius])


def on_cylinder(uv: np.ndarray, rho: float, turn: float) -> np.ndarray:
    """The disc wrapped round a cylinder (axis along x) that turns by ``turn`` from its middle to its edge."""
    if turn < 1e-9:
        return np.column_stack([uv, np.zeros(len(uv))])
    radius = rho / turn
    a = uv[:, 1] / radius
    return np.column_stack([uv[:, 0], radius * np.sin(a), radius * np.cos(a) - radius])


def sample(shape: str, n: int, turn_deg: float, rng: np.random.Generator) -> np.ndarray:
    """One rough set of ``n`` residue centres on ``shape``."""
    uv, rho = disc_points(n, rng)
    xyz = (on_cap if shape == "cap" else on_cylinder)(uv, rho, float(np.radians(turn_deg)))
    return xyz + rng.normal(0.0, ROUGHNESS, size=(len(xyz), 3)) * np.array([ALONG, ALONG, 1.0])


def row(shape: str, n: int, turn_deg: int, reps: int = REPS) -> dict:
    """The projection of ``reps`` point sets of one kind, summarised."""
    rng = np.random.default_rng(7000 + n * 13 + turn_deg)
    kept, error, false, rms, ok = [], [], [], [], 0
    for _ in range(reps):
        xyz = sample(shape, n, turn_deg, rng)
        proj = project(xyz)
        f = fidelity(xyz, proj.xy)
        kept.append(f.kept)
        error.append(f.error)
        false.append(f.false)
        rms.append(proj.rms)
        ok += f.ok

    def spread(values: list) -> list[float]:
        v = np.array(values, dtype=float)
        return [round(float(np.median(v)), 3), round(float(np.percentile(v, 10)), 3),
                round(float(np.percentile(v, 90)), 3)]

    return {"shape": shape, "residues": n, "turn_deg": turn_deg, "sets": reps, "kept": spread(kept),
            "distance_error": spread(error), "false_neighbours_median": float(np.median(false)),
            "false_neighbours_max": int(max(false)), "out_of_plane_rms": spread(rms),
            "share_drawn": round(ok / reps, 3)}


def run(reps: int = REPS) -> dict:
    return {"thresholds": {"neighbours_kept_min": KEPT_MIN, "distance_error_max": ERROR_MAX,
                           "neighbour_within": NEAR, "apart_beyond": APART},
            "points": {"area_per_residue": AREA, "min_distance": MIN_DISTANCE, "roughness": ROUGHNESS,
                       "roughness_along_surface": ALONG, "sets_per_row": reps},
            "rows": [row(shape, n, turn, reps) for shape in SHAPES for n in SIZES for turn in TURNS]}


def summary(result: dict) -> str:
    """The sets of 20 and 30 residues as a short table: the medians of the two sizes, and the share of sets drawn."""
    rows = result["rows"]

    def pick(shape: str, n: int, turn: int) -> dict:
        return next(r for r in rows if (r["shape"], r["residues"], r["turn_deg"]) == (shape, n, turn))

    def band(shape: str, turn: int, key: str, places: int) -> str:
        values = [round(pick(shape, n, turn)[key][0], places) for n in (20, 30)]
        lo, hi = min(values), max(values)
        return f"{lo:.{places}f}" if lo == hi else f"{lo:.{places}f}-{hi:.{places}f}"

    def drawn(shape: str, turn: int) -> str:
        values = [100.0 * pick(shape, n, turn)["share_drawn"] for n in (20, 30)]
        lo, hi = min(values), max(values)
        return f"{lo:.0f} %" if round(lo) == round(hi) else f"{lo:.0f}-{hi:.0f} %"

    out = [f"A projection is drawn if it keeps at least {100 * KEPT_MIN:.0f} % of neighbour relations and its distances "
           f"are off by no more than {100 * ERROR_MAX:.0f} %.",
           f"Points: one per {AREA:g} square angstrom, none closer than {MIN_DISTANCE:g} angstrom, roughened by "
           f"{ROUGHNESS:g} angstrom; {result['points']['sets_per_row']} sets a row.",
           "Medians over the sets of 20 and of 30 residues; drawn: the share of sets that pass both thresholds.", "",
           f"{'shape':<15}{'turn':>6}  {'kept':<12}{'distance error':<16}drawn"]
    for shape, turns in (("cap", (0, 30, 45, 60, 75, 90, 120)), ("cylinder", (60, 90, 120, 150))):
        for turn in turns:
            name = "flat" if turn == 0 else ("spherical cap" if shape == "cap" else "cylinder")
            out.append(f"{name:<15}{turn:>5}°  {band(shape, turn, 'kept', 2):<12}"
                       f"{band(shape, turn, 'distance_error', 3):<16}{drawn(shape, turn)}")
    return "\n".join(out)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "tests" / "data" / "plane-calibration.json"),
                    help="the JSON file to write (tests/data/plane-calibration.json)")
    ap.add_argument("--reps", type=int, default=REPS, help=f"point sets for each row ({REPS})")
    args = ap.parse_args(argv)
    result = run(args.reps)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(summary(result))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
