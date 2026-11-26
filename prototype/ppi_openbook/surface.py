"""A molecular surface drawn with NumPy alone: a height field seen from the front, shaded, as a PNG.

A SKETCH, not part of bindview: nothing here is tested or promised. A figure of two proteins opened like a book
needs no renderer from outside: the surface below is the solvent-excluded one as far as a viewer can see it (the
spheres grown by a probe, then worn back by it), and the whole of it is a z-buffer, one erosion and a few gradients.
"""

from __future__ import annotations

import math
import struct
import zlib

import numpy as np

VDW = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8, "SE": 1.9}
PROBE = 1.4


def radii(elements) -> np.ndarray:
    return np.array([VDW.get(e.upper(), 1.8) for e in elements], dtype=float)


def zbuffer(xyz: np.ndarray, rad: np.ndarray, scale: float, bounds, cut: "float | None" = None
            ) -> "tuple[np.ndarray, np.ndarray]":
    """Height (ångström, towards the viewer) and the atom in front, per pixel. ``bounds`` = x0, x1, y0, y1.

    ``cut`` takes away everything in front of that height: where a sphere is cut through, the height is
    the height of the cut. Of two spheres at one height in a pixel the first in ``xyz`` is in front.
    """
    x0, x1, y0, y1 = bounds
    w, h = int(math.ceil((x1 - x0) * scale)), int(math.ceil((y1 - y0) * scale))
    z = np.full((h, w), -np.inf, dtype=np.float32)
    front = np.full((h, w), -1, dtype=np.int32)
    px, py, pr = (xyz[:, 0] - x0) * scale, (y1 - xyz[:, 1]) * scale, rad * scale
    # From the front: most spheres of a body are behind what is drawn already, and are seen to be by one look.
    top = (xyz[:, 2] + rad if cut is None else np.minimum(xyz[:, 2] + rad, cut)).astype(np.float32)
    for i in np.argsort(-top, kind="stable").tolist():
        r = pr[i]
        a0, a1 = max(0, int(math.floor(px[i] - r))), min(w, int(math.ceil(px[i] + r)) + 1)
        b0, b1 = max(0, int(math.floor(py[i] - r))), min(h, int(math.ceil(py[i] + r)) + 1)
        if a0 >= a1 or b0 >= b1:
            continue
        sub, first = z[b0:b1, a0:a1], front[b0:b1, a0:a1]
        if top[i] < sub.min():
            continue
        xs = np.arange(a0, a1) + 0.5 - px[i]
        ys = np.arange(b0, b1) + 0.5 - py[i]
        d2 = xs[None, :] ** 2 + ys[:, None] ** 2
        half = np.sqrt(np.maximum(r * r - d2, 0.0)) / scale
        zz = (xyz[i, 2] + half).astype(np.float32)
        if cut is not None:
            zz = np.where(zz <= cut, zz, np.where(xyz[i, 2] - half < cut, cut, -np.inf)).astype(np.float32)
        better = (d2 < r * r) & ((zz > sub) | ((zz == sub) & (first > i)))
        sub[better] = zz[better]
        first[better] = i
    return z, front


def wear_back(z: np.ndarray, probe_px: float, scale: float, rows: int = 128) -> np.ndarray:
    """Erosion of a height field by a ball: what a probe rolling over the grown spheres leaves.

    The ball is as high to the left as to the right of its middle, and above as below it: the lowest of the four
    places is found first and the ball's height taken from that once (the same number, to the bit). ``rows`` of the
    picture at a time, so that they stay in the processor's cache.
    """
    r, reach = int(math.ceil(probe_px)), probe_px * probe_px
    h, w = z.shape
    pad = np.pad(z, r, constant_values=-np.inf)
    out = np.full_like(z, np.inf)
    for y in range(0, h, rows):
        here = out[y:y + rows]
        n = len(here)
        around, low = pad[y:y + n + 2 * r], np.empty_like(here)
        for dx in range(r + 1):
            if dx * dx > reach:
                break
            pair = np.minimum(around[:, r - dx:r - dx + w], around[:, r + dx:r + dx + w])
            for dy in range(r + 1):
                d2 = dx * dx + dy * dy
                if d2 > reach:
                    break
                np.minimum(pair[r - dy:r - dy + n], pair[r + dy:r + dy + n], out=low)
                np.subtract(low, math.sqrt(reach - d2) / scale, out=low)
                np.minimum(here, low, out=here)
    return out


def blur(a: np.ndarray, radius: int, passes: int = 3) -> np.ndarray:
    """A box blur run three times: near enough a Gaussian, and it needs nothing but sums."""
    if radius < 1:
        return a
    out = a.astype(np.float64)
    for _ in range(passes):
        for axis in (0, 1):
            pad = np.concatenate([np.repeat(np.take(out, [0], axis), radius + 1, axis), out,
                                  np.repeat(np.take(out, [-1], axis), radius, axis)], axis)
            c = np.cumsum(pad, axis)
            n = out.shape[axis]
            hi = np.take(c, np.arange(2 * radius + 1, 2 * radius + 1 + n), axis)
            lo = np.take(c, np.arange(0, n), axis)
            out = (hi - lo) / (2 * radius + 1)
    return out


#: Licorice: an atom that is not carbon has its own colour, carbon has the colour of what it belongs to.
ELEMENT = {"N": (0.19, 0.34, 0.80), "O": (0.83, 0.24, 0.16), "S": (0.80, 0.62, 0.10), "P": (0.88, 0.48, 0.10),
           "SE": (0.80, 0.50, 0.10)}
STICK = 0.30         # ångström, the half width of a stick


def sticks(xyz: np.ndarray, elements, carbon, radius: float = STICK, step: float = 0.10, colours=None
           ) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Atoms as licorice, for :func:`render`: a string of small spheres along every bond.

    Two atoms are bonded when they are close enough to be (1.95 Å, 2.25 Å where one is sulfur or
    phosphorus): there are no hydrogens and no bond orders here, and a stick needs neither. Comes back as
    centres, radii and colours, half of a bond in the colour of each of its atoms. ``colours`` gives every
    atom a colour of its own (by what its residue is coloured by, say) in place of its element's.
    """
    xyz = np.asarray(xyz, dtype=float)
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    big = np.array([e.upper() in ("S", "P", "SE") for e in elements])
    bonded = np.argwhere(np.triu((d < np.where(big[:, None] | big[None, :], 2.25, 1.95)) & (d > 0.4), 1))
    colour = (np.array([ELEMENT.get(e.upper(), tuple(carbon)) for e in elements], dtype=float) if colours is None
              else np.asarray(colours, dtype=float).reshape(len(xyz), 3))
    points, colours = [xyz], [colour]
    for i, j in bonded:
        t = np.linspace(0.0, 1.0, max(2, int(math.ceil(d[i, j] / step))) + 1)[1:-1, None]
        points.append(xyz[i] + t * (xyz[j] - xyz[i]))
        colours.append(np.where(t < 0.5, colour[i], colour[j]))
    centres = np.concatenate(points)
    return centres, np.full(len(centres), radius), np.concatenate(colours)


def render(xyz: np.ndarray, elements, colours: np.ndarray, bounds, scale: float, oversample: int = 2,
           probe: float = PROBE, flat: float = 0.0, light=(-0.45, 0.55, 0.70),
           groups: "np.ndarray | None" = None, licorice=None, cut: "float | None" = None, hatched: bool = True
           ) -> "tuple[np.ndarray, np.ndarray | None]":
    """An RGBA picture (uint8) of the surface of these atoms, seen from +z.

    ``colours`` is one RGB triple (0 to 1) per atom. ``flat`` between 0 and 1 takes the shading out, for a
    picture that is to lie under a drawing. ``groups`` is a whole number per atom (-1 for none): a white
    line is drawn on the surface wherever the group changes, as a border is on a map, and the group that
    is in front comes back per pixel beside the picture (-2 where there is no surface, -3 where a stick
    is in front). ``licorice`` is what :func:`sticks` gives: atoms drawn as sticks in the same picture,
    in front of the surface where they are in front of it and hidden by it where they are not. ``cut``
    is a height: what is in front of it is taken away, and where the body is cut through, the cut is
    drawn as a flat, hatched face (the sticks are not cut); without ``hatched`` as a flat, plain one.
    """
    s = scale * oversample
    rad = radii(elements)
    # (Grown, cut a probe's radius beyond the cut, then worn back: the surface that is left is cut where asked.)
    z, front = zbuffer(xyz, rad + probe, s, bounds, None if cut is None else cut + probe)
    z = wear_back(z, probe * s, s)
    mask = np.isfinite(z)
    cut_face = mask & (z > cut - 1e-3) if cut is not None else np.zeros(mask.shape, dtype=bool)
    tint = colours[np.clip(front, 0, None)]
    stick = np.zeros(mask.shape, dtype=bool)
    ghost, ghost_tint = np.zeros(mask.shape, dtype=bool), None
    if licorice is not None and len(licorice[0]):
        zs, fs = zbuffer(np.asarray(licorice[0], dtype=float), np.asarray(licorice[1], dtype=float), s, bounds)
        stick = zs > np.where(mask, z, -np.inf)
        # A stick that the surface covers is not lost: it shows through, pale, where it lies.
        ghost = np.isfinite(zs) & ~stick
        ghost_tint = np.asarray(licorice[2])[np.clip(fs, 0, None)]
        z = np.where(stick, zs, z)
        tint = np.where(stick[..., None], ghost_tint, tint)
        mask = mask | stick
        cut_face &= ~stick
    if not mask.any():
        return np.zeros((int(z.shape[0] / oversample), int(z.shape[1] / oversample), 4), dtype=np.uint8), None
    floor = float(z[mask].min()) - 2.0
    z0 = np.where(mask, z, floor).astype(np.float64)

    gy, gx = np.gradient(z0)
    nx, ny, nz = -gx * s, gy * s, np.ones_like(z0)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    lx, ly, lz = np.array(light) / np.linalg.norm(light)
    diffuse = np.clip((nx * lx + ny * ly + nz * lz) / norm, 0.0, 1.0)
    facing = nz / norm                                           # 1 where the surface looks at the viewer

    # A hollow is darker than a knob: the height against the height around it.
    around = blur(z0, max(1, int(round(3.0 * s))))
    hollow = np.clip(1.0 + 0.085 * (z0 - around), 0.55, 1.06)
    near = (z0 - floor) / max(1e-6, float(z0.max()) - floor)     # 1 at the front, 0 at the back

    shade = (0.50 + 0.50 * diffuse) * hollow * (0.80 + 0.20 * facing ** 0.6)
    shade = (1.0 - flat) * shade + flat * (0.86 + 0.14 * hollow)
    # (A stick keeps its own light: it is thin, and the shade of the hollow it lies in would swallow it.)
    shade = np.where(stick, 0.62 + 0.38 * diffuse, shade)
    rgb = tint * shade[..., None]
    haze = np.where(stick, 0.0, (1.0 - near) * 0.22 * (1.0 - flat))[..., None]
    rgb = rgb * (1.0 - haze) + haze

    # Lines where one part of the surface stands in front of another, and around the whole.
    step = np.zeros_like(z0)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        step = np.maximum(step, np.abs(z0 - np.roll(z0, (dy, dx), axis=(0, 1))))
    crease = np.clip((blur((step > 1.3).astype(float), 1, 1) * mask), 0.0, 1.0)
    inner = wear_edge(mask, max(1, int(round(0.16 * s))))
    rim = mask & ~inner
    ink = np.array([0.20, 0.23, 0.28])
    line = np.clip(0.55 * crease * (1.0 - 0.6 * flat) + rim * 0.85, 0.0, 1.0)[..., None]
    rgb = rgb * (1.0 - line) + ink * line
    if cut_face.any():                                           # the cut: flat, hatched, with a line around it
        rows, cols = np.indices(cut_face.shape)
        hatch = ((rows + cols) % max(4, int(round(0.95 * s)))) < max(1, int(round(0.16 * s)))
        paper = np.where((hatch & hatched)[..., None], np.array([0.66, 0.69, 0.73]), np.array([0.87, 0.88, 0.90]))
        rgb = np.where(cut_face[..., None], paper, rgb)
        around = cut_face & ~wear_edge(cut_face, max(1, int(round(0.10 * s)))) & wear_edge(mask, 2)
        rgb = np.where(around[..., None], ink, rgb)
    if stick.any():                                              # a dark edge, so a stick reads on any ground
        edge_of = stick & ~wear_edge(stick, max(1, int(round(0.045 * s))))
        dark = (0.62 * edge_of)[..., None]
        rgb = rgb * (1.0 - dark) + ink * dark
    if ghost.any() and ghost_tint is not None:
        through = (0.42 * ghost)[..., None]
        rgb = rgb * (1.0 - through) + ghost_tint * through

    h, w = (z.shape[0] // oversample) * oversample, (z.shape[1] // oversample) * oversample
    in_front = None
    if groups is not None:                                       # borders, as on a map: no two can overlap
        g = np.where(mask, np.asarray(groups)[np.clip(front, 0, None)], -2)
        g = np.where(stick, -3, g)
        edge = np.zeros(mask.shape, dtype=bool)
        for axis in (0, 1):
            other = np.roll(g, -1, axis=axis)
            differs = (g != other) & (g > -2) & (other > -2)
            edge |= differs | np.roll(differs, 1, axis=axis)
        border = np.clip(blur(edge.astype(float), max(1, int(round(0.10 * s))), 2) * 2.4, 0.0, 1.0)[..., None]
        rgb = rgb * (1.0 - 0.92 * border) + 0.92 * border
        in_front = g[:h:oversample, :w:oversample]

    rgba = np.dstack([np.clip(rgb, 0.0, 1.0), mask.astype(np.float64)])
    rgba[..., :3] *= rgba[..., 3:4]                              # premultiplied, for the averaging
    small = rgba[:h, :w].reshape(h // oversample, oversample, w // oversample, oversample, 4).mean(axis=(1, 3))
    alpha = small[..., 3:4]
    small[..., :3] = np.where(alpha > 1e-6, small[..., :3] / np.maximum(alpha, 1e-6), 0.0)
    return (np.clip(small, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), in_front


def wear_edge(mask: np.ndarray, px: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(px):
        m = out
        out = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    return out


def png(rgba: np.ndarray) -> bytes:
    """The picture as a PNG. Every row is stored as its difference from the pixel to the left (the "Sub"
    filter): a shaded surface changes little from one pixel to the next, and packs to about half that way."""
    h, w, _ = rgba.shape
    left = np.zeros_like(rgba)
    left[:, 1:] = rgba[:, :-1]
    rows = (rgba.astype(np.int16) - left.astype(np.int16)).astype(np.uint8)
    raw = b"".join(b"\x01" + rows[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
