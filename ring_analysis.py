#!/usr/bin/env python3
"""
Ring analysis for 2D materials with periodic boundary conditions.

Detects rings using the Franzblau shortest-path algorithm, computes
ring size distribution, and produces a visualisation with colour-coded
polygons overlaid on the atomic structure.

Supports XYZ (with Lattice= in comment) and LAMMPS data files.

Usage:
    python ring_analysis.py structure.xyz --cutoff 1.9 --max-ring 12
    python ring_analysis.py structure.data --cutoff 1.85 -o my_rings.png
"""

import argparse
import sys
import re
import numpy as np
from collections import deque, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.colors import Normalize
from pathlib import Path


# ============================================================
# I/O
# ============================================================

def read_xyz(filename):
    """
    Read XYZ file. Attempts to parse lattice from the comment line
    (Lattice="ax ay az bx by bz cx cy cz" format, as used by ASE/QUIP).
    If no lattice is found, guesses from coordinate extents or asks.
    """
    with open(filename, 'r') as f:
        natoms = int(f.readline().strip())
        comment = f.readline().strip()

        symbols = []
        positions = []
        for _ in range(natoms):
            parts = f.readline().split()
            symbols.append(parts[0])
            positions.append([float(x) for x in parts[1:4]])

    positions = np.array(positions)
    cell = None

    # Try Lattice="..." (ASE / extended XYZ)
    match = re.search(r'[Ll]attice\s*=\s*"([^"]+)"', comment)
    if match:
        vals = [float(x) for x in match.group(1).split()]
        if len(vals) == 9:
            cell = np.array(vals).reshape(3, 3)
            print(f"  Parsed Lattice from comment line.")

    # Try LAMMPS-style bounds in comment: "xlo xhi ylo yhi zlo zhi"
    if cell is None:
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', comment)
        if len(nums) >= 6:
            vals = [float(x) for x in nums[:6]]
            lx = vals[1] - vals[0]
            ly = vals[3] - vals[2]
            lz = vals[5] - vals[4]
            if lx > 0 and ly > 0:
                cell = np.diag([lx, ly, lz])
                # Shift positions to start from origin
                positions[:, 0] -= vals[0]
                positions[:, 1] -= vals[2]
                positions[:, 2] -= vals[4]
                print(f"  Parsed box bounds from comment: {lx:.2f} x {ly:.2f} x {lz:.2f}")

    if cell is None:
        xlo, xhi = positions[:, 0].min(), positions[:, 0].max()
        ylo, yhi = positions[:, 1].min(), positions[:, 1].max()
        zlo, zhi = positions[:, 2].min(), positions[:, 2].max()
        print("No lattice found in XYZ comment line.")
        print(f"  Comment: '{comment}'")
        print(f"  Coordinate extents: x=[{xlo:.2f}, {xhi:.2f}], y=[{ylo:.2f}, {yhi:.2f}]")
        resp = input(
            "Enter cell as 'ax ay az bx by bz cx cy cz' or press Enter to use extents: "
        ).strip()
        if resp:
            vals = [float(x) for x in resp.split()]
            if len(vals) == 9:
                cell = np.array(vals).reshape(3, 3)
            elif len(vals) == 3:
                cell = np.diag(vals)
            elif len(vals) == 2:
                cell = np.diag([vals[0], vals[1], 20.0])
            else:
                print("Could not parse. Using extents.")
                cell = np.diag([xhi - xlo, yhi - ylo, max(zhi - zlo, 20.0)])
                positions[:, 0] -= xlo
                positions[:, 1] -= ylo
        else:
            cell = np.diag([xhi - xlo, yhi - ylo, max(zhi - zlo, 20.0)])
            positions[:, 0] -= xlo
            positions[:, 1] -= ylo

    return symbols, positions, cell


def read_lammps_data(filename):
    """Read LAMMPS data file (atomic, charge, full, or molecular style)."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    natoms = 0
    xlo, xhi = 0.0, 0.0
    ylo, yhi = 0.0, 0.0
    zlo, zhi = 0.0, 0.0
    xy, xz, yz = 0.0, 0.0, 0.0
    atom_style = None
    masses = {}

    atom_lines = []
    section = None

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            if section == 'Atoms' and len(atom_lines) >= natoms:
                section = None
            continue

        # Header fields
        if 'atoms' in stripped and section is None:
            try:
                natoms = int(stripped.split()[0])
            except ValueError:
                pass
            continue
        if 'xlo xhi' in stripped:
            parts = stripped.split()
            xlo, xhi = float(parts[0]), float(parts[1])
            continue
        if 'ylo yhi' in stripped:
            parts = stripped.split()
            ylo, yhi = float(parts[0]), float(parts[1])
            continue
        if 'zlo zhi' in stripped:
            parts = stripped.split()
            zlo, zhi = float(parts[0]), float(parts[1])
            continue
        if 'xy xz yz' in stripped:
            parts = stripped.split()
            xy, xz, yz = float(parts[0]), float(parts[1]), float(parts[2])
            continue

        # Section headers
        if stripped.startswith('Masses'):
            section = 'Masses'
            continue
        if stripped.startswith('Atoms'):
            section = 'Atoms'
            # Check for style hint: "Atoms # full" etc.
            m = re.search(r'#\s*(\w+)', stripped)
            if m:
                atom_style = m.group(1).lower()
            continue
        if stripped.startswith(('Bonds', 'Angles', 'Dihedrals', 'Impropers',
                                'Velocities', 'Pair Coeffs')):
            section = stripped.split()[0]
            continue

        # Parse section content
        if section == 'Masses':
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    masses[int(parts[0])] = float(parts[1])
                except ValueError:
                    pass
        elif section == 'Atoms':
            atom_lines.append(stripped)
            if len(atom_lines) >= natoms:
                section = None

    # Detect atom style from column count if not given
    if atom_style is None and atom_lines:
        # Strip image flags (integers at end)
        ncols = len(atom_lines[0].split())
        if ncols == 5:
            atom_style = 'atomic'
        elif ncols == 6:
            atom_style = 'charge'
        elif ncols >= 7:
            atom_style = 'full'

    # Column indices for (id, type, x, y, z)
    style_map = {
        'atomic':    (0, 1, 2, 3, 4),
        'charge':    (0, 1, 3, 4, 5),    # id type q x y z
        'molecular': (0, 2, 3, 4, 5),    # id mol type x y z
        'full':      (0, 2, 4, 5, 6),    # id mol type q x y z
    }
    cols = style_map.get(atom_style, (0, 1, 2, 3, 4))

    positions = np.zeros((natoms, 3))
    types = np.zeros(natoms, dtype=int)

    for line in atom_lines:
        parts = line.split()
        idx = int(parts[cols[0]]) - 1
        types[idx] = int(parts[cols[1]])
        positions[idx] = [float(parts[cols[2]]),
                          float(parts[cols[3]]),
                          float(parts[cols[4]])]

    cell = np.array([
        [xhi - xlo, 0.0,       0.0],
        [xy,        yhi - ylo,  0.0],
        [xz,        yz,         zhi - zlo]
    ])

    # Map types to symbols via masses (rounded to nearest integer amu)
    amu_to_sym = {
        1: 'H', 4: 'He', 7: 'Li', 9: 'Be', 11: 'B', 12: 'C', 14: 'N',
        16: 'O', 19: 'F', 23: 'Na', 24: 'Mg', 27: 'Al', 28: 'Si', 31: 'P',
        32: 'S', 35: 'Cl', 40: 'Ar', 48: 'Ti', 52: 'Cr', 55: 'Mn', 56: 'Fe',
        59: 'Co', 58: 'Ni', 64: 'Cu', 65: 'Zn', 70: 'Ga', 73: 'Ge', 75: 'As',
        79: 'Se', 80: 'Br', 96: 'Mo', 184: 'W', 34: 'S', 128: 'Te'
    }
    symbols = []
    for t in types:
        if t in masses:
            sym = amu_to_sym.get(round(masses[t]), f'Type{t}')
        else:
            sym = f'Type{t}'
        symbols.append(sym)

    return symbols, positions, cell


def read_structure(filename):
    """Detect format and read structure."""
    path = Path(filename)
    suffix = path.suffix.lower()

    if suffix == '.xyz':
        return read_xyz(filename)
    if suffix in ('.data', '.lmp', '.lammps'):
        return read_lammps_data(filename)

    # Fallback: check if first line is an integer (XYZ) or text (LAMMPS)
    with open(filename, 'r') as f:
        first = f.readline().strip()
    try:
        int(first)
        return read_xyz(filename)
    except ValueError:
        return read_lammps_data(filename)


# ============================================================
# Graph construction with PBC
# ============================================================

def build_graph(positions, cell, cutoff, pbc_dims=(True, True, False)):
    """
    Build neighbour list with periodic boundary conditions.

    Returns dict: neighbors[i] = [(j, (sx, sy, sz)), ...]
    where (sx, sy, sz) is the integer image shift for the bond i->j.
    """
    natoms = len(positions)
    inv_cell = np.linalg.inv(cell)

    # Fractional coordinates
    frac = positions @ inv_cell

    neighbors = defaultdict(list)

    # Periodic image offsets to check
    ranges = [[-1, 0, 1] if pbc else [0] for pbc in pbc_dims]
    images = []
    for ix in ranges[0]:
        for iy in ranges[1]:
            for iz in ranges[2]:
                images.append((ix, iy, iz))

    cutoff_sq = cutoff * cutoff

    for i in range(natoms):
        for j in range(i, natoms):
            for img in images:
                if i == j and img == (0, 0, 0):
                    continue

                dfrac = frac[j] - frac[i] + np.array(img, dtype=float)
                dr = dfrac @ cell
                dist_sq = np.dot(dr, dr)

                if dist_sq < cutoff_sq:
                    neighbors[i].append((j, img))
                    neg_img = (-img[0], -img[1], -img[2])
                    neighbors[j].append((i, neg_img))

    return neighbors


# ============================================================
# Franzblau shortest-path ring detection
# ============================================================

def find_shortest_ring_through_bond(i, j, shift_ij, neighbors, max_ring):
    """
    Remove bond (i -> j, shift_ij), then BFS from i at image (0,0,0)
    looking for j at image shift_ij. Returns the ring path
    [(atom, image), ...] or None if no ring within max_ring.
    """
    target = (j, shift_ij)
    start = (i, (0, 0, 0))

    # BFS
    queue = deque()
    queue.append((start, [start]))
    visited = {start}

    while queue:
        (cur_atom, cur_img), path = queue.popleft()

        if len(path) >= max_ring:
            continue

        for (nb, nb_shift) in neighbors[cur_atom]:
            # Skip the removed bond: i(0,0,0) -> j(shift_ij)
            if (cur_atom == i
                    and cur_img == (0, 0, 0)
                    and nb == j
                    and nb_shift == shift_ij):
                continue

            new_img = (cur_img[0] + nb_shift[0],
                       cur_img[1] + nb_shift[1],
                       cur_img[2] + nb_shift[2])
            new_state = (nb, new_img)

            if new_state == target:
                return path + [new_state]

            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [new_state]))

    return None


def canonicalise_ring(atom_indices):
    """Canonical form: lexicographically smallest rotation or reflection."""
    n = len(atom_indices)
    fwd = list(atom_indices)
    rev = list(reversed(atom_indices))
    candidates = []
    for seq in (fwd, rev):
        for k in range(n):
            candidates.append(tuple(seq[k:] + seq[:k]))
    return min(candidates)


def detect_all_rings(neighbors, natoms, max_ring=12, verbose=True):
    """
    Franzblau shortest-path ring detection.

    Returns dict: {canonical_ring_tuple: [(atom, image), ...], ...}
    """
    processed = set()
    rings = {}

    total_bonds = sum(len(neighbors[i]) for i in range(natoms)) // 2
    done = 0

    for i in range(natoms):
        for (j, shift) in neighbors[i]:
            # Canonical undirected edge
            neg_shift = (-shift[0], -shift[1], -shift[2])
            edge_id = (min(i, j), max(i, j),
                       shift if i < j else neg_shift)
            if edge_id in processed:
                continue
            processed.add(edge_id)
            done += 1

            # Search both directions to catch both rings sharing this bond
            for (a, b, s_ab) in [(i, j, shift), (j, i, neg_shift)]:
                path = find_shortest_ring_through_bond(
                    a, b, s_ab, neighbors, max_ring
                )
                if path is not None:
                    atoms = tuple(s[0] for s in path)
                    canon = canonicalise_ring(atoms)
                    if canon not in rings:
                        rings[canon] = path

            if verbose and done % 500 == 0:
                print(f"    processed {done}/{total_bonds} bonds, "
                      f"{len(rings)} rings so far...")

    return rings


# ============================================================
# Visualisation
# ============================================================

def unwrap_ring_coordinates(ring_path, positions, cell):
    """Compute unwrapped 2D coordinates for a ring, accounting for PBC."""
    coords = []
    for (atom_idx, img) in ring_path:
        shift = np.array(img, dtype=float)
        pos = positions[atom_idx] + shift @ cell
        coords.append(pos[:2])
    return np.array(coords)


def plot_ring_structure(positions, cell, rings, symbols, output='rings.png',
                        show_distribution=True):
    """
    Plot the 2D structure with ring polygons coloured by ring size
    and (optionally) the ring size distribution.
    """
    ring_data = []
    for canon, path in rings.items():
        size = len(path)
        coords = unwrap_ring_coordinates(path, positions, cell)
        ring_data.append((size, coords))

    if not ring_data:
        print("No rings found, nothing to plot.")
        return

    sizes_all = [r[0] for r in ring_data]
    min_size = min(sizes_all)
    max_size = max(sizes_all)

    cmap = plt.cm.viridis
    norm = Normalize(vmin=min_size - 0.5, vmax=max_size + 0.5)

    if show_distribution:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(16, 7),
            gridspec_kw={'width_ratios': [2.5, 1]}
        )
    else:
        fig, ax1 = plt.subplots(figsize=(12, 10))
        ax2 = None

    # ---- Ring polygons ----
    # Sort so smaller rings are drawn on top of larger ones
    ring_data.sort(key=lambda x: -x[0])

    for size, coords in ring_data:
        colour = cmap(norm(size))
        poly = MplPolygon(coords, closed=True,
                          facecolor=colour, edgecolor='k',
                          linewidth=0.3, alpha=0.7, zorder=2)
        ax1.add_patch(poly)

    # ---- Atoms ----
    pos2d = positions[:, :2]
    atom_cmap = {
        'C': '#333333', 'B': '#E74C3C', 'N': '#3498DB', 'O': '#E67E22',
        'H': '#95A5A6', 'S': '#F1C40F', 'Mo': '#8E44AD', 'W': '#7F8C8D',
        'Se': '#E67E22', 'Si': '#2ECC71', 'P': '#FF6F61', 'Ge': '#1ABC9C'
    }
    unique_sym = sorted(set(symbols))
    for sym in unique_sym:
        mask = np.array([s == sym for s in symbols])
        c = atom_cmap.get(sym, '#888888')
        ax1.scatter(pos2d[mask, 0], pos2d[mask, 1],
                    s=10, c=c, edgecolors='k', linewidths=0.2,
                    zorder=5, label=sym)

    # ---- Cell boundary ----
    cell2d = cell[:2, :2]
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    box = corners @ cell2d
    ax1.plot(box[:, 0], box[:, 1], 'k-', linewidth=1.5, zorder=6)

    ax1.set_aspect('equal')
    ax1.set_xlabel(r'$x$ ($\mathrm{\AA}$)', fontsize=12)
    ax1.set_ylabel(r'$y$ ($\mathrm{\AA}$)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.set_title('Ring structure', fontsize=13)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1,
                        ticks=range(min_size, max_size + 1),
                        shrink=0.8)
    cbar.set_label('Ring size', fontsize=11)

    # ---- Distribution ----
    counts = defaultdict(int)
    for s in sizes_all:
        counts[s] += 1

    size_list = sorted(counts.keys())
    count_list = [counts[s] for s in size_list]
    total = sum(count_list)

    if ax2 is not None:
        colours = [cmap(norm(s)) for s in size_list]
        ax2.bar(size_list, count_list, color=colours,
                edgecolor='k', linewidth=0.5)
        ax2.set_xlabel('Ring size', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Ring size distribution', fontsize=13)
        ax2.set_xticks(size_list)

        # Annotate counts on bars
        for s, c in zip(size_list, count_list):
            ax2.text(s, c + total * 0.01, str(c),
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {output}")

    # Print table
    print(f"\nRing size distribution ({total} rings total)")
    print("-" * 35)
    for s, c in zip(size_list, count_list):
        pct = c / total * 100
        print(f"  {s:2d}-membered:  {c:5d}   ({pct:5.1f}%)")
    print("-" * 35)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Ring analysis for 2D materials with periodic '
                    'boundary conditions (Franzblau shortest-path rings).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ring_analysis.py graphene.xyz
  python ring_analysis.py amorphous_BN.data --cutoff 1.85 --max-ring 15
  python ring_analysis.py structure.xyz --cutoff 1.9 -o rings_output.png
        """
    )
    parser.add_argument('input', help='Structure file (XYZ or LAMMPS data)')
    parser.add_argument('--cutoff', type=float, default=1.9,
                        help='First-neighbour bond cutoff in angstrom '
                             '(default: 1.9)')
    parser.add_argument('--max-ring', type=int, default=12,
                        help='Maximum ring size to search for (default: 12)')
    parser.add_argument('-o', '--output', default='rings.png',
                        help='Output figure filename (default: rings.png)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting, just print distribution')
    args = parser.parse_args()

    # --- Read ---
    print(f"Reading {args.input} ...")
    symbols, positions, cell = read_structure(args.input)
    natoms = len(positions)
    species = sorted(set(symbols))
    print(f"  {natoms} atoms, species: {species}")
    print(f"  Cell vectors:")
    for i, label in enumerate('abc'):
        v = cell[i]
        print(f"    {label} = ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})")

    # --- Build graph ---
    print(f"\nBuilding neighbour graph (cutoff = {args.cutoff} A, "
          f"PBC in x,y) ...")
    neighbors = build_graph(positions, cell, args.cutoff,
                            pbc_dims=(True, True, False))

    coordinations = np.array([len(neighbors[i]) for i in range(natoms)])
    print(f"  Coordination: min={coordinations.min()}, "
          f"max={coordinations.max()}, "
          f"mean={coordinations.mean():.2f}")

    n_under = np.sum(coordinations < 3)
    n_over = np.sum(coordinations > 3)
    if n_under > 0:
        print(f"  WARNING: {n_under} atom(s) with coordination < 3 "
              f"(dangling bonds or cutoff too small)")
    if n_over > 0:
        print(f"  WARNING: {n_over} atom(s) with coordination > 3 "
              f"(cutoff might be too large)")

    nbonds = sum(len(neighbors[i]) for i in range(natoms)) // 2
    print(f"  Total bonds: {nbonds}")

    # --- Ring detection ---
    print(f"\nDetecting rings (max size = {args.max_ring}) ...")
    rings = detect_all_rings(neighbors, natoms, args.max_ring, verbose=True)
    print(f"  Found {len(rings)} unique rings.")

    if len(rings) == 0:
        print("No rings detected. Check your cutoff and structure.")
        sys.exit(0)

    # --- Plot ---
    if not args.no_plot:
        print("\nPlotting ...")
        plot_ring_structure(positions, cell, rings, symbols,
                            output=args.output)
    else:
        # Just print distribution
        sizes = [len(path) for path in rings.values()]
        counts = defaultdict(int)
        for s in sizes:
            counts[s] += 1
        total = sum(counts.values())
        print(f"\nRing size distribution ({total} rings total)")
        print("-" * 35)
        for s in sorted(counts):
            c = counts[s]
            print(f"  {s:2d}-membered:  {c:5d}   ({c/total*100:5.1f}%)")
        print("-" * 35)


if __name__ == '__main__':
    main()
