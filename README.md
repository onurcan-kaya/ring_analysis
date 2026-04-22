# Ring Analysis for 2D Materials

Shortest-path ring detection for 2D materials with periodic boundary conditions, following the Franzblau algorithm (Franzblau, 1991). Computes ring size distributions and produces colour-coded polygon visualisations overlaid on the atomic structure.

Designed for sp2-bonded 2D networks such as graphene, h-BN, amorphous carbon, amorphous boron nitride and similar systems.

## Requirements

* Python 3.7+
* NumPy
* Matplotlib

No other dependencies. Does not require ASE, networkx or any atomistic simulation package.

## Usage

```bash
python ring\_analysis.py structure.xyz
python ring\_analysis.py structure.data --cutoff 1.85 --max-ring 15 -o output.png
python ring\_analysis.py structure.xyz --no-plot
```

### Arguments

|Argument|Default|Description|
|-|-|-|
|`input`|(required)|Structure file, XYZ or LAMMPS data format|
|`--cutoff`|1.9|First-neighbour bond cutoff in angstrom|
|`--max-ring`|12|Maximum ring size to search for|
|`-o`, `--output`|`rings.png`|Output figure filename|
|`--no-plot`|off|Skip plotting, only print the distribution table|

## Supported input formats

### XYZ

Extended XYZ with a `Lattice="..."` field in the comment line (second line), as written by ASE, QUIP and similar tools:

```
50
Lattice="12.30 0.00 0.00 6.15 10.65 0.00 0.00 0.00 20.00" Properties=species:S:1:pos:R:3
C  0.000000  0.000000  10.000000
C  0.819837  0.473333  10.000000
...
```

If no `Lattice` tag is found, the script attempts to parse numeric bounds from the comment line and otherwise prompts interactively.

### LAMMPS data

Standard LAMMPS data files with atom styles `atomic`, `charge`, `molecular` or `full`. The style is auto-detected from the column count or from the `# style` hint after the `Atoms` header. Atom types are mapped to element symbols via the `Masses` section (nearest integer AMU lookup). Triclinic cells (xy, xz, yz tilts) are supported.

## Output

The script prints a coordination summary and a ring size distribution table to stdout:

```
Ring size distribution (25 rings total)
-----------------------------------
   6-membered:     25   (100.0%)
-----------------------------------
```

The figure contains two panels: the ring structure map (left) and a bar chart of the distribution (right).

