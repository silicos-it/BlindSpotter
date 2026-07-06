# BlindSpotter

BlindSpotter triangulates ligand binding hotspots by turning residue-wise distance preferences into intersecting geometric constraints, yielding precise 3D coordinates for one or more binding sites.

## BlindSpotter: From PMF to Binding Hotspot

BlindSpotter is a Python tool designed to identify preferred binding hotspots of a ligand to a protein target from LiGaMD simulations. The workflow combines statistical reweighting, distance mapping, and geometric intersection analysis to pinpoint where the ligand most favorably resides.

#### 1. LiGaMD reweighting and PMF calculation

- For each MD frame, the distances between the ligand and every protein residue are computed.
- Using the LiGaMD boost potential applied on ligand–protein interactions, the probability distribution along each distance axis is reweighted.
- From these distributions, local minima of the potential of mean force (PMF) are identified. Each minimum corresponds to a preferred ligand distance relative to that residue.

#### 2. Sphere representation of residue–ligand preferences

- Each residue can define one or more spheres in 3D space:
  - Center = C-alpha position of the residue.
  - Radius = preferred ligand distance (from a PMF minimum).
- By default, the global PMF minimum is used. Secondary minima within `--pmfCutoff` kcal/mol of the deepest minimum are also accepted as candidate radii (up to `--maxMinima` per residue). This captures multi-modal distance preferences when the ligand samples more than one binding mode.
- For example, if ASP-145 shows the lowest PMF at 25 Å, the ligand’s most favorable position lies somewhere on the surface of a sphere with center at ASP-145 and radius 25 Å. If a second PMF minimum appears at 40 Å within the cutoff, a second candidate sphere is added for that residue.
- This procedure is repeated for all selected residues.

#### 3. Geometric intersection to localize hotspot(s)

- When all spheres share a common intersection, the least-squares fit converges to a single point:
  - 2 spheres intersect in a circle.
  - 3 spheres intersect in two points.
  - 4 or more spheres converge to a single point.
- With many residues contributing, their intersection collapses to a unique 3D coordinate representing one binding hotspot.
- When the ligand visits multiple distinct sites, residue spheres no longer share one intersection. BlindSpotter handles this with a RANSAC-based multi-hotspot detector (`FindHotspots`):
  1. Repeatedly sample minimal 4-sphere subsets and solve for a candidate intersection point.
  2. Count inlier residues whose candidate radii are consistent with that point (within `--tolerance` Å).
  3. Cluster and de-duplicate high-consensus hypotheses.
  4. Refine each surviving hotspot with a final least-squares fit on its inlier residues.

The number of hotspots is auto-detected by default. Use `--hotspots K` to request exactly K hotspots (the top K by consensus are reported).

#### 4. Output

The output PDB (default: `hotspot.pdb`) contains one `ATOM` record per detected hotspot:

- **Occupancy** encodes consensus strength (inlier count normalized to the largest hotspot).
- **B-factor** encodes fit quality (RMS residual of the sphere intersection).

A per-hotspot summary (coordinates, uncertainty, RMS, inlier count) is printed to stdout.

## Usage

```bash
python BlindSpotter.py \
  -c centers.pdb \
  -p topology.parm7 \
  -l 1 \
  -b boosts.txt \
  -t trajectories.txt \
  -o hotspot.pdb
```

### Multi-hotspot options

| Option | Default | Description |
|--------|---------|-------------|
| `--hotspots` | `0` | Number of hotspots to report. `0` = auto-detect. |
| `--tolerance` | `1.0` | Inlier distance tolerance (Å) when grouping spheres into a common intersection. |
| `--pmfCutoff` | `1.0` | PMF depth window (kcal/mol) above the global minimum for accepting secondary minima as candidate radii. |
| `--maxMinima` | `3` | Maximum number of candidate radii (PMF minima) extracted per residue. |
| `--minInliers` | `0` | Minimum residues required to accept a hotspot. `0` = automatic (`max(4, 10% of residues)`). |

### Other options

| Option | Default | Description |
|--------|---------|-------------|
| `-f` / `--fraction` | `1.0` | Fraction of centers to randomly select. |
| `-o` / `--output` | `hotspot.pdb` | Output PDB file name. |
| `--cutoff` | `10` | Histogram cutoff used by the PyReweighting script. |
| `--boostingType` | `single` | Boosting type: `single` or `dual`. |
