#!/usr/bin/env python

import argparse
import sys
import os
import random
import numpy as np

try:
    from scipy.optimize import least_squares
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False




VERSION = "1.0"
CPPTRAJ = "/usr/local/amber24/bin/cpptraj"



# _residuals function
# ###################

def _residuals(x: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Residuals f_i(x) = ||x - c_i|| - r_i."""
    diffs = x[None, :] - centers  # (N,3)
    dists = np.linalg.norm(diffs, axis=1)  # (N,)
    return dists - radii



# _jacobian function
# ##################

def _jacobian(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Jacobian J_ij = ∂f_i/∂x_j = (x_j - c_ij)/||x - c_i||.
    If x coincides with a center, regularize to avoid division by zero.
    """
    diffs = x[None, :] - centers  # (N,3)
    dists = np.linalg.norm(diffs, axis=1)
    eps = 1e-12
    dists = np.where(dists < eps, eps, dists)
    return diffs / dists[:, None]  # (N,3)



# __covariance_from_jacobian function
# ###################################

def _covariance_from_jacobian(x: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, float]:
    """Return covariance matrix (3x3) and residual variance s2."""
    r = _residuals(x, centers, radii)
    J = _jacobian(x, centers)
    JTJ = J.T @ J
    # Regularize if near-singular
    try:
        JTJ_inv = np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        JTJ_inv = np.linalg.pinv(JTJ)
    N = centers.shape[0]
    dof = max(N - 3, 1)
    s2 = float((r @ r) / dof)
    cov = s2 * JTJ_inv
    return cov, s2



# _gauss_newton function
# ######################

def _gauss_newton(
    x0: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[np.ndarray, bool]:
    """
    Simple damped Gauss-Newton fallback optimizer.
    Returns (x, converged).
    """
    x = x0.astype(float).copy()
    lam = 1e-3  # small damping
    for _ in range(max_iter):
        r = _residuals(x, centers, radii)  # (N,)
        J = _jacobian(x, centers)         # (N,3)
        JTJ = J.T @ J
        g = J.T @ r
        try:
            step = -np.linalg.solve(JTJ + lam * np.eye(3), g)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(JTJ + lam * np.eye(3)) @ g
        x_new = x + step
        if np.linalg.norm(step) < tol * (1.0 + np.linalg.norm(x)):
            return x_new, True
        # simple acceptance rule
        if np.sum(_residuals(x_new, centers, radii) ** 2) < np.sum(r ** 2):
            # good step; slightly decrease damping
            lam *= 0.7
            x = x_new
        else:
            # reject; increase damping
            lam *= 2.0
    return x, False



# IntersectSpheres function
# ##########################

def IntersectSpheres(
    centers: np.ndarray,
    radii: np.ndarray,
    x0: np.ndarray | None = None,
    use_scipy: bool | None = None,
) -> dict:
    """
    Compute the best-fit intersection point of spheres.

    Parameters
    ----------
    centers : (N,3) array
        Sphere centers.
    radii : (N,) array
        Sphere radii (same units as centers).
    x0 : (3,) array, optional
        Initial guess. Defaults to centroid of centers.
    use_scipy : bool, optional
        Force using SciPy or the fallback Gauss-Newton. Default: use SciPy if available.

    Returns
    -------
    result : dict with keys
        - 'x'        : (3,) best-fit coordinates
        - 'cov'      : (3,3) covariance matrix
        - 'sigma'    : (3,) 1-sigma uncertainty per axis (sqrt of diag(cov))
        - 'rms'      : float, RMS residual
        - 'converged': bool, optimizer convergence flag
        - 'method'   : str, 'scipy' or 'gauss-newton'
        - 'dof'      : int, degrees of freedom (N-3)
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must be an (N,3) array")
    if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
        raise ValueError("radii must be an (N,) array matching centers")
    N = centers.shape[0]
    if N < 4:
        raise ValueError("At least 4 spheres are required to constrain a 3D point; you provided N=%d." % N)

    if x0 is None:
        x0 = centers.mean(axis=0)  # reasonable starting guess

    if use_scipy is None:
        use_scipy = _HAVE_SCIPY

    if use_scipy:
        # Robust trust-region reflective least-squares
        res = least_squares(
            fun=_residuals,
            x0=x0,
            jac=lambda x, centers, radii: _jacobian(x, centers),
            args=(centers, radii),
            method="lm",  # Levenberg–Marquardt (good for small problems)
        )
        x = res.x
        converged = res.success
        method = "scipy"
    else:
        x, converged = _gauss_newton(x0, centers, radii)
        method = "gauss-newton"

    cov, s2 = _covariance_from_jacobian(x, centers, radii)
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    rms = float(np.sqrt(np.mean(_residuals(x, centers, radii) ** 2)))
    dof = max(N - 3, 1)

    return {
        "x": x,
        "cov": cov,
        "sigma": sigma,
        "rms": rms,
        "converged": bool(converged),
        "method": method,
        "dof": dof,
    }



# bootstrap_uncertainty function
# ##############################

def bootstrap_uncertainty(
    centers: np.ndarray,
    radii: np.ndarray,
    n_boot: int = 500,
    random_state: int | None = 0,
    use_scipy: bool | None = None,
) -> dict:
    """
    Optional: non-parametric bootstrap to assess uncertainty robustness.
    Resamples the N spheres with replacement and re-fits each time.

    Returns dict with keys:
        - 'mean' : (3,) bootstrap mean of solutions
        - 'std'  : (3,) bootstrap standard deviation per axis
        - 'samples' : (n_boot, 3) all bootstrap solutions
    """
    rng = np.random.default_rng(random_state)
    N = centers.shape[0]
    sols = []
    base = IntersectSpheres(centers, radii, use_scipy=use_scipy)["x"]
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        cs = centers[idx]
        rs = radii[idx]
        res = IntersectSpheres(cs, rs, x0=base, use_scipy=use_scipy)
        sols.append(res["x"])
    sols = np.array(sols)
    return {"mean": sols.mean(axis=0), "std": sols.std(axis=0, ddof=1), "samples": sols}



# ParsePMF function
# #################

def ParsePMF(pmf_file: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a two-column PMF .xvg file and return (dists, pmfs) arrays."""
    dists = []
    pmfs = []

    try:
        fi = open(pmf_file, "r")
    except OSError:
        return None

    with fi:
        for LINE in fi.readlines():
            LINE = LINE.strip()
            if LINE == "" or (len(LINE) > 0 and (LINE[0] == "#" or LINE[0] == "@")): continue
            FIELDS = LINE.split()
            if len(FIELDS) < 2:
                continue
            try:
                d = float(FIELDS[0])
                p = float(FIELDS[1])
            except ValueError:
                continue
            dists.append(d)
            pmfs.append(p)

    if len(dists) == 0:
        return None

    return np.asarray(dists, dtype=float), np.asarray(pmfs, dtype=float)



# ExtractRadii function
# #####################

def ExtractRadii(
    dists: np.ndarray,
    pmfs: np.ndarray,
    pmf_cutoff: float,
    max_minima: int,
) -> list[float]:
    """
    Return the distances (radii) at local minima whose PMF lies within
    pmf_cutoff of the global minimum.

    The global minimum is always returned first. The remaining local minima are
    sorted by PMF depth (deepest first) and the result is truncated to at most
    max_minima entries.
    """
    n = len(pmfs)
    if n == 0:
        return []

    gmin = int(np.argmin(pmfs))
    baseline = float(pmfs[gmin])

    minima = [(float(pmfs[gmin]), float(dists[gmin]))]
    for i in range(n):
        if i == gmin:
            continue
        left = pmfs[i - 1] if i > 0 else np.inf
        right = pmfs[i + 1] if i < n - 1 else np.inf
        if pmfs[i] <= left and pmfs[i] <= right and (pmfs[i] - baseline) <= pmf_cutoff:
            minima.append((float(pmfs[i]), float(dists[i])))

    minima.sort(key=lambda t: t[0])

    radii: list[float] = []
    for _, d in minima:
        if all(abs(d - r) > 1e-9 for r in radii):
            radii.append(d)
        if len(radii) >= max_minima:
            break

    return radii



def _local_minima_indices(pmfs: np.ndarray) -> list[int]:
    """Return indices of local minima in a 1D PMF profile."""
    n = len(pmfs)
    indices = []
    for i in range(n):
        left = pmfs[i - 1] if i > 0 else np.inf
        right = pmfs[i + 1] if i < n - 1 else np.inf
        if pmfs[i] <= left and pmfs[i] <= right:
            indices.append(i)
    return indices



# ScorePMFWellDefined function
# ############################

def ScorePMFWellDefined(
    dists: np.ndarray,
    pmfs: np.ndarray,
    pmf_cutoff: float,
    basin_cutoff: float,
    max_basin_width: float,
    min_barrier: float,
    max_competing_minima: int,
) -> dict:
    """
    Score whether a PMF profile has a well-defined global minimum.

    Returns a dict with passes (bool), metrics, and failure reasons.
    """
    gmin = int(np.argmin(pmfs))
    baseline = float(pmfs[gmin])
    threshold = baseline + basin_cutoff

    # Basin width: span where PMF stays within basin_cutoff of the minimum
    left = gmin
    while left > 0 and pmfs[left] <= threshold:
        left -= 1
    right = gmin
    while right < len(pmfs) - 1 and pmfs[right] <= threshold:
        right += 1
    basin_width = float(dists[right] - dists[left])

    # Competing minima within pmf_cutoff of the global minimum
    competing = []
    for i in _local_minima_indices(pmfs):
        if i == gmin:
            continue
        if (pmfs[i] - baseline) <= pmf_cutoff:
            competing.append(i)

    n_competing = len(competing)

    # Barrier height to the nearest competing minimum (in index space)
    barrier_height = np.inf
    for ci in competing:
        lo, hi = (gmin, ci) if gmin < ci else (ci, gmin)
        if hi > lo:
            barrier = float(np.max(pmfs[lo:hi + 1]) - baseline)
            barrier_height = min(barrier_height, barrier)

    reasons = []
    if basin_width > max_basin_width:
        reasons.append("flat_basin")
    if n_competing > max_competing_minima:
        reasons.append("multi_minimum")
    if n_competing > 0 and barrier_height < min_barrier:
        reasons.append("low_barrier")

    return {
        "passes": len(reasons) == 0,
        "basin_width": basin_width,
        "barrier_height": float(barrier_height) if np.isfinite(barrier_height) else None,
        "n_competing_minima": n_competing,
        "reasons": reasons,
    }



# FindSpatialNeighbors function
# #############################

def FindSpatialNeighbors(
    centers: np.ndarray,
    mode: str = "distance",
    cutoff: float = 8.0,
    k: int = 5,
) -> list[list[int]]:
    """Return spatial neighbor index lists for each residue."""
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    neighbors: list[list[int]] = [[] for _ in range(n)]

    if mode == "distance":
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if float(np.linalg.norm(centers[i] - centers[j])) <= cutoff:
                    neighbors[i].append(j)
    elif mode == "knn":
        k_eff = min(k, n - 1)
        if k_eff <= 0:
            return neighbors
        for i in range(n):
            dists = [float(np.linalg.norm(centers[i] - centers[j])) for j in range(n) if j != i]
            idxs = [j for j in range(n) if j != i]
            order = np.argsort(dists)[:k_eff]
            neighbors[i] = [idxs[o] for o in order]
    else:
        raise ValueError("neighbor mode must be 'distance' or 'knn'")

    return neighbors



def _pmf_profile_on_grid(dists: np.ndarray, pmfs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a PMF onto a distance grid and min-subtract for shape comparison."""
    profile = np.interp(grid, dists, pmfs, left=pmfs[0], right=pmfs[-1])
    profile = profile - np.min(profile)
    return profile



# ScorePMFNeighborSimilarity function
# ###################################

def ScorePMFNeighborSimilarity(
    index: int,
    profiles: list[tuple[np.ndarray, np.ndarray]],
    neighbor_lists: list[list[int]],
    grid: np.ndarray,
    min_neighbors: int = 2,
    min_corr: float = 0.6,
) -> dict:
    """
    Score PMF similarity between a residue and its spatial neighbors.

    If the residue has fewer than min_neighbors spatial neighbors, the neighbor
    test is skipped (applicable=None).
    """
    neighbors = neighbor_lists[index]
    if len(neighbors) < min_neighbors:
        return {"applicable": False, "passes": None, "corr": None, "reasons": []}

    profile_i = _pmf_profile_on_grid(profiles[index][0], profiles[index][1], grid)
    neighbor_profiles = [
        _pmf_profile_on_grid(profiles[j][0], profiles[j][1], grid) for j in neighbors
    ]
    median_profile = np.median(neighbor_profiles, axis=0)

    if np.std(profile_i) < 1e-12 or np.std(median_profile) < 1e-12:
        corr = 1.0 if np.allclose(profile_i, median_profile) else 0.0
    else:
        corr = float(np.corrcoef(profile_i, median_profile)[0, 1])

    reasons = []
    if corr < min_corr:
        reasons.append("neighbor_outlier")

    return {
        "applicable": True,
        "passes": corr >= min_corr,
        "corr": corr,
        "reasons": reasons,
    }



# FilterResiduesByPMF function
# ############################

def FilterResiduesByPMF(
    resids: list[int],
    centers: list,
    profiles: list[tuple[np.ndarray, np.ndarray]],
    radii_per_res: list[list[float]],
    mode: str,
    pmf_cutoff: float,
    basin_cutoff: float,
    max_basin_width: float,
    min_barrier: float,
    max_competing_minima: int,
    neighbor_mode: str,
    neighbor_dist: float,
    neighbor_k: int,
    min_neighbor_corr: float,
    min_neighbors: int,
) -> tuple[list[int], np.ndarray, list[list[float]], list[dict]]:
    """
    Filter residues by PMF quality and/or neighbor similarity.

    mode: none | quality | neighbors | both | either
    """
    n = len(resids)
    centers_arr = np.asarray(centers, dtype=float)

    if mode == "none":
        report = [{"resid": resids[i], "kept": True, "reasons": []} for i in range(n)]
        return resids, centers_arr, radii_per_res, report

    neighbor_lists = FindSpatialNeighbors(centers_arr, neighbor_mode, neighbor_dist, neighbor_k)

    d_min = min(float(p[0].min()) for p in profiles)
    d_max = max(float(p[0].max()) for p in profiles)
    grid = np.arange(d_min, d_max + 1.0, 1.0)

    quality_scores = [
        ScorePMFWellDefined(
            profiles[i][0], profiles[i][1],
            pmf_cutoff, basin_cutoff, max_basin_width, min_barrier, max_competing_minima,
        )
        for i in range(n)
    ]
    neighbor_scores = [
        ScorePMFNeighborSimilarity(
            i, profiles, neighbor_lists, grid, min_neighbors, min_neighbor_corr,
        )
        for i in range(n)
    ]

    kept_resids = []
    kept_centers = []
    kept_radii = []
    report = []

    for i in range(n):
        q_pass = quality_scores[i]["passes"]
        n_score = neighbor_scores[i]
        n_pass = n_score["passes"]
        n_applicable = n_score["applicable"]

        if mode == "quality":
            keep = q_pass
        elif mode == "neighbors":
            keep = n_applicable and n_pass is True
        elif mode == "both":
            keep = q_pass and (not n_applicable or n_pass is True)
        elif mode == "either":
            keep = q_pass or (n_applicable and n_pass is True)
        else:
            raise ValueError("pmfFilter must be one of: none, quality, neighbors, both, either")

        reasons = []
        if not keep:
            if not q_pass:
                reasons.extend(quality_scores[i]["reasons"])
            if n_applicable and n_pass is False:
                reasons.extend(n_score["reasons"])
            elif mode == "neighbors" and not n_applicable:
                reasons.append("insufficient_neighbors")

        entry = {
            "resid": resids[i],
            "kept": keep,
            "quality_pass": q_pass,
            "neighbor_pass": n_pass,
            "neighbor_applicable": n_applicable,
            "basin_width": quality_scores[i]["basin_width"],
            "barrier_height": quality_scores[i]["barrier_height"],
            "n_competing_minima": quality_scores[i]["n_competing_minima"],
            "neighbor_corr": n_score["corr"],
            "reasons": reasons,
        }
        report.append(entry)

        if keep:
            kept_resids.append(resids[i])
            kept_centers.append(centers[i])
            kept_radii.append(radii_per_res[i])

    n_kept = len(kept_resids)
    n_rejected = n - n_kept
    print("")
    print("PMF filter (%s): kept %d / %d residues" % (mode, n_kept, n))
    if n_rejected > 0:
        print("Rejected residues:")
        for entry in report:
            if not entry["kept"]:
                print("  resid %d: %s" % (entry["resid"], ", ".join(entry["reasons"]) or "filtered"))

    return kept_resids, np.asarray(kept_centers, dtype=float), kept_radii, report



# FindHotspots function
# #####################

def FindHotspots(
    centers: np.ndarray,
    radii_per_res: list[list[float]],
    tol: float = 1.0,
    min_inliers: int | None = None,
    n_iter: int = 2000,
    k: int | None = None,
    random_state: int | None = 0,
) -> list[dict]:
    """
    Detect one or more sphere-intersection hotspots via a RANSAC scheme.

    Each residue provides a center and a list of candidate radii (PMF minima).
    Rather than forcing all spheres into a single least-squares fit, this
    repeatedly samples minimal 4-sphere subsets, solves for a candidate point,
    and counts how many residues are consistent with that point (an inlier is a
    residue having at least one candidate radius r with |||x - c|| - r| <= tol).

    High-consensus hypotheses are clustered spatially and de-duplicated by inlier
    overlap, then each surviving hotspot is refined with IntersectSpheres on its
    inliers. Returns a list of hotspot dicts sorted by descending inlier count.

    Parameters
    ----------
    centers : (N,3) array
        Residue C-alpha positions.
    radii_per_res : list of lists
        radii_per_res[i] holds the candidate radii for residue i.
    tol : float
        Inlier distance tolerance (Angstrom).
    min_inliers : int, optional
        Minimum residues required to accept a hotspot. Defaults to
        max(4, 10% of residues).
    n_iter : int
        Number of RANSAC iterations.
    k : int, optional
        If set (>0), keep only the top-k hotspots by consensus.
    random_state : int, optional
        Seed for reproducible sampling.
    """
    centers = np.asarray(centers, dtype=float)
    n_res = centers.shape[0]

    if min_inliers is None:
        min_inliers = max(4, int(0.1 * n_res))
    min_inliers = max(4, int(min_inliers))

    rng = random.Random(random_state)

    valid_res = [i for i in range(n_res) if len(radii_per_res[i]) > 0]
    if len(valid_res) < 4:
        return []

    def best_residual(i: int, x: np.ndarray) -> float | None:
        """Smallest |||x - c_i|| - r| over residue i's candidate radii within tol."""
        dist = float(np.linalg.norm(x - centers[i]))
        best = None
        for r in radii_per_res[i]:
            resid = abs(dist - r)
            if resid <= tol and (best is None or resid < best):
                best = resid
        return best

    def inlier_set(x: np.ndarray) -> set[int]:
        return {i for i in valid_res if best_residual(i, x) is not None}

    hypotheses = []  # (x, inlier_set)
    for _ in range(n_iter):
        sample = rng.sample(valid_res, 4)
        sub_centers = np.array([centers[i] for i in sample])
        sub_radii = np.array([rng.choice(radii_per_res[i]) for i in sample])
        try:
            res = IntersectSpheres(sub_centers, sub_radii)
        except Exception:
            continue
        x = res["x"]
        if not np.all(np.isfinite(x)):
            continue
        inliers = inlier_set(x)
        if len(inliers) >= min_inliers:
            hypotheses.append((x, inliers))

    if len(hypotheses) == 0:
        return []

    hypotheses.sort(key=lambda h: len(h[1]), reverse=True)

    # Greedy de-duplication: drop hypotheses that are spatially close to, or share
    # most of their inliers with, a stronger hypothesis already kept.
    selected = []
    for x, inliers in hypotheses:
        keep = True
        for sx, sinliers in selected:
            overlap = len(inliers & sinliers) / max(1, min(len(inliers), len(sinliers)))
            if float(np.linalg.norm(x - sx)) <= 2.0 * tol or overlap > 0.5:
                keep = False
                break
        if keep:
            selected.append((x, inliers))

    hotspots = []
    for x, inliers in selected:
        inlier_list = sorted(inliers)
        if len(inlier_list) < 4:
            continue
        ref_centers = []
        ref_radii = []
        for i in inlier_list:
            dist = float(np.linalg.norm(x - centers[i]))
            best_r = None
            best_resid = None
            for r in radii_per_res[i]:
                resid = abs(dist - r)
                if best_resid is None or resid < best_resid:
                    best_resid = resid
                    best_r = r
            ref_centers.append(centers[i])
            ref_radii.append(best_r)
        result = IntersectSpheres(np.array(ref_centers), np.array(ref_radii), x0=x)
        hotspots.append({
            "x": result["x"],
            "sigma": result["sigma"],
            "rms": result["rms"],
            "n_inliers": len(inlier_list),
            "converged": result["converged"],
        })

    hotspots.sort(key=lambda h: h["n_inliers"], reverse=True)

    if k is not None and k > 0:
        if len(hotspots) < k:
            print("Warning: requested %d hotspots but only %d were found" % (k, len(hotspots)))
        hotspots = hotspots[:k]

    return hotspots



# Parse commannd-line arguments
# #############################

def ParseCommandline():

	parser = argparse.ArgumentParser()

	parser.add_argument("-v",
					"--version",
					action="store_true", 
					help="returns the version of the script",
					required=False)
					
	parser.add_argument("-c",
						"--centers",
						type=str,			
						help="PDB file to specify the reference atoms and their coordinates. "
							 "The following bash commands can be used to extract N random CA atom cards "
							 "from a PDB file: > cat pdb_file | grep \" CA \" | shuf -n N > outfile" ,
						required=True)
					
	parser.add_argument("-p",
						"--parm",
						type=str,			
						help="AMBER parameter file describing the trajectory files. ",
						required=True)
					
	parser.add_argument("-l",
						"--ligand",
						type=int,			
						help="ligand id (number) as it occurs in the parameter file",
						required=True)
					
	parser.add_argument("-b",
						"--boosts",
						type=str,			
						help="TEXT file that contains the full path of all boost files that should be processed",
						required=True)
					
	parser.add_argument("-t",
						"--trajectories",
						type=str,			
						help="TEXT file that contains the full path of all trajectory files that should be processed",
						required=True)

	parser.add_argument("-o",
						"--output",
						type=str,			
						help="PDB file name of the hotspot center",
						required=False,
						default="hotspot.pdb")

	parser.add_argument("-f",
						"--fraction",
						type=float,			
						help="defines the fraction of centers that should be randomly selected",
						required=False,
						default=1.0)
				
	parser.add_argument("--cutoff",
						type=int,			
						help="Specifies the histogram cutoff number that is used in the PyReweighting script",
						required=False,
						default=10)
				
	parser.add_argument("--boostingType",
						choices=['single', 'dual'],			
						help="Specifies whether a single, or dual/triple boosting was applied",
						required=False,
						default='single')

	parser.add_argument("--hotspots",
						type=int,			
						help="Number of hotspots to report. 0 means auto-detect (default)",
						required=False,
						default=0)

	parser.add_argument("--tolerance",
						type=float,			
						help="Inlier distance tolerance (Angstrom) used when grouping spheres into a common intersection",
						required=False,
						default=1.0)

	parser.add_argument("--pmfCutoff",
						type=float,			
						help="PMF depth window (kcal/mol) above the global minimum for accepting secondary PMF minima as candidate radii",
						required=False,
						default=1.0)

	parser.add_argument("--maxMinima",
						type=int,			
						help="Maximum number of candidate radii (PMF minima) extracted per residue",
						required=False,
						default=3)

	parser.add_argument("--minInliers",
						type=int,			
						help="Minimum number of residues that must support a hotspot. 0 means automatic (max(4, 10%% of residues))",
						required=False,
						default=0)

	parser.add_argument("--pmfFilter",
						choices=["none", "quality", "neighbors", "both", "either"],
						help="PMF quality filter mode before hotspot detection",
						required=False,
						default="none")

	parser.add_argument("--maxBasinWidth",
						type=float,
						help="Maximum basin width (Angstrom) at basinCutoff above the PMF minimum",
						required=False,
						default=6.0)

	parser.add_argument("--basinCutoff",
						type=float,
						help="Energy above minimum (kcal/mol) used to measure basin width",
						required=False,
						default=1.0)

	parser.add_argument("--minBarrier",
						type=float,
						help="Minimum barrier height (kcal/mol) to the next competing PMF minimum",
						required=False,
						default=1.0)

	parser.add_argument("--maxCompetingMinima",
						type=int,
						help="Maximum number of competing PMF minima within pmfCutoff of the global minimum",
						required=False,
						default=1)

	parser.add_argument("--neighborMode",
						choices=["distance", "knn"],
						help="How to define spatial neighbors for PMF similarity",
						required=False,
						default="distance")

	parser.add_argument("--neighborDist",
						type=float,
						help="C-alpha distance cutoff (Angstrom) for neighbor mode 'distance'",
						required=False,
						default=8.0)

	parser.add_argument("--neighborK",
						type=int,
						help="Number of nearest neighbors for neighbor mode 'knn'",
						required=False,
						default=5)

	parser.add_argument("--minNeighborCorr",
						type=float,
						help="Minimum Pearson correlation vs median neighbor PMF profile",
						required=False,
						default=0.6)

	parser.add_argument("--minNeighbors",
						type=int,
						help="Minimum spatial neighbors required to apply the neighbor similarity test",
						required=False,
						default=2)

	args = parser.parse_args()
	
	# --version
	if args.version:
		print("Version %s" % (VERSION))
		sys.exit(0)
	
	# --centers
	if not os.path.exists(args.centers):
		print("Can not find atom centers file %s - quitting with error code 1" % (args.centers))
		sys.exit(1)
	if not os.path.isfile(args.centers):
		print("The atom centers file %s is errorneous - quitting with error code 1" % (args.centers))
		sys.exit(1)
	
	# --ligand
	if args.ligand < 0:
		print("Ligand id should be larger or equal than 0 - quitting with error code 1")
		sys.exit(1)
	
	# --trajectories
	if not os.path.exists(args.trajectories):
		print("Can not find the trajectories file %s - quitting with error code 1" % (args.trajectories))
		sys.exit(1)
	if not os.path.isfile(args.trajectories):
		print("The trajectories file %s is errorneous - quitting with error code 1" % (args.trajectories))
		sys.exit(1)
	
	# --boosts
	if not os.path.exists(args.boosts):
		print("Can not find the boosts file %s - quitting with error code 1" % (args.boosts))
		sys.exit(1)
	if not os.path.isfile(args.boosts):
		print("The boosts file %s is errorneous - quitting with error code 1" % (args.boosts))
		sys.exit(1)
	
	# --parm
	if not os.path.exists(args.parm):
		print("Can not find the parm file %s - quitting with error code 1" % (args.parm))
		sys.exit(1)
	if not os.path.isfile(args.parm):
		print("The parm file %s is errorneous - quitting with error code 1" % (args.parm))
		sys.exit(1)
	
	# --hotspots
	if args.hotspots < 0:
		print("The number of hotspots should be larger or equal than 0 - quitting with error code 1")
		sys.exit(1)
	
	# --tolerance
	if args.tolerance <= 0.0:
		print("The tolerance should be larger than 0 - quitting with error code 1")
		sys.exit(1)
	
	# --pmfCutoff
	if args.pmfCutoff < 0.0:
		print("The pmfCutoff should be larger or equal than 0 - quitting with error code 1")
		sys.exit(1)
	
	# --maxMinima
	if args.maxMinima < 1:
		print("The maxMinima should be larger or equal than 1 - quitting with error code 1")
		sys.exit(1)
	
	# --minInliers
	if args.minInliers < 0:
		print("The minInliers should be larger or equal than 0 - quitting with error code 1")
		sys.exit(1)
	
	# --maxBasinWidth
	if args.maxBasinWidth <= 0.0:
		print("The maxBasinWidth should be larger than 0 - quitting with error code 1")
		sys.exit(1)
	
	# --basinCutoff
	if args.basinCutoff <= 0.0:
		print("The basinCutoff should be larger than 0 - quitting with error code 1")
		sys.exit(1)
	
	# --minBarrier
	if args.minBarrier < 0.0:
		print("The minBarrier should be larger or equal than 0 - quitting with error code 1")
		sys.exit(1)
	
	# --maxCompetingMinima
	if args.maxCompetingMinima < 0:
		print("The maxCompetingMinima should be larger or equal than 0 - quitting with error code 1")
		sys.exit(1)
	
	# --neighborDist
	if args.neighborDist <= 0.0:
		print("The neighborDist should be larger than 0 - quitting with error code 1")
		sys.exit(1)
	
	# --neighborK
	if args.neighborK < 1:
		print("The neighborK should be larger or equal than 1 - quitting with error code 1")
		sys.exit(1)
	
	# --minNeighborCorr
	if args.minNeighborCorr < -1.0 or args.minNeighborCorr > 1.0:
		print("The minNeighborCorr should be between -1 and 1 - quitting with error code 1")
		sys.exit(1)
	
	# --minNeighbors
	if args.minNeighbors < 0:
		print("The minNeighbors should be larger or equal than 0 - quitting with error code 1")
		sys.exit(1)
	
	# Return
	return args
	


# Read the atom centers
# #####################

def ReadAtomCenters(fname, fraction):

	RESIDS = []
	COORDS = []

	try:
		f = open(fname, "r")
	except OSError:
		print("Could not open the atoms centers file %s - quitting with error code 1" % (fname))
		sys.exit(1)
	
	with f:
		for LINE in f.readlines():
			LINE = LINE.strip()
			if LINE == "": continue
			if LINE[:4] != "ATOM" and LINE[:6] != "HETATM": continue
			RESIDS.append(int(LINE[22:26]))
			COORDS.append([float(LINE[30:38]), float(LINE[38:46]), float(LINE[46:54])])
	
	indices = list(range(0, len(COORDS)))
	selection = random.sample(indices, int(fraction * len(indices)))

	return ([RESIDS[i] for i in selection], [COORDS[i] for i in selection])
			
	

# Read the trajectory files
# #########################

def ReadTrajectoryFiles(fname):

	TRAJECTORIES = []
	
	try:
		f = open(fname, "r")
	except OSError:
		print("Could not open the trajectories file %s - quitting with error code 1" % (fname))
		sys.exit(1)
		
	with f:
		for LINE in f.readlines():
			LINE = LINE.strip()
			if LINE == "" or LINE[0] == "#": continue
			if not os.path.exists(LINE):
				print("Can not find the trajectory file %s - quitting with error code 1" % (LINE))
				sys.exit(1)
			if not os.path.isfile(LINE):
				print("The trajectory file %s is errorneous - quitting with error code 1" % (LINE))
				sys.exit(1)
			TRAJECTORIES.append(LINE)
	
	return TRAJECTORIES
			
	

# Process the boost files
# #######################

def ProcessBoostFiles(boostFileName, boostingType, outFile):

	BOOSTFILES = []	
	
	try:
		f = open(boostFileName, "r")
	except OSError:
		print("Could not open the boosts file %s - quitting with error code 1" % (boostFileName))
		sys.exit(1)
		
	with f:
		for LINE in f.readlines():
			LINE = LINE.strip()
			if LINE == "" or LINE[0] == "#": continue
			if not os.path.exists(LINE):
				print("Can not find the boosts file %s - quitting with error code 1" % (LINE))
				sys.exit(1)
			if not os.path.isfile(LINE):
				print("The boosts file %s is errorneous - quitting with error code 1" % (LINE))
				sys.exit(1)
			BOOSTFILES.append(LINE)
	
	COUNTER = 1
	if boostingType == "single": MIN_FIELDS = 7
	if boostingType == "dual": MIN_FIELDS = 8

	fo = open("%s" % (outFile), "w")
	for FN in BOOSTFILES:
		try:
			f = open(FN, "r")
		except OSError:
			print("Could not open the boosts file %s - quitting with error code 1" % (FN))
			sys.exit(1)
		with f:
			for LINE in f.readlines():
				LINE = LINE.strip()
				if LINE == "" or (len(LINE) > 0 and LINE[0] == "#"): continue
				FIELDS = LINE.split()
				if len(FIELDS) < MIN_FIELDS:
					print("Warning: Line does not have enough fields (need at least %d), skipping: %s" % (MIN_FIELDS, LINE))
					continue
				if MIN_FIELDS == 7:
					try:
						v = float(FIELDS[6])
						fo.write("%.5f %d %.5f\n" % (v / (0.001987 * 300.0), COUNTER, v))
						COUNTER += 1
					except ValueError:
						print("Warning: Could not parse float from column 7, skipping: %s" % LINE)
						continue
				if MIN_FIELDS == 8:
					try:
						v = float(FIELDS[6]) + float(FIELDS[7])
						fo.write("%.5f %d %.5f\n" % (v / (0.001987 * 300.0), COUNTER, v))
						COUNTER += 1
					except ValueError:
						print("Warning: Could not parse floats from column 7 or 8, skipping: %s" % LINE)
						continue

	fo.close()

	


# Calculate distances with cpptraj
# ################################

def CalculateDistances(trajectoryFiles, distancesOutFile, parmFile, resids, ligandid):

	f = open("cpptraj.inp", "w")
	f.write("parm %s\n" % (parmFile))
	for TF in trajectoryFiles: f.write("trajin %s\n" % (TF))
	for RESID in resids: f.write("distance %d :%d :%d@CA out %s\n" % (RESID, ligandid, RESID, distancesOutFile))
	f.write("run\n")
	f.close()
	CMD = "%s -i cpptraj.inp" % (CPPTRAJ)
	
	return os.system(CMD)




# Main
# ####

if __name__ == "__main__":

	PROFILES = []
	RADII_PER_RES = []

	# Parse command-line
	# ##################
	
	args = ParseCommandline()
	
	
	# Read centers file
	# #################
	
	RESIDS, COORDS = ReadAtomCenters(args.centers, args.fraction)


	# Read the trajectory files
	# #########################
	
	trajectoryFiles = ReadTrajectoryFiles(args.trajectories)
	

	# Calculate the distances
	# #######################
	
	CalculateDistances(trajectoryFiles, "distances.txt", args.parm, RESIDS, args.ligand)
	
	
	# Process the boost files
	# #######################
	
	ProcessBoostFiles(args.boosts, args.boostingType, "processed.gamd.txt")
	
	
	# Loop over all centers to calculate PMFs
	# #######################################

	for index in range(len(RESIDS)):
	
		fi = open("distances.txt", "r")
		fo = open("tmp.txt", "w")
		for LINE in fi.readlines():
			LINE = LINE.strip()
			if LINE == "" or (len(LINE) > 0 and LINE[0] == "#"): continue
			FIELDS = LINE.split()
			if len(FIELDS) <= index + 1:
				print("Warning: Line does not have enough fields, skipping: %s" % LINE)
				continue
			fo.write("%s\n" % (FIELDS[index+1]))
		fi.close()
		fo.close()
	
		# Get max and min from distance.dat file
		# ######################################
	
		fi = open("tmp.txt", "r")
		MAX = None
		MIN = None
		for LINE in fi.readlines():
			LINE = LINE.strip()
			if LINE == "": continue
			try:
				VALUE = float(LINE)
				if MAX is None:
					MAX = VALUE
					MIN = VALUE
				else:
					if MAX < VALUE: MAX = VALUE
					if MIN > VALUE: MIN = VALUE
			except ValueError:
				continue
		fi.close()
		
		if MAX is None or MIN is None:
			print("Error: Could not extract valid distance values for residue %d - quitting with error code 1" % RESIDS[index])
			sys.exit(1)
	
		# Run the 1D reweighting script
		# #############################
	
		script_path = os.path.join(os.path.dirname(__file__), "PyReweighting-1D.py")
		CMD = "%s %s -input tmp.txt -T 300 -cutoff %d -Xdim %d %d -disc 1 -Emax 20 -job amdweight_CE -weight processed.gamd.txt" % (sys.executable, script_path, args.cutoff, MIN, MAX)
		os.system(CMD)
	
		# Extract candidate radii from the PMF minima
		# ###########################################
	
		parsed = ParsePMF("pmf-c2-tmp.txt.xvg")
		if parsed is None:
			print("Error: Could not extract PMF data for residue %d - quitting with error code 1" % RESIDS[index])
			sys.exit(1)
		dists, pmfs = parsed
		radii = ExtractRadii(dists, pmfs, args.pmfCutoff, args.maxMinima)
		if len(radii) == 0:
			print("Error: Could not extract PMF minima for residue %d - quitting with error code 1" % RESIDS[index])
			sys.exit(1)
		PROFILES.append((dists, pmfs))
		RADII_PER_RES.append(radii)
	
		# Cleanup
		# #######
	
		if os.path.exists("weights-c1-tmp.txt.xvg"): os.remove("weights-c1-tmp.txt.xvg")
		if os.path.exists("weights-c2-tmp.txt.xvg"): os.remove("weights-c2-tmp.txt.xvg")
		if os.path.exists("weights-c3-tmp.txt.xvg"): os.remove("weights-c3-tmp.txt.xvg")
		if os.path.exists("pmf-c1-tmp.txt.xvg"): os.remove("pmf-c1-tmp.txt.xvg")
		if os.path.exists("pmf-c2-tmp.txt.xvg"): os.remove("pmf-c2-tmp.txt.xvg")
		if os.path.exists("pmf-c3-tmp.txt.xvg"): os.remove("pmf-c3-tmp.txt.xvg")


	# Filter PMF profiles
	# ###################

	RESIDS, centers, RADII_PER_RES, pmf_report = FilterResiduesByPMF(
		RESIDS,
		COORDS,
		PROFILES,
		RADII_PER_RES,
		args.pmfFilter,
		args.pmfCutoff,
		args.basinCutoff,
		args.maxBasinWidth,
		args.minBarrier,
		args.maxCompetingMinima,
		args.neighborMode,
		args.neighborDist,
		args.neighborK,
		args.minNeighborCorr,
		args.minNeighbors,
	)

	if len(RESIDS) < 4:
		print("Error: Fewer than 4 residues remain after PMF filtering - try relaxing filter settings - quitting with error code 1")
		sys.exit(1)


	# Detect hotspots
	# ###############

	minInliers = args.minInliers if args.minInliers > 0 else None
	k = args.hotspots if args.hotspots > 0 else None

	hotspots = FindHotspots(centers, RADII_PER_RES, tol=args.tolerance, min_inliers=minInliers, k=k)

	if len(hotspots) == 0:
		print("Error: No hotspots could be detected - try increasing --tolerance or --pmfCutoff - quitting with error code 1")
		sys.exit(1)

	print("Detected %d hotspot(s)" % len(hotspots))

	max_inliers = max(h["n_inliers"] for h in hotspots)

	fo = open(args.output, "w")
	for i, h in enumerate(hotspots):
		x = h["x"]
		sigma = h["sigma"]
		occ = h["n_inliers"] / max_inliers
		bfac = h["rms"]

		print("")
		print("Hotspot %d" % (i + 1))
		print("  Coordinates  : [%.6f, %.6f, %.6f]" % tuple(x))
		print("  1-sigma (xyz): [%.6e, %.6e, %.6e]" % tuple(sigma))
		print("  RMS residual : %.6f" % h["rms"])
		print("  Inliers      : %d" % h["n_inliers"])
		print("  Converged    : %s" % h["converged"])

		fo.write("ATOM  %5d  O   POC %5d    %8.3f%8.3f%8.3f%6.2f%6.2f\n" % (
			i + 1, i + 1, x[0], x[1], x[2], occ, bfac))
	fo.close()
	
