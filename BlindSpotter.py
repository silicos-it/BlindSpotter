#!/usr/bin/env python

import argparse
import sys
import os
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
		print("Ligand id should be larger or equal than 0 - quitting with error code 1" % (args.centers))
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
	
	# Return
	return args
	


# Read the atom centers
# #####################

def ReadAtomCenters(fname):

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
			if LINE is None or LINE == "": continue
			if LINE[:4] != "ATOM" and LINE[:6] != "HETATM": continue
			RESIDS.append(int(LINE[21:26]))
			COORDS.append([float(LINE[30:38]), float(LINE[38:46]), float(LINE[46:54])])
	
	return(RESIDS, COORDS)
			
	

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
			if LINE is None or LINE == "": continue
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

def ProcessBoostFiles(boostFileName, outFile):

	BOOSTFILES = []	
	
	try:
		f = open(boostFileName, "r")
	except OSError:
		print("Could not open the boosts file %s - quitting with error code 1" % (boostFileName))
		sys.exit(1)
		
	with f:
		for LINE in f.readlines():
			LINE = LINE.strip()
			if LINE is None or LINE == "": continue
			if not os.path.exists(LINE):
				print("Can not find the boosts file %s - quitting with error code 1" % (LINE))
				sys.exit(1)
			if not os.path.isfile(LINE):
				print("The boosts file %s is errorneous - quitting with error code 1" % (LINE))
				sys.exit(1)
			BOOSTFILES.append(LINE)
	
	COUNTER = 1
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
				if LINE is None or LINE == "" or LINE[0] == "#": continue
				FIELDS = LINE.split()
				v = float(FIELDS[6])
				fo.write("%.5f %d %.5f\n" % (v / (0.001987 * 300.0), COUNTER, v))
				COUNTER += 1
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

	RADII = []

	# Parse command-line
	# ##################
	
	args = ParseCommandline()
	
	
	# Read centers file
	# #################
	
	RESIDS, COORDS = ReadAtomCenters(args.centers)


	# Read the trajectory files
	# #########################
	
	trajectoryFiles = ReadTrajectoryFiles(args.trajectories)
	

	# Calculate the distances
	# #######################
	
	CalculateDistances(trajectoryFiles, "distances.txt", args.parm, RESIDS, args.ligand)
	
	
	# Process the boost files
	# #######################
	
	ProcessBoostFiles(args.boosts, "processed.gamd.txt")
	
	
	# Loop over all centers to calculate PMFs
	# #######################################

	for index in range(len(RESIDS)):
	
		fi = open("distances.txt", "r")
		fo = open("tmp.txt", "w")
		for LINE in fi.readlines():
			LINE = LINE.strip()
			if LINE is None or LINE == "" or LINE[0] == "#": continue
			FIELDS = LINE.split()
			fo.write("%s\n" % (FIELDS[index+1]))
		fi.close()
		fo.close()
	
		# Get max and min from distance.dat file
		# ######################################
	
		fi = open("tmp.txt", "r")
		LINE = fi.readline()
		LINE = LINE.strip()
		MAX = float(LINE)
		MIN = float(LINE)
		fi.close()
		fi = open("tmp.txt", "r")
		for LINE in fi.readlines():
			LINE = LINE.strip()
			if LINE is None or LINE == "": continue
			VALUE = float(LINE)
			if MAX < VALUE: MAX = VALUE
			if MIN > VALUE: MIN = VALUE
		fi.close()
	
		# Run the 1D reweighting script
		# #############################
	
		CMD = "./PyReweighting-1D.py -input tmp.txt -T 300 -cutoff 75 -Xdim %d %d -disc 1 -Emax 20 -job amdweight_CE -weight processed.gamd.txt" % (MIN, MAX)
		os.system(CMD)
	
		# Get the radius with the lowest pmf
		# ##################################
	
		fi = open("pmf-c2-tmp.txt.xvg", "r")
		for LINE in fi.readlines():
			LINE = LINE.strip()
			if LINE is None or LINE == "" or LINE[0] == "#" or LINE[0] == "@": continue
			FIELDS = LINE.split()
			D = float(FIELDS[0])
			PMF = float(FIELDS[1])
			if PMF == 0.0: break
		fi.close()
		RADII.append(D)
	
		# Cleanup
		# #######
	
		if os.path.exists("weights-c1-tmp.txt.xvg"): os.remove("weights-c1-tmp.txt.xvg")
		if os.path.exists("weights-c2-tmp.txt.xvg"): os.remove("weights-c2-tmp.txt.xvg")
		if os.path.exists("weights-c3-tmp.txt.xvg"): os.remove("weights-c3-tmp.txt.xvg")
		if os.path.exists("pmf-c1-tmp.txt.xvg"): os.remove("pmf-c1-tmp.txt.xvg")
		if os.path.exists("pmf-c2-tmp.txt.xvg"): os.remove("pmf-c2-tmp.txt.xvg")
		if os.path.exists("pmf-c3-tmp.txt.xvg"): os.remove("pmf-c3-tmp.txt.xvg")


	# Minimize
	# ########

	centers = np.asarray(COORDS)
	radii = np.asarray(RADII)
    
	result = IntersectSpheres(centers, radii)
	x = result["x"]
	sigma = result["sigma"]

	print("Method       :", result["method"])
	print("Converged    :", result["converged"])
	print("Best-fit x   : [%.6f, %.6f, %.6f]" % tuple(x))
	print("1-sigma (xyz): [%.6e, %.6e, %.6e]" % tuple(sigma))
	print("RMS residual :", result["rms"])
	print("DoF          :", result["dof"])

	fo = open("hotspot.pdb", "w")
	fo.write("ATOM      1  O   POC     1    %8.3f%8.3f%8.3f  1.00  0.00\n" % (x[0], x[1], x[2]))
	fo.close()
	
