
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.integrate import quadrature

import xarray
import argparse

from msh import load_mesh
from ops import trsk_mats


def ujet(alat, lat0, lat1, uamp, rsph):
    """
    Helper to integrate PSI from U = -1/R * d/d.lat(PSI) via
    quadrature...

    U(lat) = A * exp(1/((lat-lat0) * (lat-lat1)))

    """

    return -rsph * uamp * np.exp(
        1.0E+0 / ((alat - lat0) * (alat - lat1)))
    

def main(name, save, pert=True):
    """
    MAIN: Form SWE initial conditions for the barotropic jet
    case.    

    Adds initial conditions to the MPAS mesh file NAME.nc,
    with the output IC's written to SAVE.nc.

    If PERT=TRUE, adds a perturbation to the layer thickness
    
    """
    # Authors: Darren Engwirda

#------------------------------------ load an MPAS mesh file

    print("Loading mesh file")

    mesh = load_mesh(name)

#------------------------------------ build TRSK matrix op's

    print("Forming coefficients")

    trsk = trsk_mats(mesh)

#------------------------------------ build a streamfunction

    print("Build streamfunction")

#-- J. Galewsky, R.K. Scott & L.M. Polvani (2004) An initial 
#-- value problem for testing numerical models of the global 
#-- shallow-water equations, Tellus A: Dynamic Meteorology & 
#-- Oceanography, 56:5, 429-440

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity

    lat0 = np.pi / 7.0          # jet lat width
    lat1 = np.pi / 2.0 - lat0
    
    umid = 8.000E+01            # jet max speed
    hbar = 1.000E+04            # mean layer hh
   
    uamp = umid / np.exp(-4. / (lat1 - lat0) ** 2)
   
#-- build a streamfunction at mesh vertices using quadrature
 
    vpsi = np.zeros(
        mesh.vert.size, dtype=np.float64)

    for node in range(mesh.vert.size):
        alat = mesh.vert.ylat[node]
        if (alat >= lat0 and alat < lat1):
            vpsi[node], _ = quadrature(
                ujet, lat0, alat, 
                args=(lat0, lat1, uamp, mesh.rsph))

    mask = (mesh.vert.ylat[:] >= lat1)

    vpsi[mask] = np.min(vpsi)

#-- form velocity on edges from streamfunction: ensures u is
#-- div-free in a discrete sense. 

#-- this comes from taking div(*) of the momentum equations,
#-- see: H. Weller, J. Thuburn, C.J. Cotter (2012):
#-- Computational Modes and Grid Imprinting on Five Quasi-
#-- Uniform Spherical C-Grids, M.Wea.Rev. 140(8): 2734–2755.
    
    print("Calc. velocity field")

    unrm = trsk.edge_grad_perp * vpsi

    uprp = trsk.edge_reco_perp * unrm

    udiv = trsk.cell_flux_sums * unrm

    print("--> max(abs(unrm)):", np.max(unrm))
    print("--> sum(div(unrm)):", np.sum(udiv))

#-- solve -g * del^2 h = div f * u_perp for layer thickness,
#-- leads to a h which is in discrete balance

    print("Find layer thickness")

    frot = 2.0 * erot * np.sin(mesh.edge.ylat)

    vrhs = trsk.cell_flux_sums * (frot * uprp)
    vrhs = vrhs * -1.00 / grav
    
    vrhs = vrhs - np.mean(vrhs)     # INT rhs dA must be 0.0
    vrhs = vrhs - np.mean(vrhs)

    hdel = spsolve(trsk.cell_del2_sums, vrhs)
   
    herr = hbar - hdel
    hdel = hdel + (                 # add offset for mean hh
        np.sum(mesh.cell.area * herr) /
        np.sum(mesh.cell.area * 1.00)
    )

#-- optional: add perturbation to the thickness distribution

    lat2 = np.pi / 4.               # perturbation constants
    lon2 = np.pi / 1.

    hmul = 120.0
    eta1 = 1. / 3.
    eta2 = 1. / 15.

    hadd = (hmul * np.cos(mesh.cell.ylat) *
       np.exp(-((mesh.cell.xlon - lon2) / eta1) ** 2) *
       np.exp(-((lat2 - mesh.cell.ylat) / eta2) ** 2)
    )

    hdel = hdel + float(pert) * hadd

#-- inject mesh with IC.'s and write output MPAS netCDF file  

    print("Output written to:", save)

    vmag = np.sqrt(unrm ** 2 + uprp ** 2)
    vvel = (
        vmag[mesh.vert.edge[:, 0] - 1] +
        vmag[mesh.vert.edge[:, 1] - 1] +
        vmag[mesh.vert.edge[:, 2] - 1]
    ) / 3.00

    init = xarray.open_dataset(name)
    init["layerThickness"] = (
        ("Time", "nCells", "nVertLevels"), 
        np.reshape(hdel, (1, mesh.cell.size, 1)))
    init["normalVelocity"] = (
        ("Time", "nEdges", "nVertLevels"), 
        np.reshape(unrm, (1, mesh.edge.size, 1)))
    init["streamfunction"] = (("nVertices"), vpsi)
    init["velocityTotals"] = (("nVertices"), vvel)

    init["bottomDepth"] = (
        ("nCells"), np.full(mesh.cell.size, 0.))
    init["fCell"] = (
        ("nCells", "nVertLevels"), np.reshape(
            2. * erot * np.sin(mesh.cell.ylat), 
            (mesh.cell.size, 1)))
    init["fEdge"] = (
        ("nEdges", "nVertLevels"), np.reshape(
            2. * erot * np.sin(mesh.edge.ylat), 
            (mesh.edge.size, 1)))
    init["fVertex"] = (
        ("nVertices", "nVertLevels"), np.reshape(
            2. * erot * np.sin(mesh.vert.ylat), 
            (mesh.vert.size, 1)))
    init["maxLevelCell"] = (
        ("nCells"), np.full(mesh.cell.size, +1, 
                            dtype=np.int32))

    init.to_netcdf(save)

    return


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--mesh-file", dest="mesh_file", type=str,
        required=True, help="Path to user mesh file.")
    
    parser.add_argument(
        "--init-file", dest="init_file", type=str,
        required=True, help="Path to user init file.")

    parser.add_argument(
        "--with-pert", dest="with_pert", type=bool,
        required=True, help="True to add hh perturb.")

    args = parser.parse_args()

    main(name=args.mesh_file,
         save=args.init_file,
         pert=args.with_pert)
