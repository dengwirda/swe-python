
import numpy as np
import xarray
import argparse

from msh import load_mesh
from ops import trsk_mats


def init(name, save, rsph):

#-- Williamson, D. L., et al. (1992) A Standard Test Set for 
#-- Numerical Approximations to the Shallow Water Equations 
#-- in Spherical Geometry, J. Comp. Phys., 102, pp. 211--224

#------------------------------------ load an MPAS mesh file

    print("Load the mesh file")

    mesh = load_mesh(name, rsph)

#------------------------------------ build TRSK matrix op's

    print("Forming coefficients")

    trsk = trsk_mats(mesh)

#-- build a stream-function, velocity field + thickness IC's

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity
    
    umag = 2.0 * np.pi * rsph / (12.0 * 86400.0)
    g_h0 = 29400.0
    
    sf_vert = rsph * umag * np.sin(mesh.vert.ylat) * -1.

    uu_edge = trsk.edge_grad_perp * (sf_vert) * -1.

    hh_cell = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.cell.ylat) ** 2) / (grav)

#-- inject mesh with IC.'s and write output MPAS netCDF file  

    print("Output written to:", save)

    init = xarray.open_dataset(name)
    init.attrs.update({"sphere_radius": mesh.rsph})
    init["xCell"] = (("nCells"), mesh.cell.xpos)
    init["yCell"] = (("nCells"), mesh.cell.ypos)
    init["zCell"] = (("nCells"), mesh.cell.zpos)
    init["areaCell"] = (("nCells"), mesh.cell.area)

    init["xEdge"] = (("nEdges"), mesh.edge.xpos)
    init["yEdge"] = (("nEdges"), mesh.edge.ypos)
    init["zEdge"] = (("nEdges"), mesh.edge.zpos)
    init["dvEdge"] = (("nEdges"), mesh.edge.vlen)    
    init["dcEdge"] = (("nEdges"), mesh.edge.clen)

    init["xVertex"] = (("nVertices"), mesh.vert.xpos)
    init["yVertex"] = (("nVertices"), mesh.vert.ypos)
    init["zVertex"] = (("nVertices"), mesh.vert.zpos)
    init["areaTriangle"] = (("nVertices"), mesh.vert.area)
    init["kiteAreasOnVertex"] = (
        ("nVertices", "vertexDegree"), mesh.vert.kite)

    init["h"] = (
        ("Time", "nCells", "nVertLevels"),
        np.reshape(hh_cell, (1, mesh.cell.size, 1)))
    init["u"] = (
        ("Time", "nEdges", "nVertLevels"),
        np.reshape(uu_edge, (1, mesh.edge.size, 1)))

    init["streamfunction"] = (("nVertices"), sf_vert)
    init["vorticity"] = (
        ("nVertices"), 
        (trsk.dual_curl_sums * uu_edge) / mesh.vert.area)

    init["tracers"] = (
        ("Time", "nCells", "nVertLevels", "nTracers"), 
        np.zeros((1, mesh.cell.size, 1, 1)))

    init["fCell"] = (("nCells"), 
        2.00E+00 * erot * np.sin(mesh.cell.ylat))
    init["fEdge"] = (("nEdges"), 
        2.00E+00 * erot * np.sin(mesh.edge.ylat))
    init["fVertex"] = (("nVertices"), 
        2.00E+00 * erot * np.sin(mesh.vert.ylat))

    print(init)

    init.to_netcdf(save, format="NETCDF3_64BIT_OFFSET")

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
        required=True, help="IC's filename to write.")

    parser.add_argument(
        "--radius", dest="radius", type=float,
        required=True, help="Value of sphere_radius.")

    args = parser.parse_args()

    init(name=args.mesh_file,
         save=args.init_file,
         rsph=args.radius)
