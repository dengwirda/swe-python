
import numpy as np
import xarray
import argparse

from msh import load_mesh, cell_quad
from ops import trsk_mats

# SWE test cases due to Williamson et al
# Authors: Darren Engwirda

def init(name, save, rsph, case):

#-- Williamson, D. L., et al. (1992) A Standard Test Set for
#-- Numerical Approximations to the Shallow Water Equations
#-- in Spherical Geometry, J. Comp. Phys., 102, pp. 211--224

#------------------------------------ load an MPAS mesh file

    print("Load the mesh file")

    mesh = load_mesh(name, rsph)

#------------------------------------ build TRSK matrix op's

    print("Forming coefficients")

    trsk = trsk_mats(mesh)

#------------------------------------ compute test-case IC's

    if (case == 2): 
        wtc2(name, save, rsph, mesh, trsk)
    
    if (case == 5): 
        wtc5(name, save, rsph, mesh, trsk)

    return


def wtc2(name, save, rsph, mesh, trsk):

#-- build a stream-function, velocity field + thickness IC's

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity

    umag = 2.0 * np.pi * rsph / (12.0 * 86400.0)
    g_h0 = 29400.0 # / 1.573125 # for thin = 1m

    sf_vert = rsph * umag * np.sin(mesh.vert.ylat) * -1.

    uu_edge = trsk.edge_grad_perp * (sf_vert) * -1.

    fh_vert = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.vert.ylat) ** 2) / (grav)

    fh_cell = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.cell.ylat) ** 2) / (grav)

    hh_cell = cell_quad(mesh, fh_cell, fh_vert)

    zb_cell = np.zeros(hh_cell.shape, dtype=np.float64)

#-- inject mesh with IC.'s and write output MPAS netCDF file

    print("Output written to:", save)

    init = xarray.open_dataset(name)
    init.attrs.update({"sphere_radius": mesh.rsph})
    init.attrs.update({"config_gravity": grav})
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
    init["h_s"] = (("nCells"), zb_cell)

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


def wtc5(name, save, rsph, mesh, trsk):

#-- build a stream-function, velocity field + thickness IC's

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity

    umag = 20.0
    g_h0 = 5960.0 * grav

    xmid = 3.0 * np.pi / 2.0
    ymid = 1.0 * np.pi / 6.0
    rrad = 1.0 * np.pi / 9.0
    hs_0 = 2000.0

    sf_vert = rsph * umag * np.sin(mesh.vert.ylat) * -1.

    uu_edge = trsk.edge_grad_perp * (sf_vert) * -1.

    fh_vert = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.vert.ylat) ** 2) / (grav)

    fh_cell = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.cell.ylat) ** 2) / (grav)

    hh_cell = cell_quad(mesh, fh_cell, fh_vert)

    rr_vert = (mesh.vert.xlon - xmid) ** 2 + \
              (mesh.vert.ylat - ymid) ** 2
    rr_vert = np.sqrt(np.minimum(rrad ** 2, rr_vert))
    fz_vert = hs_0 * (1.0 - rr_vert / rrad)

    rr_cell = (mesh.cell.xlon - xmid) ** 2 + \
              (mesh.cell.ylat - ymid) ** 2
    rr_cell = np.sqrt(np.minimum(rrad ** 2, rr_cell))
    fz_cell = hs_0 * (1.0 - rr_cell / rrad) 

    zb_cell = cell_quad(mesh, fz_cell, fz_vert)
    
    hh_cell = hh_cell - zb_cell

#-- inject mesh with IC.'s and write output MPAS netCDF file

    print("Output written to:", save)

    init = xarray.open_dataset(name)
    init.attrs.update({"sphere_radius": mesh.rsph})
    init.attrs.update({"config_gravity": grav})
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
    init["h_s"] = (("nCells"), zb_cell)

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
        "--test-case", dest="test_case", type=int,
        required=True, help="Test case number (2-6).")

    parser.add_argument(
        "--radius", dest="radius", type=float,
        required=True, help="Value of sphere_radius.")

    args = parser.parse_args()

    init(name=args.mesh_file,
         save=args.init_file,
         rsph=args.radius,
         case=args.test_case)
