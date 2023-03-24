
import time
import numpy as np
import xarray
import argparse

from msh import load_mesh, cell_quad, dual_quad
from ops import trsk_mats

# SWE test cases for linear wave problems
# Authors: Darren Engwirda

def init(name, save, rsph, case, xmid, ymid):

#------------------------------------ load an MPAS mesh file

    print("Loading the mesh file...")

    mesh = load_mesh(name, rsph)
    
#------------------------------------ build TRSK matrix op's

    print("Forming coefficients...")

    trsk = trsk_mats(mesh)

#------------------------------------ compute test-case IC's

    xmid = xmid * np.pi / 180.0
    ymid = ymid * np.pi / 180.0

    if (case == 1): 
        ltc1(name, save, rsph, mesh, trsk, xmid, ymid)

    if (case == 2): 
        ltc2(name, save, rsph, mesh, trsk, xmid, ymid)

    return


def ltc1(name, save, rsph, mesh, trsk, xmid, ymid):

#-- simple isolated gravity-wave test-case

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity

    uu_edge = np.zeros(mesh.edge.size, dtype=np.float64)

    hh_cell = np.exp(
            - 100. * (mesh.cell.xlon - xmid) ** 2 + \
            - 100. * (mesh.cell.ylat - ymid) ** 2 ) \
            + 500.0

    zb_cell = np.zeros(mesh.cell.size, dtype=np.float64)

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


def ltc2(name, save, rsph, mesh, trsk, xmid, ymid):

#-- earth-topography tsunami-wave test-case

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity

    data = xarray.open_dataset(name)

    if ("bed_elevation" not in data.variables.keys()):
        raise ValueError("Elevation data not found.")

    if ("ice_thickness" not in data.variables.keys()):
        raise ValueError("Elevation data not found.")

    zb_cell = np.asarray(
        data["bed_elevation"][:], dtype=np.float64)
    zb_cell+= np.asarray(
        data["ice_thickness"][:], dtype=np.float64)

    zb_cell = np.minimum(-10., zb_cell)  # not too shallow

    uu_edge = np.zeros(mesh.edge.size, dtype=np.float64)

    hh_cell = -zb_cell + 5.0 * np.exp(
            - 250. * (mesh.cell.xlon - xmid) ** 2 + \
            - 250. * (mesh.cell.ylat - ymid) ** 2 )

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
        required=True, help="Test case number (1-2).")

    parser.add_argument(
        "--radius", dest="radius", type=float,
        required=True, help="Value of sphere_radius [m].")

    parser.add_argument(
        "--wave-xmid", dest="wave_xmid", type=float,
        default=+180.,
        required=False, 
        help="Centre of wave in lon. direction [deg].")

    parser.add_argument(
        "--wave-ymid", dest="wave_ymid", type=float,
        default=+0.00,
        required=False, 
        help="Centre of wave in lat. direction [deg].")

    args = parser.parse_args()

    init(name=args.mesh_file,
         save=args.init_file,
         rsph=args.radius,
         case=args.test_case,
         xmid=args.wave_xmid,
         ymid=args.wave_ymid)
