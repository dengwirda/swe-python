
import time
import numpy as np
from scipy.sparse.linalg import gcrotmk
from scipy.integrate import quadrature

import xarray
import argparse

from msh import load_mesh, cell_quad, dual_quad
from ops import trsk_mats

# SWE test cases due to Williamson et al
# Authors: Darren Engwirda

def init(name, save, rsph, case):

#-- Williamson, D. L., et al. (1992) A Standard Test Set for
#-- Numerical Approximations to the Shallow Water Equations
#-- in Spherical Geometry, J. Comp. Phys., 102, pp. 211--224

#------------------------------------ load an MPAS mesh file

    print("Loading the mesh file...")

    mesh = load_mesh(name, rsph)
    
#------------------------------------ build TRSK matrix op's

    print("Forming coefficients...")

    trsk = trsk_mats(mesh)

#------------------------------------ compute test-case IC's

    if (case == 2): 
        wtc2(name, save, rsph, mesh, trsk)

    if (case == 22): 
        wtcb(name, save, rsph, mesh, trsk)

    if (case == 222): 
        wtcc(name, save, rsph, mesh, trsk)
    
    if (case == 4): 
        wtc4(name, save, rsph, mesh, trsk)

    if (case == 5): 
        wtc5(name, save, rsph, mesh, trsk)

    return


def wtc2(name, save, rsph, mesh, trsk):

#-- build a stream-function, velocity field + thickness IC's

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity

    umag = 2.0 * np.pi * rsph / (12.0 * 86400.0)
    g_h0 = 29400.0

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


def wtcb(name, save, rsph, mesh, trsk):

#-- build a stream-function, velocity field + thickness IC's
#-- TC2 "thin", as per Peixoto

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity

    umag = 2.0 * np.pi * rsph / (12.0 * 86400.0)
    g_h0 = 29400.0

    sf_vert = rsph * umag * np.sin(mesh.vert.ylat) * -1.

    uu_edge = trsk.edge_grad_perp * (sf_vert) * -1.

    fh_vert = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.vert.ylat) ** 2) / (grav)

    fh_cell = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.cell.ylat) ** 2) / (grav)

    zb_cell = cell_quad(mesh, fh_cell, fh_vert)

    hh_cell = np.ones(zb_cell.shape, dtype=np.float64)

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


def wtcc(name, save, rsph, mesh, trsk):

#-- build a stream-function, velocity field + thickness IC's
#-- TC2 "thin", as per Peixoto, but with -ve "bump" to build
#-- in dynamicsm a'la TC5

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity

    umag = 2.0 * np.pi * rsph / (12.0 * 86400.0)
    g_h0 = 29400.0

    xmid = 3.0 * np.pi / 2.0
    ymid = 1.0 * np.pi / 6.0
    rrad = 1.0 * np.pi / 9.0
    hs_0 = 1.0

    sf_vert = rsph * umag * np.sin(mesh.vert.ylat) * -1.

    uu_edge = trsk.edge_grad_perp * (sf_vert) * -1.

    fz_vert = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.vert.ylat) ** 2) / (grav)

    fz_cell = (g_h0 - (
        rsph * erot * umag + 0.5 * umag ** 2) * \
        np.sin(mesh.cell.ylat) ** 2) / (grav)

    z1_cell = cell_quad(mesh, fz_cell, fz_vert)

    rr_vert = (mesh.vert.xlon - xmid) ** 2 + \
              (mesh.vert.ylat - ymid) ** 2
    rr_vert = np.sqrt(np.minimum(rrad ** 2, rr_vert))
    fz_vert = fz_vert - hs_0 * (1.0 - rr_vert / rrad)

    rr_cell = (mesh.cell.xlon - xmid) ** 2 + \
              (mesh.cell.ylat - ymid) ** 2
    rr_cell = np.sqrt(np.minimum(rrad ** 2, rr_cell))
    fz_cell = fz_cell - hs_0 * (1.0 - rr_cell / rrad)

    zb_cell = cell_quad(mesh, fz_cell, fz_vert)

    hh_cell = z1_cell - zb_cell + \
        np.ones(zb_cell.shape, dtype=np.float64)

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


def ufn4(alat, umag, rsph):
    
    ubar = umag * (2.0 * np.sin(alat) * np.cos(alat)) ** 14

    return rsph * ubar * -1.0


def hfn4(alat, umag, erot, rsph):

    ubar = umag * (2.0 * np.sin(alat) * np.cos(alat)) ** 14

    frot = +2.0 * erot * np.sin(alat)

    return (rsph * frot + ubar * np.tan(alat)) * ubar


def wtc4(name, save, rsph, mesh, trsk):

#-- build a stream-function, velocity field + thickness IC's

    erot = 7.292E-05            # Earth's omega
    grav = 9.80616              # gravity

    xx_0 = 0.00 * np.pi / 1.0
    yy_0 = 1.00 * np.pi / 4.0

    g_h0 = 100000.0
    ff_0 = 2.00 * erot * np.sin(np.pi / 4.0)

    umag = 40.0

    sf_0 =-0.03 * g_h0 / ff_0

    sig0 = (2.0 * mesh.rsph / 1.0E+06) ** 2 

    print("Computing streamfunction...")

    sf_vert = np.zeros(
        mesh.vert.size, dtype=np.float64)
    sf_cell = np.zeros(
        mesh.cell.size, dtype=np.float64)

    for vert in range(mesh.vert.size):
        sf_vert[vert], _ = quadrature(
            ufn4, -np.pi / 2., mesh.vert.ylat[vert], 
            miniter=16, args=(umag, mesh.rsph))

    cc_0 = np.sin(yy_0) * np.sin(mesh.vert.ylat) + \
           np.cos(yy_0) * np.cos(mesh.vert.ylat) * \
           np.cos(mesh.vert.xlon - xx_0)

    sf_b = sf_0 * \
        np.exp(-sig0 * (1.0 - cc_0) / (1.0 + cc_0))

    sf_vert = sf_b + sf_vert

    print("--> done: vert!")

    for cell in range(mesh.cell.size):
        sf_cell[cell], _ = quadrature(
            ufn4, -np.pi / 2., mesh.cell.ylat[cell], 
            miniter=16, args=(umag, mesh.rsph))

    cc_0 = np.sin(yy_0) * np.sin(mesh.cell.ylat) + \
           np.cos(yy_0) * np.cos(mesh.cell.ylat) * \
           np.cos(mesh.cell.xlon - xx_0)

    sf_b = sf_0 * \
        np.exp(-sig0 * (1.0 - cc_0) / (1.0 + cc_0))

    print(np.max(np.abs(sf_b)), np.max(np.abs(sf_cell)))

    sf_cell = sf_b + sf_cell

    print("--> done: cell!")



    print("Computing velocity field...")

    uu_edge = trsk.edge_grad_perp * sf_vert * -1.

   #vv_edge = trsk.edge_flux_perp * uu_edge * -1.
    vv_edge = trsk.edge_grad_norm * sf_cell * -1.

    du_cell = trsk.cell_flux_sums * uu_edge

    print("--> max(abs(unrm)):", np.max(uu_edge))
    print("--> sum(div(unrm)):", np.sum(du_cell))

    """
    ff_vert = 2.0 * erot * np.sin(mesh.vert.ylat)
    ff_cell = 2.0 * erot * np.sin(mesh.cell.ylat)

    dh_vert = np.zeros(
        mesh.vert.size, dtype=np.float64)
    dh_cell = np.zeros(
        mesh.cell.size, dtype=np.float64)

    for vert in range(mesh.vert.size):
        dh_vert[vert], _ = quadrature(
            hfn4, -np.pi / 2., mesh.vert.ylat[vert], 
            miniter=16, args=(umag, erot, mesh.rsph))

    cc_0 = np.sin(yy_0) * np.sin(mesh.vert.ylat) + \
           np.cos(yy_0) * np.cos(mesh.vert.ylat) * \
           np.cos(mesh.vert.xlon - xx_0)

    sf_b = sf_0 * \
        np.exp(-sig0 * (1.0 - cc_0) / (1.0 + cc_0))

    fh_vert = (g_h0 - dh_vert + ff_vert * sf_b) / grav

    print("--> done: vert!")

    for cell in range(mesh.cell.size):
        dh_cell[cell], _ = quadrature(
            hfn4, -np.pi / 2., mesh.cell.ylat[cell], 
            miniter=16, args=(umag, erot, mesh.rsph))

    cc_0 = np.sin(yy_0) * np.sin(mesh.cell.ylat) + \
           np.cos(yy_0) * np.cos(mesh.cell.ylat) * \
           np.cos(mesh.cell.xlon - xx_0)

    sf_b = sf_0 * \
        np.exp(-sig0 * (1.0 - cc_0) / (1.0 + cc_0))

    fh_cell = (g_h0 - dh_cell + ff_cell * sf_b) / grav

    print("--> done: cell!")


    hh_cell = cell_quad(mesh, fh_cell, fh_vert)
    """

#-- solve -g * del^2 h = div f * u_perp for layer thickness,
#-- leads to a h which is in discrete balance

    print("Computing layer thickness...")

    ff_vert = 2.0 * erot * np.sin(mesh.vert.ylat)
    ff_edge = trsk.edge_stub_sums * ff_vert
    ff_edge = ff_edge / mesh.edge.area

    rh_edge = trsk.cell_flux_sums * (ff_edge * vv_edge)
    rh_edge = rh_edge * -1.0 / grav

    rh_edge = rh_edge - np.mean(rh_edge) # INT rhs must be 0
    rh_edge = rh_edge - np.mean(rh_edge)

    ttic = time.time()
    hh_cell, info = gcrotmk(trsk.cell_del2_sums, 
                            rh_edge, 
                            tol=1.0E-12, m=50, k=25)
    ttoc = time.time()
    print(ttoc - ttic)    

    if (info != +0): raise Exception("Did not converge!")


    hh_cell = g_h0 / grav + hh_cell
    


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
