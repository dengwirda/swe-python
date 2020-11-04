
import os
import time
from importlib import import_module
import numpy as np
import netCDF4 as nc

import jigsawpy
import argparse

from ops import init_mesh_sign, eval_cell_div_, eval_edge_hpgf


def swe(path_name):
    """
    SWE: a very simple driver to solve the SWE using spherical 
    MPAS meshes.

    """
    # Authors: Darren Engwirda

    mesh = load_mesh(path_name)
    base = load_defs(path_name, mesh)

    mesh = init_mesh_sign(base, mesh)

    unrm = base.unrm[:, 0]
    zlev = base.zlev[:, :]
    hdel = base.zlev[:, 0] - base.zlev[:, 1]

    uold = unrm
    hold = hdel

    for step in range(base.cnfg.iter):

        plev = np.zeros(                        # p=0 at surf.
            (mesh.cell.size, 2), dtype=float)

    #-- a TVD-RK2 + FB scheme, a'la ROMS (but with thickness
    #-- updated via the standard "monotone" TVD-RK approach)

        BETA = 1.0 / 3.0

        divh = eval_cell_div_(base, mesh, unrm, hdel)

        h1st = hdel - 1.0 * base.cnfg.step * divh

        hbar = hdel * (1.0 - BETA) + h1st * (0.0 + BETA)

        hpgf = hbar[mesh.edge.cell[:, 1] - 1] - \
               hbar[mesh.edge.cell[:, 0] - 1]
        hpgf = hpgf / mesh.edge.clen

        u1st = unrm - 1.0 * base.cnfg.step * hpgf * base.cnfg.grav
        

        htmp = hdel * 0.5 + h1st * 0.5
        utmp = unrm * 0.5 + u1st * 0.5

        divh = eval_cell_div_(base, mesh, u1st, h1st)

        h2nd = htmp - 0.5 * base.cnfg.step * divh

        BETA = 2.0 / 3.0

        hbar = h1st * (1.0 - BETA) + h2nd * (0.0 + BETA)

        hpgf = hbar[mesh.edge.cell[:, 1] - 1] - \
               hbar[mesh.edge.cell[:, 0] - 1]
        hpgf = hpgf / mesh.edge.clen

        u2nd = utmp - 0.5 * base.cnfg.step * hpgf * base.cnfg.grav
        

        uold = unrm; unrm = u2nd
        hold = hdel; hdel = h2nd



    #-- compute total energy, see Ringler et al: A unified approach 
    #-- to energy conservation and potential vorticity dynamics for
    #-- arbitrarily structured C-grids

        hmid = hdel[mesh.edge.cell[:, 0] - 1] + \
               hdel[mesh.edge.cell[:, 1] - 1]
        hmid = hmid * 0.5

        ke = 0.5 * unrm ** 2 * hmid * mesh.edge.vlen * mesh.edge.clen
        pe = base.cnfg.grav * (0.5 * hdel - zlev[:, 1]) * mesh.cell.area * hdel

        etot = np.sum(ke) + np.sum(pe)
        
        print(etot)
        


        if (step % base.cnfg.freq == 0):

            mout = jigsawpy.jigsaw_msh_t()
            mout.point = np.zeros(
                mesh.cell.size, dtype=mout.VERT3_t)
            mout.point["coord"][:, 0] = mesh.cell.xpos[:]
            mout.point["coord"][:, 1] = mesh.cell.ypos[:]
            mout.point["coord"][:, 2] = mesh.cell.zpos[:]
            mout.tria3 = np.zeros(
                mesh.vert.size, dtype=mout.TRIA3_t)
            mout.tria3["index"] = mesh.vert.cell[:] - 1
            mout.value = hdel[:]

            jigsawpy.savevtk(os.path.join(
                path_name, "out", "flow" + str(step)), mout)

    return


def load_defs(path_name, mesh):

    stem = path_name.replace(os.path.sep, ".")

    class base: pass

    defs = base()
    defs.unrm = getattr(import_module(
        stem + ".define"), "setunrm")(mesh)
    defs.zlev = getattr(import_module(
        stem + ".define"), "setzlev")(mesh)
    defs.cnfg = getattr(import_module(
        stem + ".define"), "setcnfg")(mesh)

    return defs


def load_mesh(path_name):

    class base: pass

    data = nc.Dataset(os.path.join(
        path_name, "base_mesh.nc"), "r")

    mesh = base()
    mesh.rsph = float(data.sphere_radius)

    mesh.cell = base()
    mesh.cell.size = int(data.dimensions["nCells"].size)
    mesh.cell.xpos = np.array(data.variables["xCell"])
    mesh.cell.ypos = np.array(data.variables["yCell"])
    mesh.cell.zpos = np.array(data.variables["zCell"])
    mesh.cell.xlon = np.array(data.variables["lonCell"])
    mesh.cell.ylat = np.array(data.variables["latCell"])
    mesh.cell.area = np.array(data.variables["areaCell"])   
    mesh.cell.vert = \
        np.array(data.variables["verticesOnCell"])
    mesh.cell.edge = \
        np.array(data.variables["edgesOnCell"])
    mesh.cell.topo = \
        np.array(data.variables["nEdgesOnCell"])

    mesh.edge = base()
    mesh.edge.size = int(data.dimensions["nEdges"].size)
    mesh.edge.xpos = np.array(data.variables["xEdge"])
    mesh.edge.ypos = np.array(data.variables["yEdge"])
    mesh.edge.zpos = np.array(data.variables["zEdge"])
    mesh.edge.xlon = np.array(data.variables["lonEdge"])
    mesh.edge.ylat = np.array(data.variables["latEdge"])
    mesh.edge.vlen = np.array(data.variables["dvEdge"])
    mesh.edge.clen = np.array(data.variables["dcEdge"])
    mesh.edge.vert = \
        np.array(data.variables["verticesOnEdge"])
    mesh.edge.cell = \
        np.array(data.variables["cellsOnEdge"])
    mesh.edge.edge = \
        np.array(data.variables["edgesOnEdge"])
    mesh.edge.topo = \
        np.array(data.variables["nEdgesOnEdge"])

    mesh.vert = base()
    mesh.vert.size = int(data.dimensions["nVertices"].size)
    mesh.vert.xpos = np.array(data.variables["xVertex"])
    mesh.vert.ypos = np.array(data.variables["yVertex"])
    mesh.vert.zpos = np.array(data.variables["zVertex"])
    mesh.vert.xlon = np.array(data.variables["lonVertex"])
    mesh.vert.ylat = np.array(data.variables["latVertex"])
    mesh.vert.area = \
        np.array(data.variables["areaTriangle"])
    mesh.vert.kite = \
        np.array(data.variables["kiteAreasOnVertex"])
    mesh.vert.edge = \
        np.array(data.variables["edgesOnVertex"])
    mesh.vert.cell = \
        np.array(data.variables["cellsOnVertex"])

    return mesh


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--path-name", dest="path_name", type=str,
        required=True, help="Path to user-def. run-dir.")

    args = parser.parse_args()

    swe(path_name=args.path_name)
