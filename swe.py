
import os
import time
import numpy as np
import netCDF4 as nc
import argparse

from msh import load_mesh, load_flow
from ops import trsk_mats

from _dx import HH_TINY, UU_TINY
from _dx import invariant, tcpu
from _dt import step_RK22, step_RK32, step_SP33, step_RK4

def strtobool(val):
    """
    Silly re-implementation of strtobool, since python has
    been deprecating these things...

    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("Invalid bool value %r" % (val,))


def swe(cnfg):
    """
    SWE: solve the nonlinear SWE on generalised MPAS meshes.

    """
    # Authors: Darren Engwirda

    cnfg.stat_freq = np.minimum(
        cnfg.save_freq, cnfg.stat_freq)
    
    cnfg.integrate = cnfg.integrate.upper()
    cnfg.operators = cnfg.operators.upper()
    cnfg.ke_upwind = cnfg.ke_upwind.upper()
    cnfg.ke_scheme = cnfg.ke_scheme.upper()
    cnfg.pv_upwind = cnfg.pv_upwind.upper()
    cnfg.pv_scheme = cnfg.pv_scheme.upper()

    cnfg.du_damp_4 = np.sqrt(cnfg.du_damp_4)
    cnfg.vu_damp_4 = np.sqrt(cnfg.vu_damp_4)

    name = cnfg.mpas_file
    path, file = os.path.split(name)
    save = os.path.join(path, "out_" + file)

    if cnfg.fb_weight:
        nweights = len(cnfg.fb_weight)
        if ("RK22" in cnfg.integrate) and (nweights != 2):
            raise Exception("RK22 requires 2 FB-weights, " +
                            "{} provided".format(nweights))
        elif ("RK32" in cnfg.integrate) and (nweights != 3):
            raise Exception("RK32 requires 3 FB-weights, " +
                            "{} provided".format(nweights))

    print("Loading input assets...")
    
    # load mesh + init. conditions
    mesh = load_mesh(name, sort=True)
    flow = load_flow(name, mesh=mesh)

    print("Creating output file...")

    init_file(name, cnfg, save)

    u0_edge = flow.uu_edge[-1, :, 0]
    uu_edge = u0_edge
    h0_cell = flow.hh_cell[-1, :, 0]
    hh_cell = h0_cell

    hh_cell = np.maximum(HH_TINY, hh_cell)

    print("Forming coefficients...")

    # set sparse spatial operators
    trsk = trsk_mats(mesh)

    # remap fe,fc is more accurate?
    flow.ff_edge = trsk.edge_stub_sums * flow.ff_vert
    flow.ff_edge = \
        (flow.ff_edge / mesh.edge.area)

    flow.ff_cell = trsk.cell_kite_sums * flow.ff_vert
    flow.ff_cell = \
        (flow.ff_cell / mesh.cell.area)

    kp_sums = np.zeros((
        cnfg.iteration // cnfg.stat_freq + 1), dtype=float)
    en_sums = np.zeros((
        cnfg.iteration // cnfg.stat_freq + 1), dtype=float)

    print("Integrating:")

    ttic = time.time(); xout = []; next = 0; freq = 0

    for step in range(cnfg.iteration + 1):

        if ("RK22" in cnfg.integrate):

            hh_cell, uu_edge, ke_cell, ke_dual, \
            rv_cell, pv_cell, \
            rv_dual, pv_dual, \
            ke_bias, pv_bias = step_RK22(
                mesh, trsk, flow, cnfg, hh_cell, uu_edge)

        if ("RK32" in cnfg.integrate):

            hh_cell, uu_edge, ke_cell, ke_dual, \
            rv_cell, pv_cell, \
            rv_dual, pv_dual, \
            ke_bias, pv_bias = step_RK32(
                mesh, trsk, flow, cnfg, hh_cell, uu_edge)

        if ("SP33" in cnfg.integrate):

            hh_cell, uu_edge, ke_cell, ke_dual, \
            rv_cell, pv_cell, \
            rv_dual, pv_dual, \
            ke_bias, pv_bias = step_SP33(
                mesh, trsk, flow, cnfg, hh_cell, uu_edge)
        
        if ("RK4" in cnfg.integrate):

            hh_cell, uu_edge, ke_cell, ke_dual, \
            rv_cell, pv_cell, \
            rv_dual, pv_dual, \
            ke_bias, pv_bias = step_RK4(
                mesh, trsk, flow, cnfg, hh_cell, uu_edge)

        if (step % cnfg.stat_freq == 0):

            kp_sums[next], \
            en_sums[next] = invariant(
                mesh, trsk, flow, cnfg, hh_cell, uu_edge)

            print("step, KE+PE, PV**2:", step,
                  (kp_sums[next] - kp_sums[0]) 
                / (kp_sums[0] + 1.E-16),
                  (en_sums[next] - en_sums[0]) 
                / (en_sums[0] + 1.E-16))

            next = next + 1

        if (step % cnfg.save_freq == 0):

            data = nc.Dataset(
                save, "a", format="NETCDF3_64BIT_OFFSET")

            data.variables["uu_edge"][freq, :, :] = \
                np.reshape(uu_edge[
                    mesh.edge.irev - 1], (1, mesh.edge.size, 1))
            data.variables["hh_cell"][freq, :, :] = \
                np.reshape(hh_cell[
                    mesh.cell.irev - 1], (1, mesh.cell.size, 1))

            du_cell = trsk.cell_flux_sums * uu_edge
            du_cell/= mesh.cell.area

            data.variables["du_cell"][freq, :, :] = \
                np.reshape(du_cell[
                    mesh.cell.irev - 1], (1, mesh.cell.size, 1))

            data.variables["ke_cell"][freq, :, :] = \
                np.reshape(ke_cell[
                    mesh.cell.irev - 1], (1, mesh.cell.size, 1))
            data.variables["ke_dual"][freq, :, :] = \
                np.reshape(ke_dual[
                    mesh.vert.irev - 1], (1, mesh.vert.size, 1))            

            data.variables["pv_cell"][freq, :, :] = \
                np.reshape(pv_cell[
                    mesh.cell.irev - 1], (1, mesh.cell.size, 1))
            data.variables["rv_cell"][freq, :, :] = \
                np.reshape(rv_cell[
                    mesh.cell.irev - 1], (1, mesh.cell.size, 1))
            data.variables["pv_dual"][freq, :, :] = \
                np.reshape(pv_dual[
                    mesh.vert.irev - 1], (1, mesh.vert.size, 1))
            data.variables["rv_dual"][freq, :, :] = \
                np.reshape(rv_dual[
                    mesh.vert.irev - 1], (1, mesh.vert.size, 1))
            
            up_dual = trsk.dual_stub_sums * pv_bias
            up_dual = up_dual / mesh.vert.area

            data.variables["pv_bias"][freq, :, :] = \
                np.reshape(up_dual[
                    mesh.vert.irev - 1], (1, mesh.vert.size, 1))

            up_dual = trsk.dual_stub_sums * ke_bias
            up_dual = up_dual / mesh.vert.area

            data.variables["ke_bias"][freq, :, :] = \
                np.reshape(up_dual[
                    mesh.vert.irev - 1], (1, mesh.vert.size, 1))
            
            ui_cell = trsk.cell_lsqr_xnrm * uu_edge
            ux_dual = trsk.dual_kite_sums * ui_cell
            ux_dual = ux_dual / mesh.vert.area

            ui_cell = trsk.cell_lsqr_ynrm * uu_edge            
            uy_dual = trsk.dual_kite_sums * ui_cell
            uy_dual = uy_dual / mesh.vert.area

            ui_cell = trsk.cell_lsqr_znrm * uu_edge            
            uz_dual = trsk.dual_kite_sums * ui_cell
            uz_dual = uz_dual / mesh.vert.area

            data.variables["ux_dual"][freq, :, :] = \
                np.reshape(ux_dual[
                    mesh.vert.irev - 1], (1, mesh.vert.size, 1))
            data.variables["uy_dual"][freq, :, :] = \
                np.reshape(uy_dual[
                    mesh.vert.irev - 1], (1, mesh.vert.size, 1))
            data.variables["uz_dual"][freq, :, :] = \
                np.reshape(uz_dual[
                    mesh.vert.irev - 1], (1, mesh.vert.size, 1))

            data.close()

            freq = freq + 1

    ttoc = time.time()

    print("Total run time:", ttoc - ttic)
    print("tcpu.thickness:", tcpu.thickness)
    print("tcpu.momentum_:", tcpu.momentum_)
    print("tcpu.upwinding:", tcpu.upwinding) 
    print("tcpu.compute_H:", tcpu.compute_H)
    print("tcpu.computeKE:", tcpu.computeKE)    
    print("tcpu.computePV:", tcpu.computePV)
    print("tcpu.advect_PV:", tcpu.advect_PV)
    print("tcpu.computeDU:", tcpu.computeDU)

    data = nc.Dataset(
        save, "a", format="NETCDF3_64BIT_OFFSET")

    data.variables["kp_sums"][:] = kp_sums
    data.variables["en_sums"][:] = en_sums
    data.close()


def init_file(name, cnfg, save):

    mesh = load_mesh(name)  # so that there's no reindexing
    flow = load_flow(name)

    data = nc.Dataset(
        save, "w", format="NETCDF3_64BIT_OFFSET")
    data.on_a_sphere = "YES"
    data.sphere_radius = mesh.rsph
    data.is_periodic = "NO"
    data.source = "swe-python"

    data.createDimension(
        "Time", cnfg.iteration // cnfg.save_freq + 1)
    data.createDimension(
        "Step", cnfg.iteration // cnfg.stat_freq + 1)

    data.createDimension("TWO", 2)
    data.createDimension("nCells", mesh.cell.size)
    data.createDimension("nEdges", mesh.edge.size)
    data.createDimension("nVertices", mesh.vert.size)
    data.createDimension("nVertLevels", 1)
    data.createDimension("maxEdges", np.max(mesh.cell.topo) * 1)
    data.createDimension("maxEdges2", np.max(mesh.cell.topo) * 2)
    data.createDimension("vertexDegree", 3)

    data.createVariable("lonCell", "f8", ("nCells"))
    data["lonCell"][:] = mesh.cell.xlon
    data.createVariable("latCell", "f8", ("nCells"))
    data["latCell"][:] = mesh.cell.ylat
    data.createVariable("xCell", "f8", ("nCells"))
    data["xCell"][:] = mesh.cell.xpos
    data.createVariable("yCell", "f8", ("nCells"))
    data["yCell"][:] = mesh.cell.ypos
    data.createVariable("zCell", "f8", ("nCells"))
    data["zCell"][:] = mesh.cell.zpos
    data.createVariable("areaCell", "f8", ("nCells"))
    data["areaCell"][:] = mesh.cell.area
    data.createVariable(
        "verticesOnCell", "i4", ("nCells", "maxEdges"))
    data["verticesOnCell"][:, :] = mesh.cell.vert
    data.createVariable(
        "edgesOnCell", "i4", ("nCells", "maxEdges"))
    data["edgesOnCell"][:, :] = mesh.cell.edge
    data.createVariable(
        "cellsOnCell", "i4", ("nCells", "maxEdges"))
    data["cellsOnCell"][:, :] = mesh.cell.cell
    data.createVariable("nEdgesOnCell", "i4", ("nCells"))
    data["nEdgesOnCell"][:] = mesh.cell.topo

    data.createVariable("lonEdge", "f8", ("nEdges"))
    data["lonEdge"][:] = mesh.edge.xlon
    data.createVariable("latEdge", "f8", ("nEdges"))
    data["latEdge"][:] = mesh.edge.ylat
    data.createVariable("xEdge", "f8", ("nEdges"))
    data["xEdge"][:] = mesh.edge.xpos
    data.createVariable("yEdge", "f8", ("nEdges"))
    data["yEdge"][:] = mesh.edge.ypos
    data.createVariable("zEdge", "f8", ("nEdges"))
    data["zEdge"][:] = mesh.edge.zpos
    data.createVariable("dvEdge", "f8", ("nEdges"))
    data["dvEdge"][:] = mesh.edge.vlen
    data.createVariable("dcEdge", "f8", ("nEdges"))
    data["dcEdge"][:] = mesh.edge.clen
    data.createVariable(
        "verticesOnEdge", "i4", ("nEdges", "TWO"))
    data["verticesOnEdge"][:, :] = mesh.edge.vert
    data.createVariable(
        "weightsOnEdge", "f8", ("nEdges", "maxEdges2"))
    data["weightsOnEdge"][:, :] = mesh.edge.wmul
    data.createVariable(
        "cellsOnEdge", "i4", ("nEdges", "TWO"))
    data["cellsOnEdge"][:, :] = mesh.edge.cell
    data.createVariable(
        "edgesOnEdge", "i4", ("nEdges", "maxEdges2"))
    data["edgesOnEdge"][:, :] = mesh.edge.edge
    data.createVariable("nEdgesOnEdge", "i4", ("nEdges"))
    data["nEdgesOnEdge"][:] = mesh.edge.topo

    data.createVariable("lonVertex", "f8", ("nVertices"))
    data["lonVertex"][:] = mesh.vert.xlon
    data.createVariable("latVertex", "f8", ("nVertices"))
    data["latVertex"][:] = mesh.vert.ylat
    data.createVariable("xVertex", "f8", ("nVertices"))
    data["xVertex"][:] = mesh.vert.xpos
    data.createVariable("yVertex", "f8", ("nVertices"))
    data["yVertex"][:] = mesh.vert.ypos
    data.createVariable("zVertex", "f8", ("nVertices"))
    data["zVertex"][:] = mesh.vert.zpos
    data.createVariable("areaTriangle", "f8", ("nVertices"))
    data["areaTriangle"][:] = mesh.vert.area
    data.createVariable(
        "kiteAreasOnVertex", "f8", ("nVertices", "vertexDegree"))
    data["kiteAreasOnVertex"][:, :] = mesh.vert.kite
    data.createVariable(
        "edgesOnVertex", "i4", ("nVertices", "vertexDegree"))
    data["edgesOnVertex"][:, :] = mesh.vert.edge
    data.createVariable(
        "cellsOnVertex", "i4", ("nVertices", "vertexDegree"))
    data["cellsOnVertex"][:, :] = mesh.vert.cell
   
    data.createVariable("zb_cell", "f8", ("nCells"))
    data["zb_cell"][:] = flow.zb_cell
    data.createVariable("ff_cell", "f8", ("nCells"))
    data["ff_cell"][:] = flow.ff_cell
    data.createVariable("ff_edge", "f8", ("nEdges"))
    data["ff_edge"][:] = flow.ff_edge
    data.createVariable("ff_vert", "f8", ("nVertices"))
    data["ff_vert"][:] = flow.ff_vert

    data.createVariable(
        "u0_edge", "f8", ("nEdges", "nVertLevels"))
    data["u0_edge"].long_name = "Normal velocity initial conditions" 
    data["u0_edge"][:] = flow.uu_edge   
    data.createVariable(
        "h0_cell", "f8", ("nCells", "nVertLevels"))    
    data["h0_cell"].long_name = "Layer thickness initial conditions"
    data["h0_cell"][:] = flow.hh_cell

    data.createVariable(
        "uu_edge", "f8", ("Time", "nEdges", "nVertLevels"))
    data["uu_edge"].long_name = "Normal velocity on edges"    
    data.createVariable(
        "hh_cell", "f8", ("Time", "nCells", "nVertLevels"))    
    data["hh_cell"].long_name = "Layer thickness on cells"

    data.createVariable(
        "du_cell", "f4", ("Time", "nCells", "nVertLevels"))    
    data["du_cell"].long_name = \
        "Divergence of velocity on cells"

    data.createVariable(
        "ke_cell", "f4", ("Time", "nCells", "nVertLevels"))
    data["ke_cell"].long_name = "Kinetic energy on cells"
    data.createVariable(
        "ke_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data["ke_dual"].long_name = "Kinetic energy on duals"    
    data.createVariable(
        "ke_bias", "f4", ("Time", "nVertices", "nVertLevels"))
    data["ke_bias"].long_name = \
        "Upwind-bias for KE, averaged to duals"
   
    data.createVariable(
        "pv_cell", "f4", ("Time", "nCells", "nVertLevels"))
    data["pv_cell"].long_name = "Potential vorticity on cells"
    data.createVariable(
        "rv_cell", "f4", ("Time", "nCells", "nVertLevels"))
    data["rv_cell"].long_name = "Relative vorticity on cells"
    data.createVariable(
        "pv_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data["pv_dual"].long_name = "Potential vorticity on duals"
    data.createVariable(
        "rv_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data["rv_dual"].long_name = "Relative vorticity on duals"
    data.createVariable(
        "pv_bias", "f4", ("Time", "nVertices", "nVertLevels"))
    data["pv_bias"].long_name = \
        "Upwind-bias for PV, averaged to duals"

    data.createVariable(
        "ux_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data["ux_dual"].long_name = "x-component of velocity"
    data.createVariable(
        "uy_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data["uy_dual"].long_name = "y-component of velocity"
    data.createVariable(
        "uz_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data["uz_dual"].long_name = "z-component of velocity"

    data.createVariable("kp_sums", "f8", ("Step"))
    data["kp_sums"].long_name = \
        "Relative change in total KE+PE over time"
    data.createVariable("en_sums", "f8", ("Step"))
    data["en_sums"].long_name = \
        "Relative change in total PV**2 over time"

    data.close()


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--mpas-file", dest="mpas_file", type=str,
        required=True, help="Path to user MPAS file.")

    parser.add_argument(
        "--time-step", dest="time_step", type=float,
        required=True, help="Length of time steps.")

    parser.add_argument(
        "--num-steps", dest="iteration", type=int,
        required=True, help="Number of time steps.")

    parser.add_argument(
        "--integrate", dest="integrate", type=str,
        default="RK32-FB",
        required=False, 
        help="Time integration = {RK32-FB}, RK22-FB.")

    parser.add_argument(
        "--pv-upwind", dest="pv_upwind", type=str,
        default="AUST-adapt",
        required=False, 
        help="Upstream formulation for PV = " + 
             "{AUST-adapt}, AUST-const, APVM, LUST.")

    parser.add_argument(
        "--pv-min-up", dest="pv_min_up", type=float,
        default=1./30.,
        required=False,
        help="Upwind PV.-flux bias {BIAS = +1./30.}.")

    parser.add_argument(
        "--pv-max-up", dest="pv_max_up", type=float,
        default=4./ 9.,
        required=False,
        help="Upwind PV.-flux bias {BIAS = +4./ 9.}.")
    
    parser.add_argument(
        "--pv-scheme", dest="pv_scheme", type=str,
        default="UPWIND",
        required=False, 
        help="PV.-flux formulation = {UPWIND}, CENTRE.")

    parser.add_argument(
        "--ke-upwind", dest="ke_upwind", type=str,
        default="AUST-const",
        required=False, 
        help="Upstream formulation for KE = " + 
             "{AUST-const}, AUST-adapt, APVM, LUST.")

    parser.add_argument(
        "--ke-min-up", dest="ke_min_up", type=float,
        default=1./90.,
        required=False,
        help="Upwind KE.-edge bias {BIAS = +1./90.}.")

    parser.add_argument(
        "--ke-max-up", dest="ke_max_up", type=float,
        default=1./15.,
        required=False,
        help="Upwind KE.-edge bias {BIAS = +1./15.}.")

    parser.add_argument(
        "--ke-scheme", dest="ke_scheme", type=str,
        default="CENTRE",
        required=False, 
        help="KE.-grad formulation = {CENTRE}, " +
                                     "CENTRE+SKINNY," +
                                     "UPWIND, " + 
                                     "UPWIND+SKINNY.")

    parser.add_argument(
        "--du-damp-2", dest="du_damp_2", type=float,
        default=1.E+00,
        required=False,
        help="DIV^2 damping coeff. {DAMP = +1.E+00}.")

    parser.add_argument(
        "--du-damp-4", dest="du_damp_4", type=float,
        default=1.E+00,
        required=False,
        help="DIV^4 damping coeff. {DAMP = +1.E+00}.")

    parser.add_argument(
        "--vu-damp-2", dest="vu_damp_2", type=float,
        default=0.E+00,
        required=False,
        help="DEL^2 damping coeff. {DAMP = +0.E+00}.")

    parser.add_argument(
        "--vu-damp-4", dest="vu_damp_4", type=float,
        default=0.E+00,
        required=False,
        help="DEL^4 damping coeff. {DAMP = +0.E+00}.")

    parser.add_argument(
        "--operators", dest="operators", type=str,
        default="TRSK-CV",
        required=False, 
        help="Discretisation = {TRSK-CV}, TRSK-MD.")

    parser.add_argument(
        "--save-freq", dest="save_freq", type=int,
        required=True, help="Save each FREQ-th step.")

    parser.add_argument(
        "--stat-freq", dest="stat_freq", type=int,
        required=False, 
        default=10000, help="Prints at FREQ-th step.")

    parser.add_argument(
        "--no-advect", dest="no_advect", 
        type=lambda x: bool(strtobool(str(x.strip()))),
        required=False, 
        default=False, help="Disable mom.-advection.")

    parser.add_argument(
        "--no-rotate", dest="no_rotate", 
        type=lambda x: bool(strtobool(str(x.strip()))),
        required=False, 
        default=False, help="Disable coriolis terms.")

    parser.add_argument(
        "--FB-weight", dest="fb_weight", type=float,
        required=False,
        nargs='*',
        help="Forward-backward weights for integrators.")


    swe(parser.parse_args())


