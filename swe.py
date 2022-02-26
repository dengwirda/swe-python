
import os
import time
import numpy as np
import netCDF4 as nc
import argparse

from msh import load_mesh, load_flow
from ops import trsk_mats


HH_TINY        = 1.0E-02
UU_TINY        = 1.0E-16

class base: pass
tcpu = base()
tcpu.thickness = 0.0E+00
tcpu.momentum_ = 0.0E+00
tcpu.upwinding = 0.0E+00
tcpu.compute_H = 0.0E+00
tcpu.computeKE = 0.0E+00
tcpu.computePV = 0.0E+00
tcpu.advect_PV = 0.0E+00
tcpu.computeDU = 0.0E+00

def swe(cnfg):
    """
    SWE: solve the nonlinear SWE on generalised MPAS meshes.

    """
    # Authors: Darren Engwirda

    cnfg.stat_freq = np.minimum(
        cnfg.save_freq, cnfg.stat_freq)
    
    cnfg.integrate = cnfg.integrate.upper()
    cnfg.operators = cnfg.operators.upper()
    cnfg.up_scheme = cnfg.up_scheme.upper()
    cnfg.ke_scheme = cnfg.ke_scheme.upper()
    cnfg.pv_scheme = cnfg.pv_scheme.upper()

    name = cnfg.mpas_file
    path, file = os.path.split(name)
    save = os.path.join(path, "out_" + file)

    # load mesh + init. conditions
    mesh = load_mesh(name, sort=True)
    flow = load_flow(name, mesh=mesh)

    init_file(name, cnfg, save)

    u0_edge = flow.uu_edge[-1, :, 0]
    uu_edge = u0_edge
    h0_cell = flow.hh_cell[-1, :, 0]
    hh_cell = h0_cell

    hh_cell = np.maximum(HH_TINY, hh_cell)

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
        cnfg.iteration // cnfg.stat_freq + 1), dtype=np.float64)
    pv_sums = np.zeros((
        cnfg.iteration // cnfg.stat_freq + 1), dtype=np.float64)

    ttic = time.time(); xout = []; next = 0; freq = 0

    for step in range(cnfg.iteration + 1):

        if ("RK11" in cnfg.integrate):

            hh_cell, uu_edge, ke_cell, ke_dual, \
            rv_cell, pv_cell, \
            rv_dual, pv_dual, \
            ke_bias, pv_bias = step_RK11(
                mesh, trsk, flow, cnfg, hh_cell, uu_edge)

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

        if (step % cnfg.stat_freq == 0):

            kp_sums[next], \
            pv_sums[next] = invariant(
                mesh, trsk, flow, cnfg, hh_cell, uu_edge)

            print("step, KE+PE, PV**2:", step,
                  (kp_sums[next] - kp_sums[0]) / kp_sums[0],
                  (pv_sums[next] - pv_sums[0]) / pv_sums[0])

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

            dh_cell = hh_cell - h0_cell

            data.variables["dh_cell"][freq, :, :] = \
                np.reshape(dh_cell[
                    mesh.cell.irev - 1], (1, mesh.cell.size, 1))

            data.close()

            freq = freq + 1

    ttoc = time.time()

    print("TCPU:", ttoc - ttic)
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

    data.variables["kk_sums"][:] = kp_sums
    data.variables["pv_sums"][:] = pv_sums

    return


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
        "uu_edge", "f8", ("Time", "nEdges", "nVertLevels"))
    data.createVariable(
        "hh_cell", "f8", ("Time", "nCells", "nVertLevels"))
    data.createVariable(
        "dh_cell", "f4", ("Time", "nCells", "nVertLevels"))

    data.createVariable(
        "ke_cell", "f4", ("Time", "nCells", "nVertLevels"))
    data.createVariable(
        "ke_dual", "f4", ("Time", "nVertices", "nVertLevels"))    
    data.createVariable(
        "ke_bias", "f4", ("Time", "nVertices", "nVertLevels"))
    
    data.createVariable(
        "pv_cell", "f4", ("Time", "nCells", "nVertLevels"))
    data.createVariable(
        "rv_cell", "f4", ("Time", "nCells", "nVertLevels"))
    data.createVariable(
        "pv_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data.createVariable(
        "rv_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data.createVariable(
        "pv_bias", "f4", ("Time", "nVertices", "nVertLevels"))

    data.createVariable(
        "ux_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data.createVariable(
        "uy_dual", "f4", ("Time", "nVertices", "nVertLevels"))
    data.createVariable(
        "uz_dual", "f4", ("Time", "nVertices", "nVertLevels"))

    data.createVariable("kk_sums", "f8", ("Step"))
    data.createVariable("pv_sums", "f8", ("Step"))

    data.close()


def invariant(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

    ff_dual = flow.ff_vert
    ff_edge = flow.ff_edge
    ff_cell = flow.ff_cell

    zb_cell = flow.zb_cell

    vv_edge = trsk.edge_lsqr_perp * uu_edge * +1.

    hh_dual, \
    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    ke_edge =  0.5 * uu_edge ** 2

    ke_edge = ke_edge * hh_edge * mesh.edge.clen \
                                * mesh.edge.vlen

    pe_cell = flow.grav * (hh_cell * 0.5 + zb_cell)

    pe_cell = pe_cell * hh_cell * mesh.cell.area

    kk_sums = np.sum(ke_edge) + np.sum(pe_cell)

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, __ = computePV(
        mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, uu_edge, vv_edge,
        ff_dual, ff_edge, ff_cell, 
        +0.0 * cnfg.time_step, 0.0 * cnfg.pv_upwind)

   #pv_sums = np.sum(
   #    +0.5 * mesh.edge.area * hh_edge * pv_edge ** 2)

    pv_sums = np.sum(
        +0.5 * mesh.vert.area * hh_dual * pv_dual ** 2)

    return kk_sums, pv_sums


def step_RK11(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

#-- standard FB.-RK(1,1) method

    ff_cell = flow.ff_cell
    ff_edge = flow.ff_edge
    ff_dual = flow.ff_vert

    zb_cell = flow.zb_cell

#-- 1st RK + FB stage

    ttic = time.time()

    BETA = (1.0 / 1.0) * ("FB" in cnfg.integrate)

    vv_edge = trsk.edge_lsqr_perp * uu_edge

    hh_dual, \
    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    uh_edge = uu_edge * hh_edge

    uh_cell = trsk.cell_flux_sums * uh_edge
    uh_cell/= mesh.cell.area

    h1_cell = (
        hh_cell - 1.0 / 1.0 * cnfg.time_step * uh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    hb_cell = h1_cell * (0.0 + 1.0 * BETA) + \
              hh_cell * (1.0 - 1.0 * BETA)

    hb_dual, \
    hb_edge = compute_H(mesh, trsk, cnfg, hb_cell, uu_edge)

    uh_edge = uu_edge * hb_edge

    ke_dual, ke_cell, ke_bias = computeKE(
        mesh, trsk, cnfg, 
        hb_cell, hb_edge, hb_dual, 
        uu_edge, vv_edge,
        +0.0 / 2.0 * cnfg.time_step, cnfg.ke_upwind)

    hk_cell = hb_cell + zb_cell 
    hk_cell = ke_cell + hk_cell * flow.grav

    hk_grad = trsk.edge_grad_norm * hk_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, trsk, cnfg, 
        hb_cell, hb_edge, hb_dual, uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step, cnfg.pv_upwind)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, uu_edge)

    u1_edge = uu_edge - 1.0 / 1.0 * cnfg.time_step * (
        hk_grad + qh_flux - uu_damp
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

    return h1_cell, u1_edge, ke_cell, ke_dual, \
           rv_cell, pv_cell, \
           rv_dual, pv_dual, ke_bias, pv_bias


def step_RK22(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

#-- A 2-stage RK2 + FB scheme, a'la ROMS:
#-- A.F. Shchepetkin, J.C. McWilliams (2005): The regional oceanic 
#-- modeling system (ROMS): a split-explicit, free-surface, 
#-- topography-following-coordinate oceanic model
#-- doi.org/10.1016/j.ocemod.2004.08.002

#-- but with thickness updated via an SSP-RK2 approach

    ff_cell = flow.ff_cell
    ff_edge = flow.ff_edge
    ff_dual = flow.ff_vert

    zb_cell = flow.zb_cell

#-- 1st RK + FB stage

    ttic = time.time()

    BETA = (1.0 / 3.0) * ("FB" in cnfg.integrate)

    vv_edge = trsk.edge_lsqr_perp * uu_edge

    hh_dual, \
    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    uh_edge = uu_edge * hh_edge

    uh_cell = trsk.cell_flux_sums * uh_edge
    uh_cell/= mesh.cell.area

    h1_cell = (
        hh_cell - 1.0 / 1.0 * cnfg.time_step * uh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    hb_cell = h1_cell * (0.0 + 1.0 * BETA) + \
              hh_cell * (1.0 - 1.0 * BETA)

    hb_dual, \
    hb_edge = compute_H(mesh, trsk, cnfg, hb_cell, uu_edge)

    uh_edge = uu_edge * hb_edge

    ke_dual, ke_cell, __ = computeKE(
        mesh, trsk, cnfg, 
        hb_cell, hb_edge, hb_dual, 
        uu_edge, vv_edge,
        +1.0 / 2.0 * cnfg.time_step, cnfg.ke_upwind)

    hk_cell = hb_cell + zb_cell 
    hk_cell = ke_cell + hk_cell * flow.grav

    hk_grad = trsk.edge_grad_norm * hk_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, __ = computePV(
        mesh, trsk, cnfg, 
        hb_cell, hb_edge, hb_dual, uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step, cnfg.pv_upwind)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, uu_edge)

    u1_edge = uu_edge - 1.0 / 1.0 * cnfg.time_step * (
        hk_grad + qh_flux - uu_damp
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 2nd RK + FB stage

    ttic = time.time()

    BETA = (2.0 / 3.0) * ("FB" in cnfg.integrate)

    hm_cell = 0.5 * hh_cell + 0.5 * h1_cell
    um_edge = 0.5 * uu_edge + 0.5 * u1_edge

    vm_edge = trsk.edge_lsqr_perp * um_edge

    h1_dual, \
    h1_edge = compute_H(mesh, trsk, cnfg, h1_cell, u1_edge)

    uh_edge = u1_edge * h1_edge

    uh_cell = trsk.cell_flux_sums * uh_edge
    uh_cell/= mesh.cell.area

    h2_cell = (
        hm_cell - 1.0 / 2.0 * cnfg.time_step * uh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    hb_cell = h2_cell * (0.0 + 0.5 * BETA) + \
              h1_cell * (0.5 - 0.5 * BETA) + \
              hh_cell * (0.5)

    hb_dual, \
    hb_edge = compute_H(mesh, trsk, cnfg, hb_cell, um_edge)

    uh_edge = um_edge * hb_edge

    ke_dual, ke_cell, ke_bias = computeKE(
        mesh, trsk, cnfg, 
        hb_cell, hb_edge, hb_dual, 
        um_edge, vm_edge,
        +1.0 / 2.0 * cnfg.time_step, cnfg.ke_upwind)

    hk_cell = hb_cell + zb_cell 
    hk_cell = ke_cell + hk_cell * flow.grav

    hk_grad = trsk.edge_grad_norm * hk_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, trsk, cnfg, 
        hb_cell, hb_edge, hb_dual, um_edge, vm_edge,
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step, cnfg.pv_upwind)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, um_edge)

    u2_edge = uu_edge - 1.0 / 1.0 * cnfg.time_step * (
        hk_grad + qh_flux - uu_damp
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

    return h2_cell, u2_edge, ke_cell, ke_dual, \
           rv_cell, pv_cell, \
           rv_dual, pv_dual, ke_bias, pv_bias


def step_RK32(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

#-- A 3-stage RK2 + FB scheme, a'la MPAS-A:
#-- L.J. Wicker, W.C. Skamarock (2002): Time-Splitting Methods for 
#-- Elastic Models Using Forward Time Schemes
#-- doi.org/10.1175/1520-0493(2002)130<2088:TSMFEM>2.0.CO;2

#-- but with FB weighting applied within each RK stage

    ff_cell = flow.ff_cell
    ff_edge = flow.ff_edge
    ff_dual = flow.ff_vert

    zb_cell = flow.zb_cell

#-- 1st RK + FB stage

    ttic = time.time()

    BETA = (1.0 / 3.0) * ("FB" in cnfg.integrate)

    vv_edge = trsk.edge_lsqr_perp * uu_edge

    hh_dual, \
    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    uh_edge = uu_edge * hh_edge

    uh_cell = trsk.cell_flux_sums * uh_edge
    uh_cell/= mesh.cell.area

    h1_cell = (
        hh_cell - 1.0 / 3.0 * cnfg.time_step * uh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    hb_cell = h1_cell * (0.0 + 1.0 * BETA) + \
              hh_cell * (1.0 - 1.0 * BETA)

    hb_dual, \
    hb_edge = compute_H(mesh, trsk, cnfg, hb_cell, uu_edge)

    uh_edge = uu_edge * hb_edge

    ke_dual, ke_cell, __ = computeKE(
        mesh, trsk, cnfg, 
        hb_cell, hb_edge, hb_dual, 
        uu_edge, vv_edge,
        +1.0 / 2.0 * cnfg.time_step, cnfg.ke_upwind)

    hk_cell = hb_cell + zb_cell 
    hk_cell = ke_cell + hk_cell * flow.grav

    hk_grad = trsk.edge_grad_norm * hk_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, __ = computePV(
        mesh, trsk, cnfg, 
        hb_cell, hb_edge, hb_dual, uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step, cnfg.pv_upwind)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, uu_edge)

    u1_edge = uu_edge - 1.0 / 3.0 * cnfg.time_step * (
        hk_grad + qh_flux - uu_damp
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 2nd RK + FB stage

    ttic = time.time()

    BETA = (1.0 / 2.0) * ("FB" in cnfg.integrate)

    v1_edge = trsk.edge_lsqr_perp * u1_edge

    h1_dual, \
    h1_edge = compute_H(mesh, trsk, cnfg, h1_cell, u1_edge)

    uh_edge = u1_edge * h1_edge

    uh_cell = trsk.cell_flux_sums * uh_edge
    uh_cell/= mesh.cell.area

    h2_cell = (
        hh_cell - 1.0 / 2.0 * cnfg.time_step * uh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    # centred at 1/4 for 1/2 step?
    hb_cell = h2_cell * (0.0 + 1.0 * BETA) + \
              hh_cell * (1.0 - 1.0 * BETA)

    hb_dual, \
    hb_edge = compute_H(mesh, trsk, cnfg, hb_cell, u1_edge)

    uh_edge = u1_edge * hb_edge

    ke_dual, ke_cell, __ = computeKE(
        mesh, trsk, cnfg, 
        hb_cell, hb_edge, hb_dual, 
        u1_edge, v1_edge,
        +1.0 / 2.0 * cnfg.time_step, cnfg.ke_upwind)

    hk_cell = hb_cell + zb_cell 
    hk_cell = ke_cell + hk_cell * flow.grav

    hk_grad = trsk.edge_grad_norm * hk_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, __ = computePV(
        mesh, trsk, cnfg, 
        hb_cell, hb_edge, hb_dual, u1_edge, v1_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step, cnfg.pv_upwind)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, u1_edge)

    u2_edge = uu_edge - 1.0 / 2.0 * cnfg.time_step * (
        hk_grad + qh_flux - uu_damp
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 3rd RK + FB stage

    ttic = time.time()

    BETA = (89. / 300) * ("FB" in cnfg.integrate)
   #BETA = (0.2969771) * ("FB" in cnfg.integrate)
    
    v2_edge = trsk.edge_lsqr_perp * u2_edge

    h2_dual, \
    h2_edge = compute_H(mesh, trsk, cnfg, h2_cell, u2_edge)

    uh_edge = u2_edge * h2_edge

    uh_cell = trsk.cell_flux_sums * uh_edge
    uh_cell/= mesh.cell.area

    h3_cell = (
        hh_cell - 1.0 / 1.0 * cnfg.time_step * uh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    # centred at 1/2 for 1/1 step?
    hb_cell = h3_cell * (0.0 + 1.0 * BETA) + \
              h2_cell * (1.0 - 2.0 * BETA) + \
              hh_cell * (0.0 + 1.0 * BETA)

    hb_dual, \
    hb_edge = compute_H(mesh, trsk, cnfg, hb_cell, u2_edge)

    uh_edge = u2_edge * hb_edge

    ke_dual, ke_cell, ke_bias = computeKE(
        mesh, trsk, cnfg, 
        hb_cell, hb_edge, hb_dual, 
        u2_edge, v2_edge,
        +1.0 / 2.0 * cnfg.time_step, cnfg.ke_upwind)

    hk_cell = hb_cell + zb_cell 
    hk_cell = ke_cell + hk_cell * flow.grav

    hk_grad = trsk.edge_grad_norm * hk_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, trsk, cnfg, 
        hb_cell, hb_edge, hb_dual, u2_edge, v2_edge,
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step, cnfg.pv_upwind)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, u2_edge)

    u3_edge = uu_edge - 1.0 / 1.0 * cnfg.time_step * (
        hk_grad + qh_flux - uu_damp
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

    return h3_cell, u3_edge, ke_cell, ke_dual, \
           rv_cell, pv_cell, \
           rv_dual, pv_dual, ke_bias, pv_bias


def step_SP33(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

#-- standard SSP-RK(3,3) method

    ff_cell = flow.ff_cell
    ff_edge = flow.ff_edge
    ff_dual = flow.ff_vert

    zb_cell = flow.zb_cell

#-- 1st RK + FB stage

    ttic = time.time()

    vv_edge = trsk.edge_lsqr_perp * uu_edge

    hh_dual, \
    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    uh_edge = uu_edge * hh_edge

    uh_cell = trsk.cell_flux_sums * uh_edge
    uh_cell/= mesh.cell.area

    h1_cell = (
        hh_cell - 1.0 / 1.0 * cnfg.time_step * uh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    ke_dual, ke_cell, __ = computeKE(
        mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, 
        uu_edge, vv_edge,
        +1.0 / 2.0 * cnfg.time_step, cnfg.ke_upwind)

    hk_cell = hh_cell + zb_cell 
    hk_cell = ke_cell + hk_cell * flow.grav

    hk_grad = trsk.edge_grad_norm * hk_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, __ = computePV(
        mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step, cnfg.pv_upwind)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, uu_edge)

    u1_edge = uu_edge - 1.0 / 1.0 * cnfg.time_step * (
        hk_grad + qh_flux - uu_damp
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 2nd RK + FB stage

    ttic = time.time()

    v1_edge = trsk.edge_lsqr_perp * u1_edge

    h1_dual, \
    h1_edge = compute_H(mesh, trsk, cnfg, h1_cell, u1_edge)

    uh_edge = u1_edge * h1_edge

    uh_cell = trsk.cell_flux_sums * uh_edge
    uh_cell/= mesh.cell.area

    h2_cell = (
        3.0 / 4.0 * hh_cell + 
        1.0 / 4.0 * h1_cell - 
            1.0 / 4.0 * cnfg.time_step * uh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    ke_dual, ke_cell, __ = computeKE(
        mesh, trsk, cnfg, 
        h1_cell, h1_edge, h1_dual, 
        u1_edge, v1_edge,
        +1.0 / 2.0 * cnfg.time_step, cnfg.ke_upwind)

    hk_cell = h1_cell + zb_cell 
    hk_cell = ke_cell + hk_cell * flow.grav

    hk_grad = trsk.edge_grad_norm * hk_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, __ = computePV(
        mesh, trsk, cnfg, 
        h1_cell, h1_edge, h1_dual, u1_edge, v1_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step, cnfg.pv_upwind)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, u1_edge)

    u2_edge = (
        3.0 / 4.0 * uu_edge +
        1.0 / 4.0 * u1_edge - 
            1.0 / 4.0 * cnfg.time_step * (
                hk_grad + qh_flux - uu_damp
    ))

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 3rd RK + FB stage

    ttic = time.time()

    v2_edge = trsk.edge_lsqr_perp * u2_edge

    h2_dual, \
    h2_edge = compute_H(mesh, trsk, cnfg, h2_cell, u2_edge)

    uh_edge = u2_edge * h2_edge

    uh_cell = trsk.cell_flux_sums * uh_edge
    uh_cell/= mesh.cell.area

    h3_cell = (
        1.0 / 3.0 * hh_cell +
        2.0 / 3.0 * h2_cell - 
            2.0 / 3.0 * cnfg.time_step * uh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    ke_dual, ke_cell, ke_bias = computeKE(
        mesh, trsk, cnfg, 
        h2_cell, h2_edge, h2_dual, 
        u2_edge, v2_edge,
        +1.0 / 2.0 * cnfg.time_step, cnfg.ke_upwind)

    hk_cell = h2_cell + zb_cell 
    hk_cell = ke_cell + hk_cell * flow.grav

    hk_grad = trsk.edge_grad_norm * hk_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, trsk, cnfg, 
        h2_cell, h2_edge, h2_dual, u2_edge, v2_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step, cnfg.pv_upwind)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, u2_edge)

    u3_edge = (
        1.0 / 3.0 * uu_edge +
        2.0 / 3.0 * u2_edge - 
            2.0 / 3.0 * cnfg.time_step * (
                hk_grad + qh_flux - uu_damp
    ))

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

    return h3_cell, u3_edge, ke_cell, ke_dual, \
           rv_cell, pv_cell, \
           rv_dual, pv_dual, ke_bias, pv_bias


def hrmn_mean(xone, xtwo):

#-- harmonic mean of two vectors (ie. biased toward lesser)

    return +2.0 * xone * xtwo / (xone + xtwo)


def upwinding(mesh, trsk, cnfg, 
              sv_dual, sm_dual, sv_cell, sv_edge,
              uu_edge, vv_edge, 
              delta_t, up_bias):

    ttic = time.time()

    if (cnfg.up_scheme == "LUST"):

    #-- upwind bias, a'la Weller
    #-- H. Weller (2012): Controlling the computational modes 
    #-- of the arbitrarily structured C grid
    #-- https://doi.org/10.1175/MWR-D-11-00221.1
        dv_edge = trsk.edge_grad_perp * sv_dual

        dx_dual = trsk.dual_lsqr_xprp * dv_edge
        dy_dual = trsk.dual_lsqr_yprp * dv_edge
        dz_dual = trsk.dual_lsqr_zprp * dv_edge

        sv_bias = np.zeros(mesh.edge.size)

        up_edge = np.where(vv_edge <= 0.0)
        dn_edge = np.where(vv_edge >= 0.0)

        up_dual = np.zeros(
            mesh.edge.size, dtype=np.int32)
        up_dual[up_edge] = \
            mesh.edge.vert[up_edge, 1] - 1
        up_dual[dn_edge] = \
            mesh.edge.vert[dn_edge, 0] - 1

        up_xdel = \
            mesh.edge.xpos - mesh.vert.xmid[up_dual]
        up_ydel = \
            mesh.edge.ypos - mesh.vert.ymid[up_dual]
        up_zdel = \
            mesh.edge.zpos - mesh.vert.zmid[up_dual]

        va_edge = np.abs(vv_edge) + UU_TINY
        ua_edge = np.abs(uu_edge) + UU_TINY
        
        BIAS = (
            up_bias * va_edge / (ua_edge + va_edge)
        )

        sv_wind = sv_dual[up_dual] + \
            up_xdel * dx_dual[up_dual] + \
            up_ydel * dy_dual[up_dual] + \
            up_zdel * dz_dual[up_dual]

        sv_edge = \
            BIAS * sv_wind + (1.0 - BIAS) * sv_edge

    if (cnfg.up_scheme == "APVM"):

        dn_edge = trsk.edge_grad_norm * sv_cell
        dp_edge = trsk.edge_grad_perp * sv_dual

        sv_bias = np.zeros(mesh.edge.size)

    #-- upwind APVM, scale w time
        up_bias = 1.0                   # hard-coded?
        sv_apvm = up_bias * uu_edge * dn_edge + \
                  up_bias * vv_edge * dp_edge

        sv_edge = sv_edge - delta_t * sv_apvm
        
    if (cnfg.up_scheme == "UUST"):

        dn_edge = trsk.edge_grad_norm * sv_cell
        dp_edge = trsk.edge_grad_perp * sv_dual

        sv_bias = np.zeros(mesh.edge.size)

    #-- upwind APVM, scale w grid
        um_edge = UU_TINY + \
            np.sqrt(uu_edge ** 2 + vv_edge ** 2)

        sv_wind = uu_edge * dn_edge / um_edge + \
                  vv_edge * dp_edge / um_edge

        ee_scal = mesh.edge.slen

        sv_edge-= up_bias * ee_scal * sv_wind

    if (cnfg.up_scheme == "AUST"):
        
    #-- AUST: anticipated upstream method; APVM meets
    #-- LUST? Upwinds in multi-dimensional sense, vs.
    #-- LUST, which upwinds via tangential dir. only.

        dn_edge = trsk.edge_grad_norm * sv_cell
        dp_edge = trsk.edge_grad_perp * sv_dual

        dm_edge = 0.5 * (
            np.abs(dn_edge * mesh.edge.clen) +
            np.abs(dp_edge * mesh.edge.vlen)
        )

        ds_dual = np.abs(sv_dual - sm_dual)

        ds_edge = \
            (trsk.edge_vert_sums * ds_dual) / 2.0
       
        sv_bias = ds_edge / (dm_edge + UU_TINY)

    #-- upwind APVM, scale w grid
        um_edge = UU_TINY + \
            np.sqrt(uu_edge ** 2 + vv_edge ** 2)

        sv_wind = uu_edge * dn_edge / um_edge + \
                  vv_edge * dp_edge / um_edge

        ee_scal = mesh.edge.slen

        sv_bias = up_bias \
            + np.minimum(+1./3. - up_bias, sv_bias)

        sv_edge-= sv_bias * ee_scal * sv_wind

    ttoc = time.time()
    tcpu.upwinding = tcpu.upwinding + (ttoc - ttic)

    return sv_edge, sv_bias


def compute_H(mesh, trsk, cnfg, hh_cell, uu_edge):

    ttic = time.time()

    if (cnfg.operators == "TRSK-CV"):

        hh_dual = trsk.dual_kite_sums * hh_cell
        hh_dual/= mesh.vert.area

        hh_edge = trsk.edge_wing_sums * hh_cell
        hh_edge/= mesh.edge.area

    if (cnfg.operators == "TRSK-MD"):

        hh_dual = trsk.dual_kite_sums * hh_cell
        hh_dual/= mesh.vert.area

        hh_edge = trsk.edge_cell_sums * hh_cell
        hh_edge*= 0.5E+00

    ttoc = time.time()
    tcpu.compute_H = tcpu.compute_H + (ttoc - ttic)

    return hh_dual, hh_edge


def computePV(mesh, trsk, cnfg, 
              hh_cell, hh_edge, hh_dual, uu_edge, vv_edge, 
              ff_dual, ff_edge, ff_cell,
              delta_t, up_bias):

    ttic = time.time()

    if (cnfg.operators == "TRSK-CV"):

    #-- RV+f on rhombi, PV on edge - more compact stencil?        
        rv_dual = trsk.dual_curl_sums * uu_edge
        rv_dual/= mesh.vert.area
        
        av_dual = rv_dual + ff_dual
        pv_dual = av_dual / hh_dual
        
        rv_cell = trsk.cell_kite_sums * rv_dual
        rv_cell/= mesh.cell.area

        av_cell = rv_cell + ff_cell
        pv_cell = av_cell / hh_cell

    #-- compute curl on rhombi -- a'la Gassmann
        rm_edge = trsk.quad_curl_sums * uu_edge
        rm_edge/= mesh.quad.area

        hv_edge = trsk.edge_stub_sums * hh_dual
        hv_edge/= mesh.edge.area

        hm_edge = 1. / 2. * hh_edge \
                + 1. / 2. * hv_edge
        
        am_edge = rm_edge + ff_edge
        pm_edge = am_edge / hm_edge

    #-- average rhombi to dual -- a'la Gassmann
        rm_dual = trsk.dual_edge_sums * rm_edge / 3.0

        am_dual = rm_dual + ff_dual
        pm_dual = am_dual / hh_dual

    #-- compute high(er)-order RV + PV on edges:
    #-- pv_edge = pv_dual + (xe - xv) * pv_d/dx
        pv_edge = trsk.edge_vert_sums * pv_dual / 2.0
        pv_edge+= trsk.edge_dual_reco * pm_dual

        pv_edge = 3. / 8. * pv_edge \
                + 5. / 8. * pm_edge

        pv_edge, up_edge = upwinding(
            mesh, trsk, cnfg, 
            pv_dual, pm_dual, pv_cell, pv_edge,
            uu_edge, vv_edge, delta_t, up_bias)
        
    if (cnfg.operators == "TRSK-MD"):

        rv_dual = trsk.dual_curl_sums * uu_edge
        rv_dual/= mesh.vert.area

        av_dual = rv_dual + ff_dual
        pv_dual = av_dual / hh_dual

        rv_cell = trsk.cell_kite_sums * rv_dual
        rv_cell/= mesh.cell.area

        av_cell = rv_cell + ff_cell
        pv_cell = av_cell / hh_cell

        rv_edge = trsk.edge_vert_sums * rv_dual / 2.0
        pv_edge = trsk.edge_vert_sums * pv_dual / 2.0
        
        rm_dual = trsk.dual_edge_sums * rv_edge / 3.0
        
        am_dual = rm_dual + ff_dual
        pm_dual = am_dual / hh_dual

        pv_edge, up_edge = upwinding(
            mesh, trsk, cnfg, 
            pv_dual, pm_dual, pv_cell, pv_edge, 
            uu_edge, vv_edge, delta_t, up_bias)

    ttoc = time.time()
    tcpu.computePV = tcpu.computePV + (ttoc - ttic)

    return rv_dual, pv_dual, rv_cell, pv_cell, \
           pv_edge, up_edge


def computeKE(mesh, trsk, cnfg, 
              hh_cell, hh_edge, hh_dual, uu_edge, vv_edge,
              delta_t, up_bias):

    ttic = time.time()

    if (cnfg.operators == "TRSK-CV"):

        up_edge = np.zeros(mesh.edge.size, dtype=float)

        if ("WEIGHT" in cnfg.ke_scheme):

            hh_thin = 1.0E+02

            ke_edge = 0.5 * (uu_edge ** 2 + 
                             vv_edge ** 2 )
 
            hh_scal = hh_thin / hh_edge

            ke_edge*= ((1.0 + hh_scal) ** 2) ** 2

            ke_cell = trsk.cell_wing_sums * ke_edge
            ke_cell/= mesh.cell.area

            ke_dual = trsk.dual_stub_sums * ke_edge
            ke_dual/= mesh.vert.area

            hh_scal = hh_thin / hh_cell

            ke_cell/= ((1.0 + hh_scal) ** 2) ** 2

        else:  # CENTRE

            ke_edge = 0.5 * (uu_edge ** 2 + 
                             vv_edge ** 2 )

            ke_dual = trsk.dual_stub_sums * ke_edge
            ke_dual/= mesh.vert.area

            ke_cell = trsk.cell_wing_sums * ke_edge
            ke_cell/= mesh.cell.area

    if (cnfg.operators == "TRSK-MD"):

        up_edge = np.zeros(mesh.edge.size, dtype=float)

        if ("WEIGHT" in cnfg.ke_scheme):

            hh_thin = 1.0E+02

            ke_edge = 0.25 * uu_edge ** 2 * \
                  mesh.edge.clen * \
                  mesh.edge.vlen

            hh_scal = hh_thin / hh_edge

            ke_edge*= ((1.0 + hh_scal) ** 2) ** 2

            ke_cell = trsk.cell_edge_sums * ke_edge
            ke_cell/= mesh.cell.area

            ke_dual = trsk.dual_edge_sums * ke_edge
            ke_dual/= mesh.vert.area

            hh_scal = hh_thin / hh_cell

            ke_cell/= ((1.0 + hh_scal) ** 2) ** 2

        else:  # CENTRE

            ke_edge = 0.25 * uu_edge ** 2 * \
                  mesh.edge.clen * \
                  mesh.edge.vlen

            ke_dual = trsk.dual_edge_sums * ke_edge
            ke_dual/= mesh.vert.area

            ke_cell = trsk.cell_edge_sums * ke_edge
            ke_cell/= mesh.cell.area

    ttoc = time.time()
    tcpu.computeKE = tcpu.computeKE + (ttoc - ttic)

    return ke_dual, ke_cell, up_edge


def advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge):

    ttic = time.time()

    pv_flux = (
        trsk.edge_flux_perp * (pv_edge * uh_edge) +
        pv_edge * (trsk.edge_flux_perp * uh_edge)
    )

    ttoc = time.time()
    tcpu.advect_PV = tcpu.advect_PV + (ttoc - ttic)

    return pv_flux * -0.50


def computeDU(mesh, trsk, cnfg, uu_edge):

    ttic = time.time()

    du_cell = trsk.cell_flux_sums * uu_edge
    du_cell/= mesh.cell.area
    du_cell*= cnfg.du_damp_2

    du_edge = trsk.edge_grad_norm * du_cell
    
    ttoc = time.time()
    tcpu.computeDU = tcpu.computeDU + (ttoc - ttic)

    return du_edge * +1.00
    

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
        default="RK22-FB",
        required=False, 
        help="Time integration = {RK22-FB}, RK32-FB.")

    parser.add_argument(
        "--up-scheme", dest="up_scheme", type=str,
        default="AUST",
        required=False, 
        help="Up-stream formulation = {AUST}, APVM, LUST.")

    parser.add_argument(
        "--pv-upwind", dest="pv_upwind", type=float,
        default=1./80.,
        required=False,
        help="Upwind PV.-flux bias {BIAS = +1./80.}.")
    
    parser.add_argument(
        "--ke-upwind", dest="ke_upwind", type=float,
        default=1./80.,
        required=False,
        help="Upwind KE.-edge bias {BIAS = +1./80.}.")

    parser.add_argument(
        "--pv-scheme", dest="pv_scheme", type=str,
        default="UPWIND",
        required=False, 
        help="PV.-flux formulation = {UPWIND}, CENTRE.")

    parser.add_argument(
        "--ke-scheme", dest="ke_scheme", type=str,
        default="CENTRE",
        required=False, 
        help="KE.-grad formulation = {CENTRE}, WEIGHT.")

    parser.add_argument(
        "--du-damp-2", dest="du_damp_2", type=float,
        default=1.E+00,
        required=False,
        help="DIV.(U) DEL^2 coeff. {DAMP = +1.E+04}.")

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

    swe(parser.parse_args())
