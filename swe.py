
import os
import time
import numpy as np
import xarray
import argparse

from msh import load_mesh, load_flow
from ops import trsk_mats


class base: pass
tcpu = base()
tcpu.thickness = 0.0E+00
tcpu.momentum_ = 0.0E+00
tcpu.computeKE = 0.0E+00
tcpu.computePV = 0.0E+00
tcpu.advect_PV = 0.0E+00
tcpu.upwind_PV = 0.0E+00

def main(cnfg):
    """
    SWE: solve the nonlinear SWE on generalised MPAS meshes.

    """
    # Authors: Darren Engwirda

    cnfg.stat_freq = np.minimum(
        cnfg.save_freq, cnfg.stat_freq)
    
    cnfg.integrate = cnfg.integrate.upper()
    cnfg.operators = cnfg.operators.upper()
    cnfg.pv_scheme = cnfg.pv_scheme.upper()

    name = cnfg.mpas_file
    path, file = os.path.split(name)
    save = os.path.join(path, "out_" + file)

    mesh = load_mesh(name)
    flow = load_flow(name)

    trsk = trsk_mats(mesh)

    u0_edge = flow.uu_edge[0, :, 0]
    uu_edge = u0_edge
    h0_cell = flow.hh_cell[0, :, 0]
    hh_cell = h0_cell

    kp_sums = np.zeros((
        cnfg.iteration // cnfg.stat_freq + 1), dtype=np.float64)
    pv_sums = np.zeros((
        cnfg.iteration // cnfg.stat_freq + 1), dtype=np.float64)

    ttic = time.time(); xout = []; next = 0

    for step in range(cnfg.iteration + 1):

        if ("RK22" in cnfg.integrate):

            hh_cell, uu_edge, ke_cell, \
            rv_cell, pv_cell, \
            rv_dual, pv_dual = step_RK22(
                mesh, trsk, flow, cnfg, hh_cell, uu_edge)

        if ("RK32" in cnfg.integrate):

            hh_cell, uu_edge, ke_cell, \
            rv_cell, pv_cell, \
            rv_dual, pv_dual = step_RK32(
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

            xnow = xarray.Dataset()
            xnow["uu_edge"] = (("nEdges", "nVertLevels"),
                np.reshape(uu_edge, (mesh.edge.size, 1)))
            xnow["hh_cell"] = (("nCells", "nVertLevels"),
                np.reshape(hh_cell, (mesh.cell.size, 1)))
            xnow["pv_dual"] = (("nVertices", "nVertLevels"),
                np.reshape(pv_dual, (mesh.vert.size, 1)))
            xnow["pv_cell"] = (("nCells", "nVertLevels"),
                np.reshape(pv_cell, (mesh.cell.size, 1)))
            xnow["rv_dual"] = (("nVertices", "nVertLevels"),
                np.reshape(rv_dual, (mesh.vert.size, 1)))
            xnow["rv_cell"] = (("nCells", "nVertLevels"),
                np.reshape(rv_cell, (mesh.cell.size, 1)))
            xnow["ke_cell"] = (("nCells", "nVertLevels"),
                np.reshape(ke_cell, (mesh.cell.size, 1)))

            ux_cell = trsk.cell_lsqr_xnrm * uu_edge
            uy_cell = trsk.cell_lsqr_ynrm * uu_edge
            uz_cell = trsk.cell_lsqr_znrm * uu_edge

            ux_dual = trsk.dual_kite_sums * ux_cell
            ux_dual = ux_dual / mesh.vert.area
            uy_dual = trsk.dual_kite_sums * uy_cell
            uy_dual = uy_dual / mesh.vert.area
            uz_dual = trsk.dual_kite_sums * uz_cell
            uz_dual = uz_dual / mesh.vert.area

            xnow["ux_dual"] = (("nVertices", "nVertLevels"),
                np.reshape(ux_dual, (mesh.vert.size, 1)))
            xnow["uy_dual"] = (("nVertices", "nVertLevels"),
                np.reshape(uy_dual, (mesh.vert.size, 1)))
            xnow["uz_dual"] = (("nVertices", "nVertLevels"),
                np.reshape(uz_dual, (mesh.vert.size, 1)))

            dh_cell = hh_cell - h0_cell

            xnow["dh_cell"] = (("nCells", "nVertLevels"),
                np.reshape(dh_cell, (mesh.cell.size, 1)))

            xout.append(xnow)

    ttoc = time.time()

    print("TCPU:", ttoc - ttic)
    print("tcpu.thickness:", tcpu.thickness)
    print("tcpu.momentum_:", tcpu.momentum_)
    print("tcpu.computeKE:", tcpu.computeKE)    
    print("tcpu.computePV:", tcpu.computePV)
    print("tcpu.upwind_PV:", tcpu.upwind_PV)    
    print("tcpu.advect_PV:", tcpu.advect_PV)

    data = xarray.Dataset()
    data = xarray.merge(
        (data, xarray.concat(xout, dim="Time")))
    data.attrs["on_a_sphere"] = "YES"
    data.attrs["sphere_radius"] = mesh.rsph
    data.attrs["is_periodic"] = "NO"
    data.attrs["source"] = "swe-python"

    data["kk_sums"] = (("nSteps"), kp_sums)
    data["pv_sums"] = (("nSteps"), pv_sums)

    data["lonCell"] = (("nCells"), mesh.cell.xlon)
    data["latCell"] = (("nCells"), mesh.cell.ylat)
    data["xCell"] = (("nCells"), mesh.cell.xpos)
    data["yCell"] = (("nCells"), mesh.cell.ypos)
    data["zCell"] = (("nCells"), mesh.cell.zpos)
    data["areaCell"] = (("nCells"), mesh.cell.area)
    data["verticesOnCell"] = (
        ("nCells", "maxEdges"), mesh.cell.vert)
    data["edgesOnCell"] = (
        ("nCells", "maxEdges"), mesh.cell.edge)
    data["cellsOnCell"] = (
        ("nCells", "maxEdges"), mesh.cell.cell)
    data["nEdgesOnCell"] = (("nCells"), mesh.cell.topo)

    data["lonEdge"] = (("nEdges"), mesh.edge.xlon)
    data["latEdge"] = (("nEdges"), mesh.edge.ylat)
    data["xEdge"] = (("nEdges"), mesh.edge.xpos)
    data["yEdge"] = (("nEdges"), mesh.edge.ypos)
    data["zEdge"] = (("nEdges"), mesh.edge.zpos)
    data["dvEdge"] = (("nEdges"), mesh.edge.vlen)
    data["dcEdge"] = (("nEdges"), mesh.edge.clen)
    data["verticesOnEdge"] = (
        ("nEdges", "TWO"), mesh.edge.vert)
    data["weightsOnEdge"] = (
        ("nEdges", "maxEdges2"), mesh.edge.wmul)
    data["cellsOnEdge"] = (
        ("nEdges", "TWO"), mesh.edge.cell)
    data["edgesOnEdge"] = (
        ("nEdges", "maxEdges2"), mesh.edge.edge)
    data["nEdgesOnEdge"] = (("nEdges"), mesh.edge.topo)

    data["lonVertex"] = (("nVertices"), mesh.vert.xlon)
    data["latVertex"] = (("nVertices"), mesh.vert.ylat)
    data["xVertex"] = (("nVertices"), mesh.vert.xpos)
    data["yVertex"] = (("nVertices"), mesh.vert.ypos)
    data["zVertex"] = (("nVertices"), mesh.vert.zpos)
    data["areaTriangle"] = (
        ("nVertices"), mesh.vert.area)
    data["kiteAreasOnVertex"] = (
        ("nVertices", "vertexDegree"), mesh.vert.kite)
    data["edgesOnVertex"] = (
        ("nVertices", "vertexDegree"), mesh.vert.edge)
    data["cellsOnVertex"] = (
        ("nVertices", "vertexDegree"), mesh.vert.cell)

    data.to_netcdf(
        save, format="NETCDF3_64BIT_OFFSET")

    return


def invariant(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

    ff_dual = flow.ff_vert
    ff_cell = flow.ff_cell
    zb_cell = flow.zb_cell

    vv_edge = trsk.edge_lsqr_perp * uu_edge * -1.

    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

   #ke_cell, \
   #ke_edge = computeKE(mesh, trsk, cnfg, uu_edge, vv_edge)

    ke_edge =  1.0 * uu_edge ** 2

    ke_edge = ke_edge * hh_edge * mesh.edge.area

    pe_cell = flow.grav * (hh_cell * .5 + zb_cell)

    pe_cell = pe_cell * hh_cell * mesh.cell.area

    kk_sums = np.sum(ke_edge) + np.sum(pe_cell)

    rv_dual, pv_dual, \
    rv_cell, pv_cell, pv_edge = computePV(
        mesh, trsk, cnfg, hh_cell, uu_edge, vv_edge, 
        ff_dual, ff_cell, 
        +0.0 / 1.0 * cnfg.time_step, cnfg.apvm_beta)

    pv_sums = np.sum(hh_cell * pv_cell ** 2)
   
    return kk_sums, pv_sums


def step_RK22(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

#-- A 2-stage RK2 + FB scheme, a'la ROMS:
#-- A.F. Shchepetk, J.C. McWilliams (2005): The regional oceanic 
#-- modeling system (ROMS): a split-explicit, free-surface, 
#-- topography-following-coordinate oceanic model
#-- doi.org/10.1016/j.ocemod.2004.08.002

    ff_cell = flow.ff_cell
    ff_dual = flow.ff_vert
    zb_cell = flow.zb_cell

#-- 1st RK + FB stage

    ttic = time.time()

    BETA = (1.0 / 3.0) * ("FB" in cnfg.integrate)

    vv_edge = trsk.edge_lsqr_perp * uu_edge * -1.

    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    uh_edge = uu_edge * hh_edge

    dd_cell = trsk.cell_flux_sums * uh_edge
    dd_cell = dd_cell / mesh.cell.area

    h1_cell = (
        hh_cell - 1.0 / 1.0 * cnfg.time_step * dd_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    hb_cell = h1_cell * (0.0 + 1.0 * BETA) + \
              hh_cell * (1.0 - 1.0 * BETA)

    hb_edge = compute_H(mesh, trsk, cnfg, hb_cell, uu_edge)

    uh_edge = uu_edge * hb_edge

    ke_cell = computeKE(mesh, trsk, cnfg, uu_edge, vv_edge)

    hk_cell = hb_cell + zb_cell 
    hk_cell = ke_cell + hk_cell * flow.grav

    hk_grad = trsk.edge_grad_norm * hk_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge = computePV(
        mesh, trsk, cnfg, hb_cell, uu_edge, vv_edge, 
        ff_dual, ff_cell, 
        +0.0 / 1.0 * cnfg.time_step, cnfg.apvm_beta)

    qh_edge = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    u1_edge = uu_edge - 1.0 / 1.0 * cnfg.time_step * (
        hk_grad + qh_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 2nd RK + FB stage

    ttic = time.time()

    BETA = (1.0 / 6.0) * ("FB" in cnfg.integrate)

    hm_cell = 0.5 * hh_cell + 0.5 * h1_cell
    um_edge = 0.5 * uu_edge + 0.5 * u1_edge

    vm_edge = trsk.edge_lsqr_perp * um_edge * -1.

    hm_edge = compute_H(mesh, trsk, cnfg, hm_cell, um_edge)

    uh_edge = um_edge * hm_edge

    dd_cell = trsk.cell_flux_sums * uh_edge
    dd_cell = dd_cell / mesh.cell.area

    h2_cell = (
        hh_cell - 1.0 / 1.0 * cnfg.time_step * dd_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    hb_cell = h2_cell * (0.5 - 1.0 * BETA) + \
              h1_cell * (0.0 + 1.0 * BETA) + \
              hh_cell * (0.5)

    hb_edge = compute_H(mesh, trsk, cnfg, hb_cell, um_edge)

    uh_edge = um_edge * hb_edge

    ke_cell = computeKE(mesh, trsk, cnfg, um_edge, vm_edge)

    hk_cell = hb_cell + zb_cell 
    hk_cell = ke_cell + hk_cell * flow.grav

    hk_grad = trsk.edge_grad_norm * hk_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge = computePV(
        mesh, trsk, cnfg, hb_cell, um_edge, vm_edge, 
        ff_dual, ff_cell, 
        +1.0 / 1.0 * cnfg.time_step, cnfg.apvm_beta)

    qh_edge = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    u2_edge = uu_edge - 1.0 / 1.0 * cnfg.time_step * (
        hk_grad + qh_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

    return h2_cell, u2_edge, ke_cell, \
           rv_cell, pv_cell, rv_dual, pv_dual


def step_RK32(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

#-- A 3-stage RK2 + FB scheme, a'la MPAS-A:
#-- L.J. Wicker, W.C. Skamarock (2002): Time-Splitting Methods for 
#-- Elastic Models Using Forward Time Schemes
#-- doi.org/10.1175/1520-0493(2002)130<2088:TSMFEM>2.0.CO;2

#-- but with FB weighting applied within each RK stage

    ff_cell = flow.ff_cell
    ff_dual = flow.ff_vert
    zb_cell = flow.zb_cell

#-- 1st RK + FB stage

    ttic = time.time()

    BETA = (1.0 / 3.0) * ("FB" in cnfg.integrate)

    vv_edge = trsk.edge_lsqr_perp * uu_edge * -1.

    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    uh_edge = uu_edge * hh_edge

    dd_cell = trsk.cell_flux_sums * uh_edge
    dd_cell = dd_cell / mesh.cell.area

    h1_cell = (
        hh_cell - 1.0 / 3.0 * cnfg.time_step * dd_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    hb_cell = h1_cell * (0.0 + 1.0 * BETA) + \
              hh_cell * (1.0 - 1.0 * BETA)

    hb_edge = compute_H(mesh, trsk, cnfg, hb_cell, uu_edge)

    uh_edge = uu_edge * hb_edge

    ke_cell = computeKE(mesh, trsk, cnfg, uu_edge, vv_edge)

    hk_cell = hb_cell + zb_cell 
    hk_cell = ke_cell + hk_cell * flow.grav

    hk_grad = trsk.edge_grad_norm * hk_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge = computePV(
        mesh, trsk, cnfg, hb_cell, uu_edge, vv_edge, 
        ff_dual, ff_cell, 
        +0.0 / 1.0 * cnfg.time_step, cnfg.apvm_beta)

    qh_edge = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    u1_edge = uu_edge - 1.0 / 3.0 * cnfg.time_step * (
        hk_grad + qh_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 2nd RK + FB stage

    ttic = time.time()

    BETA = (1.0 / 5.0) * ("FB" in cnfg.integrate)

    v1_edge = trsk.edge_lsqr_perp * u1_edge * -1.

    h1_edge = compute_H(mesh, trsk, cnfg, h1_cell, u1_edge)

    uh_edge = u1_edge * h1_edge

    dd_cell = trsk.cell_flux_sums * uh_edge
    dd_cell = dd_cell / mesh.cell.area

    h2_cell = (
        hh_cell - 1.0 / 2.0 * cnfg.time_step * dd_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    hb_cell = h2_cell * (0.0 + 1.0 * BETA) + \
              h1_cell * (1.0 - 1.0 * BETA)

    hb_edge = compute_H(mesh, trsk, cnfg, hb_cell, u1_edge)

    uh_edge = u1_edge * hb_edge

    ke_cell = computeKE(mesh, trsk, cnfg, u1_edge, v1_edge)

    hk_cell = hb_cell + zb_cell 
    hk_cell = ke_cell + hk_cell * flow.grav

    hk_grad = trsk.edge_grad_norm * hk_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge = computePV(
        mesh, trsk, cnfg, hb_cell, u1_edge, v1_edge, 
        ff_dual, ff_cell, 
        +1.0 / 3.0 * cnfg.time_step, cnfg.apvm_beta)

    qh_edge = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    u2_edge = uu_edge - 1.0 / 2.0 * cnfg.time_step * (
        hk_grad + qh_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 3rd RK + FB stage

    ttic = time.time()

    BETA = (1.0 / 7.0) * ("FB" in cnfg.integrate)
    
    v2_edge = trsk.edge_lsqr_perp * u2_edge * -1.

    h2_edge = compute_H(mesh, trsk, cnfg, h2_cell, u2_edge)

    uh_edge = u2_edge * h2_edge

    dd_cell = trsk.cell_flux_sums * uh_edge
    dd_cell = dd_cell / mesh.cell.area

    h3_cell = (
        hh_cell - 1.0 / 1.0 * cnfg.time_step * dd_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    hb_cell = h3_cell * (0.0 + 1.0 * BETA) + \
              h2_cell * (1.0 - 2.0 * BETA) + \
              hh_cell * (0.0 + 1.0 * BETA)

    hb_edge = compute_H(mesh, trsk, cnfg, hb_cell, u2_edge)

    uh_edge = u2_edge * hb_edge

    ke_cell = computeKE(mesh, trsk, cnfg, u2_edge, v2_edge)

    hk_cell = hb_cell + zb_cell 
    hk_cell = ke_cell + hk_cell * flow.grav

    hk_grad = trsk.edge_grad_norm * hk_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge = computePV(
        mesh, trsk, cnfg, hb_cell, u2_edge, v2_edge, 
        ff_dual, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step, cnfg.apvm_beta)

    qh_edge = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    u3_edge = uu_edge - 1.0 / 1.0 * cnfg.time_step * (
        hk_grad + qh_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

    return h3_cell, u3_edge, ke_cell, \
           rv_cell, pv_cell, rv_dual, pv_dual


def compute_H(mesh, trsk, cnfg, hh_cell, uu_edge):

    if (cnfg.operators == "TRSK-CV"):

        hh_edge = trsk.edge_wing_sums * hh_cell
        hh_edge = hh_edge / mesh.edge.area

    if (cnfg.operators == "TRSK-MD"):

        hh_edge = trsk.edge_cell_sums * hh_cell
        hh_edge = hh_edge * 0.5E+00

    return hh_edge


def computePV(mesh, trsk, cnfg, hh_cell, uu_edge, vv_edge, 
              ff_dual, ff_cell,
              delta_t, pv_damp):

    ttic = time.time()

    if (cnfg.operators == "TRSK-CV"):

    #-- RV+f on dual, & PV on edge - more compact hh_edge?
        hh_dual = trsk.dual_kite_sums * hh_cell
        hh_dual = hh_dual / mesh.vert.area

        rv_dual = trsk.dual_curl_sums * uu_edge
        rv_dual = rv_dual / mesh.vert.area
        
        av_dual = rv_dual + ff_dual
        pv_dual = av_dual / hh_dual
        
        pv_cell = trsk.cell_kite_sums * pv_dual
        pv_cell = pv_cell / mesh.cell.area

        av_cell = pv_cell * hh_cell
        rv_cell = av_cell - ff_cell

        av_edge = trsk.edge_stub_sums * av_dual
        av_edge = av_edge / mesh.edge.area

        hh_edge = compute_H(
            mesh, trsk, cnfg, hh_cell, uu_edge)

        av_edge = upwind_SV(
            mesh, trsk, cnfg, av_dual, av_cell, av_edge,
            uu_edge, vv_edge, delta_t, pv_damp)

        pv_edge = av_edge / hh_edge

    if (cnfg.operators == "TRSK-MD"):

        hh_dual = trsk.dual_kite_sums * hh_cell
        hh_dual = hh_dual / mesh.vert.area

        rv_dual = trsk.dual_curl_sums * uu_edge
        rv_dual = rv_dual / mesh.vert.area

        av_dual = rv_dual + ff_dual
        pv_dual = av_dual / hh_dual

        pv_cell = trsk.cell_kite_sums * pv_dual
        pv_cell = pv_cell / mesh.cell.area

        av_cell = pv_cell * hh_cell
        rv_cell = av_cell - ff_cell

        pv_edge = trsk.edge_vert_sums * pv_dual
        pv_edge = pv_edge * 0.5E+00

        pv_edge = upwind_SV(
            mesh, trsk, cnfg, pv_dual, pv_cell, pv_edge, 
            uu_edge, vv_edge, delta_t, pv_damp)

    ttoc = time.time()
    tcpu.computePV = tcpu.computePV + (ttoc - ttic)

    return rv_dual, pv_dual, rv_cell, pv_cell, \
           pv_edge


def upwind_SV(mesh, trsk, cnfg, sv_dual, sv_cell, sv_edge,
              uu_edge, vv_edge, 
              delta_t, sv_damp):

    ttic = time.time()

    if (cnfg.pv_scheme == "LUST"):

    #-- upwind bias, a'la Weller
    #-- H. Weller (2012): Controlling the computational modes 
    #-- of the arbitrarily structured C grid
    #-- https://doi.org/10.1175/MWR-D-11-00221.1
        gv_edge = trsk.edge_grad_perp * sv_dual

        gx_dual = trsk.dual_lsqr_xprp * gv_edge
        gy_dual = trsk.dual_lsqr_yprp * gv_edge
        gz_dual = trsk.dual_lsqr_zprp * gv_edge

        up_edge = vv_edge >= 0.
        dn_edge = vv_edge <= 0.

        up_dual = np.zeros(
            mesh.edge.size, dtype=np.int32)
        up_dual[up_edge] = \
            mesh.edge.vert[up_edge, 1] - 1
        up_dual[dn_edge] = \
            mesh.edge.vert[dn_edge, 0] - 1

        up_xdel = \
            mesh.edge.xmid - mesh.vert.xmid[up_dual]
        up_ydel = \
            mesh.edge.ymid - mesh.vert.ymid[up_dual]
        up_zdel = \
            mesh.edge.zmid - mesh.vert.zmid[up_dual]

        ZERO = 1.E-12 * np.max(uu_edge)

        va_edge = np.abs(vv_edge) + ZERO
        ua_edge = np.abs(uu_edge) + ZERO
        
        BIAS = (
            +1. / 6. * va_edge / (ua_edge + va_edge)
        )

        sv_wind = sv_dual[up_dual] + \
            up_xdel * gx_dual[up_dual] + \
            up_ydel * gy_dual[up_dual] + \
            up_zdel * gz_dual[up_dual]

        sv_edge = \
            BIAS * sv_wind + (1.0 - BIAS) * sv_edge

    if (cnfg.pv_scheme == "APVM"):

    #-- upwind APVM, a'la Ringler
        gn_edge = trsk.apvm_grad_norm * sv_dual * -1.
        gp_edge = trsk.apvm_grad_perp * sv_dual * -1.

        sv_apvm = sv_damp * uu_edge * gn_edge + \
                  sv_damp * vv_edge * gp_edge

        sv_edge = sv_edge - delta_t * sv_apvm
        
    ttoc = time.time()
    tcpu.upwind_PV = tcpu.upwind_PV + (ttoc - ttic)

    return sv_edge


def computeKE(mesh, trsk, cnfg, uu_edge, vv_edge):

    ttic = time.time()

    if (cnfg.operators == "TRSK-CV"):

        ke_edge = 0.5 * (uu_edge ** 2 + vv_edge ** 2)

        ke_cell = trsk.cell_wing_sums * ke_edge
        ke_cell = ke_cell / mesh.cell.area
        
        ux_cell = trsk.cell_lsqr_xnrm * uu_edge
        uy_cell = trsk.cell_lsqr_ynrm * uu_edge
        uz_cell = trsk.cell_lsqr_znrm * uu_edge
        
        k2_cell = 0.5 * (ux_cell ** 2 + 
                         uy_cell ** 2 +
                         uz_cell ** 2)
        
        ke_cell = 0.5 * ke_cell + 0.5 * k2_cell
        
    if (cnfg.operators == "TRSK-MD"):

        ke_edge = 0.25 * uu_edge ** 2 * \
                  mesh.edge.clen * \
                  mesh.edge.vlen
        ke_cell = trsk.cell_edge_sums * ke_edge
        ke_cell = ke_cell / mesh.cell.area
        
    ttoc = time.time()
    tcpu.computeKE = tcpu.computeKE + (ttoc - ttic)

    return ke_cell


def advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge):

    ttic = time.time()

    pv_flux = (
        trsk.edge_flux_perp * (pv_edge * uh_edge) +
        pv_edge * (trsk.edge_flux_perp * uh_edge)
    )

    ttoc = time.time()
    tcpu.advect_PV = tcpu.advect_PV + (ttoc - ttic)

    return pv_flux * -0.50
    

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
        "--pv-scheme", dest="pv_scheme", type=str,
        default="APVM",
        required=False, 
        help="PV.-flux formulation = {APVM}, LUST.")

    parser.add_argument(
        "--apvm-beta", dest="apvm_beta", type=float,
        default=.5E+00,
        required=False, 
        help="APVM multipling coeff. {BETA = 0.5}.")

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

    main(parser.parse_args())
