
import numpy as np

""" SWE rhs. evaluations for various Runge-Kutta methods 
"""
#-- Darren Engwirda

from _dx import invariant, upwinding, tcpu, \
                compute_H, computePV, \
                computeKE, advect_PV, \
                computeDU, computeVU

def rhs_all_h(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

    hh_dual, \
    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    uh_edge = uu_edge * hh_edge

    uh_cell = trsk.cell_flux_sums * uh_edge
    uh_cell/= mesh.cell.area

    return uh_cell


def rhs_fst_h(mesh, trsk, flow, cnfg, hh_cell, uu_edge):
    return \
    rhs_all_h(mesh, trsk, flow, cnfg, hh_cell, uu_edge)


def rhs_all_u(mesh, trsk, flow, cnfg, hb_cell, uu_edge):

    ff_cell = flow.ff_cell
    ff_edge = flow.ff_edge
    ff_dual = flow.ff_vert

    ff_cell*= (cnfg.no_rotate == False)
    ff_edge*= (cnfg.no_rotate == False)
    ff_dual*= (cnfg.no_rotate == False)

    zb_cell = flow.zb_cell

    vv_edge = trsk.edge_lsqr_perp * uu_edge

    hb_dual, \
    hb_edge = compute_H(mesh, trsk, cnfg, hb_cell, uu_edge)

    uh_edge = uu_edge * hb_edge

    ke_dual, ke_cell, ke_bias = computeKE(
        mesh, trsk, cnfg, 
        hb_cell, hb_edge, hb_dual, 
        uu_edge, vv_edge,
        +1.0 / 2.0 * cnfg.time_step)

    hk_cell = hb_cell + zb_cell 
    hk_cell = (cnfg.no_advect == False) * ke_cell \
            + hk_cell * flow.grav

    hk_grad = trsk.edge_grad_norm * hk_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, trsk, cnfg, 
        hb_cell, hb_edge, hb_dual, uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, uu_edge)

    return hk_grad + qh_flux - uu_damp, \
        ke_cell, ke_dual, ke_bias, \
        rv_cell, pv_cell, \
        rv_dual, pv_dual, pv_bias


def rhs_slw_u(mesh, trsk, flow, cnfg, hb_cell, uu_edge):
    
    ff_cell = flow.ff_cell
    ff_edge = flow.ff_edge
    ff_dual = flow.ff_vert

    ff_cell*= (cnfg.no_rotate == False)
    ff_edge*= (cnfg.no_rotate == False)
    ff_dual*= (cnfg.no_rotate == False)

    zb_cell = flow.zb_cell

    vv_edge = trsk.edge_lsqr_perp * uu_edge

    hh_dual, \
    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    uh_edge = uu_edge * hh_edge

    ke_dual, ke_cell, ke_bias = computeKE(
        mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, 
        uu_edge, vv_edge,
        +1.0 / 2.0 * cnfg.time_step)

    ke_grad = trsk.edge_grad_norm * ke_cell
    ke_grad*= (cnfg.no_advect == False)

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, 
        uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, uu_edge)

    return uu_damp - qh_flux - ke_grad


