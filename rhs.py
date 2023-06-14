
import numpy as np

""" SWE rhs. evaluations for various Runge-Kutta methods 
"""
#-- Darren Engwirda

from _dx import tcpu, \
                compute_H, computePV, \
                computeKE, advect_PV, \
                computeVV, \
                computeDU, computeVU, \
                computeMS

def rhs_fst_h(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

    hh_dual, \
    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    uh_edge = uu_edge * hh_edge

    uh_cell = trsk.cell_flux_sums * uh_edge
    uh_cell/= mesh.cell.area
    
    if (cnfg.no_h_tend): uh_cell *= 0.

    uh_cell[flow.is_mask] = 0.

    return uh_cell


def rhs_all_h(mesh, trsk, flow, cnfg, hh_cell, uu_edge):
    return \
    rhs_fst_h(mesh, trsk, flow, cnfg, hh_cell, uu_edge)


def rhs_slw_u(mesh, trsk, flow, cnfg, 
              hh_cell, uu_edge,
              ht_cell, ut_edge):
    
    ff_cell = flow.ff_cell
    ff_edge = flow.ff_edge
    ff_dual = flow.ff_vert

    zb_cell = flow.zb_cell

    dw_edge = 0.0

    vv_edge = computeVV(mesh, trsk, cnfg, uu_edge)

    hh_dual, \
    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    uh_edge = uu_edge * hh_edge

    ke_dual, ke_cell, ke_bias = computeKE(
        mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, 
        uu_edge, vv_edge,
        +1.0 / 2.0 * cnfg.time_step)

    ke_grad = trsk.edge_grad_norm * ke_cell
    if(cnfg.no_advect): ke_grad *= 0.

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, 
        uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, uu_edge)
    
    uu_damp+= computeVU(mesh, trsk, cnfg, uu_edge)

    if (cnfg.equations in "MADSEN-SORENSEN"):
        dw_edge = computeMS(
            mesh, trsk, cnfg, 
                hh_cell, uu_edge, ht_cell, ut_edge)
                
    ru_edge = (
        ke_grad + qh_flux - uu_damp - dw_edge
    )
                
    if (cnfg.no_u_tend): ru_edge *= 0.
    
    ru_edge[flow.uu_mask] = 0.
    
    return ru_edge, \
        ke_cell, ke_dual, ke_bias, \
        rv_cell, pv_cell, \
        rv_dual, pv_dual, pv_bias


def rhs_fst_u(mesh, trsk, flow, cnfg, 
              hh_cell, uu_edge,
              ht_cell, ut_edge):

    zb_cell = flow.zb_cell

    hz_cell = (hh_cell + zb_cell) * flow.grav
    
    hz_grad = trsk.edge_grad_norm * hz_cell
    
    if (cnfg.no_u_tend): hz_grad *= 0.
    
    hz_grad[flow.uu_mask] = 0.
    
    return hz_grad


def rhs_all_u(mesh, trsk, flow, cnfg, 
              hh_cell, uu_edge,
              ht_cell, ut_edge):

    fu_edge = rhs_fst_u(
        mesh, trsk, flow, cnfg, 
            hh_cell, uu_edge, ht_cell, ut_edge)

    su_edge, \
    ke_cell, ke_dual, ke_bias, \
    rv_cell, pv_cell, \
    rv_dual, pv_dual, pv_bias = rhs_slw_u(
        mesh, trsk, flow, cnfg, 
            hh_cell, uu_edge, ht_cell, ut_edge)

    return fu_edge + su_edge, \
        ke_cell, ke_dual, ke_bias, \
        rv_cell, pv_cell, \
        rv_dual, pv_dual, pv_bias


