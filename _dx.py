
import time
import numpy as np

""" SWE spatial discretisation using TRSK-like operators
"""
#-- Darren Engwirda

HH_TINY        = 1.0E-02
HH_THIN        = 1.0E+02
UU_TINY        = 1.0E-12
PV_TINY        = 1.0E-13
KE_TINY        = 1.0E+03

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
tcpu.computeVU = 0.0E+00
tcpu.computeDW = 0.0E+00
tcpu.computeCd = 0.0E+00

def hrmn_mean(xone, xtwo):

#-- harmonic mean of two vectors (ie. biased toward lesser)

    return +2.0 * xone * xtwo / (xone + xtwo)


def diag_vars(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

#-- compute diagnostic variables from the current state

    ff_dual = flow.ff_vert
    ff_edge = flow.ff_edge
    ff_cell = flow.ff_cell

    ff_cell*= (cnfg.no_rotate == False)
    ff_edge*= (cnfg.no_rotate == False)
    ff_dual*= (cnfg.no_rotate == False)

    zb_cell = flow.zb_cell

    vv_edge = trsk.edge_lsqr_perp * uu_edge * +1.

    hh_dual, \
    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    ke_dual, ke_cell, ke_bias = computeKE(
        mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, 
        uu_edge, vv_edge,
        +0.0 * cnfg.time_step)

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, uu_edge, vv_edge,
        ff_dual, ff_edge, ff_cell, 
        +0.0 * cnfg.time_step)

    return ke_cell, ke_dual, \
           rv_cell, pv_cell, \
           rv_dual, pv_dual, ke_bias, pv_bias


def invariant(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

#-- compute the discrete energy and enstrophy invariants

    ff_dual = flow.ff_vert
    ff_edge = flow.ff_edge
    ff_cell = flow.ff_cell

    ff_cell*= (cnfg.no_rotate == False)
    ff_edge*= (cnfg.no_rotate == False)
    ff_dual*= (cnfg.no_rotate == False)

    zb_cell = flow.zb_cell

    vv_edge = trsk.edge_lsqr_perp * uu_edge * +1.

    hh_dual, \
    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    ke_edge =  0.5 * uu_edge ** 2

    ke_edge = ke_edge * hh_edge * mesh.edge.clen \
                                * mesh.edge.vlen

    pe_cell = flow.grav * (hh_cell * 0.5 - zb_cell)

    pe_cell = pe_cell * hh_cell * mesh.cell.area

    kk_sums = np.sum(ke_edge) + np.sum(pe_cell)

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, __ = computePV(
        mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, uu_edge, vv_edge,
        ff_dual, ff_edge, ff_cell, 
        +0.0 * cnfg.time_step)

   #pv_sums = np.sum(
   #    +0.5 * mesh.edge.area * hh_edge * pv_edge ** 2)

    pv_sums = np.sum(
        +0.5 * mesh.vert.area * hh_dual * pv_dual ** 2)

    return kk_sums, pv_sums


def upwinding(mesh, trsk, cnfg, 
              sm_dual, sv_dual, sv_cell, sv_edge,
              uu_edge, vv_edge, 
              delta_t, sv_tiny,
              up_kind, up_min_, up_max_):

    ttic = time.time()

    sv_bias = np.zeros(mesh.edge.size, dtype=float)

    if (up_kind == "LUST"):

    #-- upwind bias, a'la Weller
    #-- H. Weller (2012): Controlling the computational modes 
    #-- of the arbitrarily structured C grid
    #-- https://doi.org/10.1175/MWR-D-11-00221.1
        dv_edge = trsk.edge_grad_perp * sv_dual

        dx_dual = trsk.dual_lsqr_xprp * dv_edge
        dy_dual = trsk.dual_lsqr_yprp * dv_edge
        dz_dual = trsk.dual_lsqr_zprp * dv_edge

        sv_bias = np.full(
            mesh.edge.size, up_min_, dtype=float)

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
            up_min_ * va_edge / (ua_edge + va_edge)
        )

        sv_wind = sv_dual[up_dual] + \
            up_xdel * dx_dual[up_dual] + \
            up_ydel * dy_dual[up_dual] + \
            up_zdel * dz_dual[up_dual]

        sv_edge = \
            BIAS * sv_wind + (1.0 - BIAS) * sv_edge

    if (up_kind == "APVM"):

        dn_edge = trsk.edge_grad_norm * sv_cell
        dp_edge = trsk.edge_grad_perp * sv_dual

        sv_bias = np.full(
            mesh.edge.size, (+0.50), dtype=float)

    #-- upwind APVM, scale w time
        up_bias = 1.0                   # hard-coded?
        sv_apvm = up_bias * uu_edge * dn_edge + \
                  up_bias * vv_edge * dp_edge

        sv_edge = sv_edge - delta_t * sv_apvm
        
    if (up_kind == "AUST-CONST"):

    #-- AUST: anticipated upstream method; APVM meets
    #-- LUST? Upwinds in multi-dimensional sense, vs.
    #-- LUST, which upwinds via tangential dir. only.

    #-- const. upwinding version

        dn_edge = trsk.edge_grad_norm * sv_cell
        dp_edge = trsk.edge_grad_perp * sv_dual

        sv_bias = np.full(
            mesh.edge.size, up_min_, dtype=float)

    #-- upwind APVM, scale w grid
        um_edge = UU_TINY + \
            np.sqrt(uu_edge ** 2 + vv_edge ** 2)

        sv_wind = uu_edge * dn_edge / um_edge + \
                  vv_edge * dp_edge / um_edge

        ee_scal = mesh.edge.slen

        sv_edge-= up_min_ * ee_scal * sv_wind

    if (up_kind == "AUST-ADAPT"):
        
    #-- AUST: anticipated upstream method; APVM meets
    #-- LUST? Upwinds in multi-dimensional sense, vs.
    #-- LUST, which upwinds via tangential dir. only.

    #-- adapt. upwinding version

        dn_edge = trsk.edge_grad_norm * sv_cell
        dp_edge = trsk.edge_grad_perp * sv_dual

        dm_edge = 0.5 * (
            np.abs(dn_edge * mesh.edge.clen) +
            np.abs(dp_edge * mesh.edge.vlen)
        )

        dm_vert = \
            (trsk.dual_edge_sums * dm_edge) / 3.

        ds_dual = np.abs(
            sv_dual - sm_dual) / (dm_vert + sv_tiny)

        ds_dual = ds_dual ** 2

        sv_bias = np.sqrt(
            (trsk.edge_vert_sums * ds_dual) / 2.
        )

    #-- upwind APVM, scale w grid
        um_edge = UU_TINY + \
            np.sqrt(uu_edge ** 2 + vv_edge ** 2)

        sv_wind = uu_edge * dn_edge / um_edge + \
                  vv_edge * dp_edge / um_edge

        ee_scal = mesh.edge.slen

        sv_bias = up_min_ + \
            np.minimum(up_max_ - up_min_, sv_bias)

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
              delta_t):

    ttic = time.time()

    if (cnfg.operators == "TRSK-CV"):

    #-- RV+f on rhombi, PV on edge - more compact stencil?        
        rv_dual = trsk.dual_curl_sums * uu_edge
        rv_dual/= mesh.vert.area
        
        rv_dual*= (cnfg.no_advect == False)

        av_dual = rv_dual + ff_dual
        pv_dual = av_dual / hh_dual
        
        rv_cell = trsk.cell_kite_sums * rv_dual
        rv_cell/= mesh.cell.area

        rv_cell*= (cnfg.no_advect == False)

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

        rm_dual*= (cnfg.no_advect == False)

        am_dual = rm_dual + ff_dual
        pm_dual = am_dual / hh_dual

    #-- compute high(er)-order RV + PV on edges:
    #-- pv_edge = pv_dual + (xe - xv) * pv_d/dx
        pv_edge = trsk.edge_vert_sums * pv_dual / 2.0
        pv_edge+= trsk.edge_dual_reco * pm_dual

        pv_edge = 1. / 3. * pv_edge \
                + 2. / 3. * pm_edge

        pv_edge, up_edge = upwinding(
            mesh, trsk, cnfg, 
            pm_dual, pv_dual, pv_cell, pv_edge,
            uu_edge, vv_edge, 
            delta_t, PV_TINY, 
            cnfg.pv_upwind, 
            cnfg.pv_min_up, cnfg.pv_max_up)

    if (cnfg.operators == "TRSK-MD"):

        rv_dual = trsk.dual_curl_sums * uu_edge
        rv_dual/= mesh.vert.area

        rv_dual*= (cnfg.no_advect == False)

        av_dual = rv_dual + ff_dual
        pv_dual = av_dual / hh_dual

        rv_cell = trsk.cell_kite_sums * rv_dual
        rv_cell/= mesh.cell.area

        rv_cell*= (cnfg.no_advect == False)

        av_cell = rv_cell + ff_cell
        pv_cell = av_cell / hh_cell

        rv_edge = trsk.edge_vert_sums * rv_dual / 2.0
        pv_edge = trsk.edge_vert_sums * pv_dual / 2.0
        
        rm_dual = trsk.dual_edge_sums * rv_edge / 3.0
        
        rm_dual*= (cnfg.no_advect == False)

        am_dual = rm_dual + ff_dual
        pm_dual = am_dual / hh_dual

        pv_edge, up_edge = upwinding(
            mesh, trsk, cnfg, 
            pm_dual, pv_dual, pv_cell, pv_edge, 
            uu_edge, vv_edge, 
            delta_t, PV_TINY, 
            cnfg.pv_upwind, 
            cnfg.pv_min_up, cnfg.pv_max_up)

    ttoc = time.time()
    tcpu.computePV = tcpu.computePV + (ttoc - ttic)

    return rv_dual, pv_dual, rv_cell, pv_cell, \
           pv_edge, up_edge


def computeKE(mesh, trsk, cnfg, 
              hh_cell, hh_edge, hh_dual, uu_edge, vv_edge,
              delta_t):

    ttic = time.time()

    if (cnfg.operators == "TRSK-CV"):

        up_edge = np.zeros(mesh.edge.size, dtype=float)

        if ("CENTRE" in cnfg.ke_scheme):

            ke_edge = 0.5 * (uu_edge ** 2 + 
                             vv_edge ** 2 )

        if ("UPWIND" in cnfg.ke_scheme):

            ke_edge = 0.5 * (uu_edge ** 2 + 
                             vv_edge ** 2 )

            ux_dual = trsk.dual_lsqr_xnrm * uu_edge
            uy_dual = trsk.dual_lsqr_ynrm * uu_edge
            uz_dual = trsk.dual_lsqr_znrm * uu_edge

            ke_dual = 0.5 * (ux_dual ** 2 + 
                             uy_dual ** 2 + 
                             uz_dual ** 2 )
            
            ke_cell = trsk.cell_kite_sums * ke_dual
            ke_cell/= mesh.cell.area

            km_dual = \
                trsk.dual_edge_sums * ke_edge / 3.0

            ke_edge, up_edge = upwinding(
                mesh, trsk, cnfg, 
                km_dual, ke_dual, ke_cell, ke_edge, 
                uu_edge, vv_edge, 
                delta_t, KE_TINY,
                cnfg.ke_upwind, 
                cnfg.ke_min_up, cnfg.ke_max_up)

        if ("SKINNY" in cnfg.ke_scheme):

            hh_scal = HH_THIN / hh_edge

            ke_edge*= ((1.0 + hh_scal) ** 2) ** 2

            ke_cell = trsk.cell_wing_sums * ke_edge
            ke_cell/= mesh.cell.area

            ke_dual = trsk.dual_stub_sums * ke_edge
            ke_dual/= mesh.vert.area

            hh_scal = HH_THIN / hh_cell

            ke_cell/= ((1.0 + hh_scal) ** 2) ** 2

        else:

            ke_cell = trsk.cell_wing_sums * ke_edge
            ke_cell/= mesh.cell.area

            ke_dual = trsk.dual_stub_sums * ke_edge
            ke_dual/= mesh.vert.area

    if (cnfg.operators == "TRSK-MD"):

        up_edge = np.zeros(mesh.edge.size, dtype=float)

        if ("CENTRE" in cnfg.ke_scheme):

            ke_edge = 0.25 * uu_edge ** 2 * \
                  mesh.edge.clen * mesh.edge.vlen

        if ("UPWIND" in cnfg.ke_scheme):

            ke_edge = 1.00 * uu_edge ** 2

            ux_dual = trsk.dual_lsqr_xnrm * uu_edge
            uy_dual = trsk.dual_lsqr_ynrm * uu_edge
            uz_dual = trsk.dual_lsqr_znrm * uu_edge

            ke_dual = 0.5 * (ux_dual ** 2 + 
                             uy_dual ** 2 + 
                             uz_dual ** 2 )
            
            ke_cell = trsk.cell_kite_sums * ke_dual
            ke_cell/= mesh.cell.area

            km_dual = \
                trsk.dual_edge_sums * ke_edge / 3.0

            ke_edge, up_edge = upwinding(
                mesh, trsk, cnfg, 
                ke_dual, ke_dual, ke_cell, ke_edge, 
                uu_edge, vv_edge, 
                delta_t, KE_TINY, 
                cnfg.ke_upwind,
                cnfg.ke_min_up, cnfg.ke_max_up) 

            ke_edge = 0.25 * ke_edge ** 1 * \
                  mesh.edge.clen * mesh.edge.vlen

        if ("SKINNY" in cnfg.ke_scheme):

            hh_scal = HH_THIN / hh_edge

            ke_edge*= ((1.0 + hh_scal) ** 2) ** 2

            ke_cell = trsk.cell_edge_sums * ke_edge
            ke_cell/= mesh.cell.area

            ke_dual = trsk.dual_edge_sums * ke_edge
            ke_dual/= mesh.vert.area

            hh_scal = HH_THIN / hh_cell

            ke_cell/= ((1.0 + hh_scal) ** 2) ** 2

        else:

            ke_cell = trsk.cell_edge_sums * ke_edge
            ke_cell/= mesh.cell.area

            ke_dual = trsk.dual_edge_sums * ke_edge
            ke_dual/= mesh.vert.area


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

    if (cnfg.du_damp_2 == 0 and 
        cnfg.du_damp_4 == 0):
        return np.zeros(mesh.edge.size, dtype=float)

    ttic = time.time()

#-- div(u.n)
    du_cell = trsk.cell_flux_sums * uu_edge
    du_cell/= mesh.cell.area
    
    d2_cell = du_cell * cnfg.du_damp_2
    d4_cell = du_cell * cnfg.du_damp_4  # NB. sqrt(vk)

#-- D^2 = grad(vk * div(u.n))
    d2_edge = trsk.edge_grad_norm * d2_cell
    d4_edge = trsk.edge_grad_norm * d4_cell

#-- div(D^2)
    du_cell = trsk.cell_flux_sums * d4_edge
    du_cell/= mesh.cell.area
    
    d4_cell = du_cell * cnfg.du_damp_4  # NB. sqrt(vk)

#-- D^4 = grad(vk * div(D^2))
    d4_edge = trsk.edge_grad_norm * d4_cell
    
    ttoc = time.time()
    tcpu.computeDU = tcpu.computeDU + (ttoc - ttic)

    return d2_edge - d4_edge


def computeVU(mesh, trsk, cnfg, uu_edge):

    if (cnfg.vu_damp_2 == 0 and 
        cnfg.vu_damp_4 == 0):
        return np.zeros(mesh.edge.size, dtype=float)

    ttic = time.time()

    v2_edge = np.zeros(mesh.edge.size, dtype=float)
    v4_edge = np.zeros(mesh.edge.size, dtype=float)

    ttoc = time.time()
    tcpu.computeVU = tcpu.computeVU + (ttoc - ttic)

    return v2_edge - v4_edge


def computeCd(mesh, trsk, cnfg, hh_cell, uu_edge):

    VONK = 0.4  # von karman parameter

    hh_edge = hrmn_mean(
        hh_cell[mesh.edge.cell[:, 0] - 1], 
        hh_cell[mesh.edge.cell[:, 1] - 1])

    cd_edge = (
        VONK / np.log(0.5 * hh_edge / cnfg.loglaw_z0)
        ) ** +2

    cd_edge = np.minimum(cnfg.loglaw_hi, cd_edge)
    cd_edge = np.maximum(cnfg.loglaw_lo, cd_edge)

    vv_edge = trsk.edge_lsqr_perp * uu_edge * +1.

    ke_edge = 0.5 * (uu_edge ** 2 + 
                     vv_edge ** 2 )

    return cd_edge * np.sqrt(2. * ke_edge) / hh_edge


def computeMS(mesh, trsk, cnfg, 
              hh_cell, uu_edge, ht_cell, ut_edge):

    hh_dual, \
    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

#-- 1/2 * h * grad(div(du/dt * h))
    uh_edge = ut_edge * hh_edge

    uh_cell = trsk.cell_flux_sums * uh_edge
    uh_cell/= mesh.cell.area

    d1_edge = trsk.edge_grad_norm * uh_cell
    d1_edge*= hh_edge / 2.0

#-- 1/6 * h^2 * grad(div(du/dt)) 
    ut_cell = trsk.cell_flux_sums * ut_edge
    ut_cell/= mesh.cell.area

    d2_edge = trsk.edge_grad_norm * ut_cell
    d2_edge*= hh_edge ** 2 / 6.0

    return d1_edge - d2_edge


