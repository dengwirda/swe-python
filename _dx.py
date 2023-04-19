
import time
import numpy as np

""" SWE spatial discretisation using TRSK-like operators
"""
#-- Darren Engwirda

HH_TINY        = 1.0E-08
HH_THIN        = 1.0E+02
UU_TINY        = 1.0E-12
PV_TINY        = 1.0E-16
KE_TINY        = 1.0E-04

LH_BIAS        = 2.0/3.0    # lo-hi flux bias

class base: pass
tcpu = base()
tcpu.thickness = 0.0E+00
tcpu.momentum_ = 0.0E+00
tcpu.upwinding = 0.0E+00
tcpu.compute_H = 0.0E+00
tcpu.computeKE = 0.0E+00
tcpu.computePV = 0.0E+00
tcpu.advect_PV = 0.0E+00
tcpu.computeVV = 0.0E+00
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

    zb_cell = flow.zb_cell

    vv_edge = computeVV(mesh, trsk, cnfg, uu_edge)

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

    zb_cell = flow.zb_cell

    vv_edge = computeVV(mesh, trsk, cnfg, uu_edge)

    hh_dual, \
    hh_edge = compute_H(mesh, trsk, cnfg, hh_cell, uu_edge)

    ke_edge = uu_edge ** 2
    ke_edge*= hh_edge * mesh.edge.area
    
    pe_cell = flow.grav * (
        hh_cell * 0.5 + zb_cell - np.min(zb_cell))

    pe_cell*= hh_cell * mesh.cell.area

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
              sm_dual, sv_dual, sv_cell, lo_edge, hi_edge,
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

        sv_edge = lo_edge * (0.0 + LH_BIAS) + \
                  hi_edge * (1.0 - LH_BIAS)

        sv_edge = \
            BIAS * sv_wind + (1. - BIAS) * sv_edge

    if (up_kind == "APVM"):

        dn_edge = trsk.edge_grad_norm * sv_cell
        dp_edge = trsk.edge_grad_perp * sv_dual

        sv_bias = np.full(
            mesh.edge.size, (+0.50), dtype=float)

    #-- upwind APVM, scale w time
        sv_apvm = uu_edge * dn_edge + \
                  vv_edge * dp_edge

        sv_edge = lo_edge * (0.0 + LH_BIAS) + \
                  hi_edge * (1.0 - LH_BIAS)

        sv_edge-= delta_t * sv_apvm
        
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

        sv_wind =(uu_edge * dn_edge + \
                  vv_edge * dp_edge ) \
                / um_edge
        sv_wind*= mesh.edge.slen
        
        sv_edge = lo_edge * (0.0 + LH_BIAS) + \
                  hi_edge * (1.0 - LH_BIAS)

        sv_edge-= up_min_ * sv_wind

    if (up_kind == "AUST-ADAPT"):
        
    #-- AUST: anticipated upstream method; APVM meets
    #-- LUST? Upwinds in multi-dimensional sense, vs.
    #-- LUST, which upwinds via tangential dir. only.

    #-- adapt. upwinding version

        dn_edge = trsk.edge_grad_norm * sv_cell
        dp_edge = trsk.edge_grad_perp * sv_dual

        dm_edge = ( ( (0.5 * (
            np.abs(dn_edge * mesh.edge.clen) +
            np.abs(dp_edge * mesh.edge.vlen)
        )   ) ** 2) ** 2) ** 2

        dm_vert = np.sqrt(np.sqrt(np.sqrt(
            (trsk.dual_edge_sums * dm_edge) / 3.
        )   )   ) + sv_tiny

        ds_dual = ( ( (
            (sv_dual - sm_dual)  / dm_vert / 1.125
            ) ** 2) ** 2) ** 2
       
        sv_bias = np.sqrt(np.sqrt(np.sqrt(
            (trsk.edge_vert_sums * ds_dual) / 2.
        )   )   )

    #-- upwind APVM, scale w grid
        um_edge = UU_TINY + \
            np.sqrt(uu_edge ** 2 + vv_edge ** 2)

        sv_wind =(uu_edge * dn_edge + \
                  vv_edge * dp_edge ) \
                / um_edge
        sv_wind*= mesh.edge.slen

        sv_bias = up_min_ + \
            np.minimum(up_max_ - up_min_, sv_bias)

        lh_bias = \
            np.maximum(1.0 / 6.0, np.sqrt(sv_bias))
            
        sv_edge = lo_edge * (0.0 + lh_bias) + \
                  hi_edge * (1.0 - lh_bias)
                  
        sv_edge-= sv_bias * sv_wind

    ttoc = time.time()
    tcpu.upwinding = tcpu.upwinding + (ttoc - ttic)

    return sv_edge, sv_bias


def compute_H(mesh, trsk, cnfg, hh_cell, uu_edge):

#-- compute discrete thickness

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

#-- compute discrete vorticity

    ttic = time.time()

    if (cnfg.operators == "TRSK-CV"):

    #-- RV+f on rhombi, PV on edge - more compact stencil?        
        rv_dual = trsk.dual_curl_sums * uu_edge
        rv_dual/= mesh.vert.area
        
        if(cnfg.no_advect): rv_dual *= 0.

        pv_dual = rv_dual + ff_dual
        pv_dual/= hh_dual
        
        rv_cell = trsk.cell_kite_sums * rv_dual
        rv_cell/= mesh.cell.area

       #if(cnfg.no_advect): rv_cell *= 0.

        pv_cell = rv_cell + ff_cell
        pv_cell/= hh_cell

    #-- compute curl on rhombi -- a'la Gassmann
        rm_edge = trsk.quad_curl_sums * uu_edge
        rm_edge/= mesh.quad.area

        if(cnfg.no_advect): rm_edge *= 0.

        hv_edge = trsk.edge_stub_sums * hh_dual
        hv_edge/= mesh.edge.area

        hm_edge = 1. / 2. * hh_edge \
                + 1. / 2. * hv_edge
        
        lo_edge = rm_edge + ff_edge
        lo_edge/= hm_edge

    #-- average rhombi to dual -- a'la Gassmann
        lo_dual = trsk.dual_edge_sums * lo_edge / 3.0

    #-- compute high(er)-order RV + PV on edges:
    #-- pv_edge = pv_dual + (xe - xv) * pv_d/dx
        hi_edge = trsk.edge_vert_sums * pv_dual / 2.0
        hi_edge+= trsk.edge_dual_reco * lo_dual

        if ("UPWIND" in cnfg.pv_scheme):

            pv_edge, up_edge = upwinding(
                mesh, trsk, cnfg, 
                lo_dual, pv_dual, pv_cell, 
                lo_edge, hi_edge,
                uu_edge, vv_edge, 
                delta_t, PV_TINY, 
                cnfg.pv_upwind, 
                cnfg.pv_min_up, cnfg.pv_max_up)
                
        if ("CENTRE" in cnfg.pv_scheme):
        
            up_edge = np.zeros(
                mesh.edge.size, dtype=np.float64)
        
            pv_edge = \
                lo_edge * (0.0 + LH_BIAS) + \
                hi_edge * (1.0 - LH_BIAS)


    if (cnfg.operators == "TRSK-MD"):

        rv_dual = trsk.dual_curl_sums * uu_edge
        rv_dual/= mesh.vert.area

        if(cnfg.no_advect): rv_dual *= 0.

        pv_dual = rv_dual + ff_dual
        pv_dual/= hh_dual

        rv_cell = trsk.cell_kite_sums * rv_dual
        rv_cell/= mesh.cell.area

       #rv_cell*= (cnfg.no_advect == False)

        pv_cell = rv_cell + ff_cell
        pv_cell/= hh_cell

        rv_edge = trsk.edge_vert_sums * rv_dual / 2.0
        pv_edge = trsk.edge_vert_sums * pv_dual / 2.0
        
        lo_edge = pv_edge
        hi_edge = pv_edge
        
        lo_dual = trsk.dual_edge_sums * lo_edge / 3.0
        
        if ("UPWIND" in cnfg.pv_scheme):

            pv_edge, up_edge = upwinding(
                mesh, trsk, cnfg, 
                lo_dual, pv_dual, pv_cell, 
                lo_edge, hi_edge,
                uu_edge, vv_edge, 
                delta_t, PV_TINY, 
                cnfg.pv_upwind, 
                cnfg.pv_min_up, cnfg.pv_max_up)
                
        if ("CENTRE" in cnfg.pv_scheme):
        
            up_edge = np.zeros(
                mesh.edge.size, dtype=np.float64)
        
            pv_edge = \
                lo_edge * (0.0 + LH_BIAS) + \
                hi_edge * (1.0 - LH_BIAS)

    ttoc = time.time()
    tcpu.computePV = tcpu.computePV + (ttoc - ttic)

    return rv_dual, pv_dual, rv_cell, pv_cell, \
           pv_edge, up_edge


def computeKE(mesh, trsk, cnfg, 
              hh_cell, hh_edge, hh_dual, uu_edge, vv_edge,
              delta_t):

#-- reconstruct kinetic energy

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

            lo_edge = ke_edge
            hi_edge = ke_edge

            lo_dual = \
                trsk.dual_edge_sums * ke_edge / 3.0

            ke_edge, up_edge = upwinding(
                mesh, trsk, cnfg, 
                lo_dual, ke_dual, ke_cell, 
                lo_edge, hi_edge, 
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

            lo_edge = ke_edge
            hi_edge = ke_edge

            lo_dual = \
                trsk.dual_edge_sums * ke_edge / 3.0

            ke_edge, up_edge = upwinding(
                mesh, trsk, cnfg, 
                lo_dual, ke_dual, ke_cell, 
                hi_edge, hi_edge, 
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

#-- energy-neutral PV. flux

    ttic = time.time()

    pv_flux = -.5 * (
        trsk.edge_flux_perp * (pv_edge * uh_edge) +
        pv_edge * (trsk.edge_flux_perp * uh_edge)
    )

    ttoc = time.time()
    tcpu.advect_PV = tcpu.advect_PV + (ttoc - ttic)

    return pv_flux
    
    
def computeVV(mesh, trsk, cnfg, uu_edge):

#-- get tangential velocity

    ttic = time.time()

    vv_edge = trsk.edge_lsqr_perp * uu_edge

    ttoc = time.time()
    tcpu.computeVV = tcpu.computeVV + (ttoc - ttic)

    return vv_edge


def computeDU(mesh, trsk, cnfg, uu_edge):

#-- damping div^k operators

    if (cnfg.du_damp_2 == 0 and 
        cnfg.du_damp_4 == 0):
        return np.zeros(mesh.edge.size, dtype=float)

    ttic = time.time()

#-- div(u.n)
    du_cell = trsk.cell_flux_sums * uu_edge
    du_cell/= mesh.cell.area

#-- D^2 = vk * grad(div(u.n))
    du_edge = trsk.edge_grad_norm * du_cell
    
    d2_edge = cnfg.du_damp_2 * du_edge
    d4_edge = cnfg.du_damp_4 * du_edge
      
#-- div(D^2)
    du_cell = trsk.cell_flux_sums * d4_edge
    du_cell/= mesh.cell.area
    
#-- D^4 = vk * grad(div(D^2))              
    d4_edge = trsk.edge_grad_norm * du_cell
    d4_edge*= cnfg.vu_damp_4        # NB. sqrt(vk)
    
    ttoc = time.time()
    tcpu.computeDU = tcpu.computeDU + (ttoc - ttic)

    return d2_edge - d4_edge


def computeVU(mesh, trsk, cnfg, uu_edge):

#-- viscous del^k operators

    if (cnfg.vu_damp_2 == 0 and 
        cnfg.vu_damp_4 == 0):
        return np.zeros(mesh.edge.size, dtype=float)

    ttic = time.time()

#-- div(u.n)
    du_cell = trsk.cell_flux_sums * uu_edge
    du_cell/=(mesh.cell.area /
              cnfg.vu_du_mul )
    
#-- rot(u.n)
    rv_dual = trsk.dual_flux_sums * uu_edge
    rv_dual/= mesh.vert.area

#-- V^2 = vk * grad(div(u.n)) - 
#--       vk * grad(rot(u.n))
    vu_edge = trsk.edge_grad_norm * du_cell - \
              trsk.edge_grad_perp * rv_dual
              
    v2_edge = cnfg.vu_damp_2 * vu_edge
    v4_edge = cnfg.vu_damp_4 * vu_edge
      
#-- div(V^2)
    du_cell = trsk.cell_flux_sums * v4_edge
    du_cell/=(mesh.cell.area /
              cnfg.vu_du_mul )
    
#-- rot(V^2)
    rv_dual = trsk.dual_flux_sums * v4_edge
    rv_dual/= mesh.vert.area
    
#-- V^4 = vk * grad(div(V^2)) - 
#--       vk * grad(rot(V^2))              
    v4_edge = trsk.edge_grad_norm * du_cell - \
              trsk.edge_grad_perp * rv_dual
    v4_edge*= cnfg.vu_damp_4        # NB. sqrt(vk)
    
    ttoc = time.time()
    tcpu.computeVU = tcpu.computeVU + (ttoc - ttic)

    return v2_edge - v4_edge


def computeCd(mesh, trsk, cnfg, hh_cell, uu_edge):

#-- loglaw bottom drag term

    VONK = 0.4  # von karman parameter

    hh_edge = hrmn_mean(
        hh_cell[mesh.edge.cell[:, 0] - 1],
        hh_cell[mesh.edge.cell[:, 1] - 1])

    hh_edge = np.maximum(HH_TINY, hh_edge)

    # NB. log(1+z/z0) "fix" to loglaw
    cd_edge = (
        VONK / np.log(
            1.0 + 0.5 * hh_edge / cnfg.loglaw_z0)
        ) ** +2

    cd_edge = np.minimum(cnfg.loglaw_hi, cd_edge)
    cd_edge = np.maximum(cnfg.loglaw_lo, cd_edge)

    vv_edge = \
        computeVV(mesh, trsk, cnfg, uu_edge)

    ke_edge = 0.5 * (uu_edge ** 2 + 
                     vv_edge ** 2 )

    return cd_edge * np.sqrt(2. * ke_edge) / hh_edge


def computeMS(mesh, trsk, cnfg, 
              hh_cell, uu_edge, ht_cell, ut_edge):

#-- "dispersive wave" terms

    ttic = time.time()

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
    
    ttoc = time.time()
    tcpu.computeDW = tcpu.computeDW + (ttoc - ttic)

    return d1_edge - d2_edge


