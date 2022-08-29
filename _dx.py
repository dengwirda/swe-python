
import time
import numpy as np

#-- SWE spatial discretisation using TRSK-like operators
#--
#-- Darren Engwirda

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

def hrmn_mean(xone, xtwo):

#-- harmonic mean of two vectors (ie. biased toward lesser)

    return +2.0 * xone * xtwo / (xone + xtwo)


def invariant(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

#-- compute the discrete energy and enstrophy invariants

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

        dm_vert = \
            (trsk.dual_edge_sums * dm_edge) / 3.

        ds_dual = np.abs(
            sv_dual - sm_dual) / (dm_vert + UU_TINY)

        ds_dual = ds_dual ** 2

        sv_bias = np.sqrt(
            (trsk.edge_vert_sums * ds_dual) / 2.
        )


       #ds_edge = \
       #    (trsk.edge_vert_sums * ds_dual) / 2.
       
       #sv_bias = ds_edge / (dm_edge + UU_TINY)

    #-- upwind APVM, scale w grid
        um_edge = UU_TINY + \
            np.sqrt(uu_edge ** 2 + vv_edge ** 2)

        sv_wind = uu_edge * dn_edge / um_edge + \
                  vv_edge * dp_edge / um_edge

        ee_scal = mesh.edge.slen

        sv_bias = up_bias \
            + np.minimum(+3./8. - up_bias, sv_bias)

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


"""
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
"""


def computeKE(mesh, trsk, cnfg, 
              hh_cell, hh_edge, hh_dual, uu_edge, vv_edge,
              delta_t, up_bias):

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

            ke_edge, up_edge = upwinding(
                mesh, trsk, cnfg, 
                ke_dual, ke_dual, ke_cell, ke_edge, 
                uu_edge, vv_edge, delta_t, up_bias)

        if ("WEIGHT" in cnfg.ke_scheme):

            hh_thin = 1.0E+02
            hh_tiny = 1.0E-08
 
            hh_scal = \
                limit_div(hh_thin, hh_edge, hh_tiny)

            ke_edge = ke_edge * \
                ((1.0 + hh_scal) ** 2) ** 2

            ke_dual = trsk.dual_stub_sums * ke_edge
            ke_dual/= mesh.vert.area

            ke_cell = trsk.cell_wing_sums * ke_edge
            ke_cell/= mesh.cell.area

            hh_scal = \
                limit_div(hh_thin, hh_cell, hh_tiny)

            ke_cell = ke_cell / \
                ((1.0 + hh_scal) ** 2) ** 2

        else:

            ke_cell = trsk.cell_wing_sums * ke_edge
            ke_cell/= mesh.cell.area

            ke_dual = trsk.dual_stub_sums * ke_edge
            ke_dual/= mesh.vert.area

    if (cnfg.operators == "TRSK-MD"):

        up_edge = np.zeros(mesh.edge.size, dtype=float)

        if ("CENTRE" in cnfg.ke_scheme):

            ke_edge = 0.25 * uu_edge ** 2 * \
                  mesh.edge.clen * \
                  mesh.edge.vlen

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

            ke_edge, up_edge = upwinding(
                mesh, trsk, cnfg, 
                ke_dual, ke_dual, ke_cell, ke_edge, 
                uu_edge, vv_edge, delta_t, up_bias)

            ke_edge = 0.25 * ke_edge ** 1 * \
                  mesh.edge.clen * \
                  mesh.edge.vlen

        if ("WEIGHT" in cnfg.ke_scheme):

            hh_thin = 1.0E+02
            hh_tiny = 1.0E-08

            hs_edge = trsk.edge_vert_sums * hh_dual
            hs_edge*= 0.5E+00
            
            hh_scal = \
                limit_div(hh_thin, hs_edge, hh_tiny)

            ke_edge = ke_edge * \
                ((1.0 + hh_scal) ** 2) ** 2

            ke_dual = trsk.dual_edge_sums * ke_edge
            ke_dual/= mesh.vert.area

            ke_cell = trsk.cell_edge_sums * ke_edge
            ke_cell/= mesh.cell.area

            hh_scal = \
                limit_div(hh_thin, hh_cell, hh_tiny)

            ke_cell = ke_cell / \
                ((1.0 + hh_scal) ** 2) ** 2

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

#-- Divergence(-only) dissipation

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

