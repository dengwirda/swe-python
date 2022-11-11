
import time
import numpy as np

""" SWE time integration via various Runge-Kutta methods
"""
#-- Darren Engwirda

from _dx import tcpu
from rhs import rhs_all_u, rhs_all_h

def step_RK22(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

#-- A 2-stage RK2 + FB scheme, a'la ROMS:
#-- A.F. Shchepetkin, J.C. McWilliams (2005): The regional oceanic 
#-- modeling system (ROMS): a split-explicit, free-surface, 
#-- topography-following-coordinate oceanic model
#-- doi.org/10.1016/j.ocemod.2004.08.002

#-- but with thickness updated via an SSP-RK2 approach

#-- 1st RK + FB stage

    ttic = time.time()

    if cnfg.fb_weight:
        BETA = cnfg.fb_weight[0] * ("FB" in cnfg.integrate)
    else:
        BETA = (1.0 / 3.0) * ("FB" in cnfg.integrate)

    rh_cell = rhs_all_h(
        mesh, trsk, flow, cnfg, hh_cell, uu_edge)

    h1_cell = (
        hh_cell - 1.0 / 1.0 * cnfg.time_step * rh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    hb_cell = h1_cell * (0.0 + 1.0 * BETA) + \
              hh_cell * (1.0 - 1.0 * BETA)

    ru_edge, \
    ke_cell, ke_dual, ke_bias, \
    rv_cell, pv_cell, \
    rv_dual, pv_dual, pv_bias = rhs_all_u(
        mesh, trsk, flow, cnfg, hb_cell, uu_edge)

    u1_edge = (
        uu_edge - 1.0 / 1.0 * cnfg.time_step * ru_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 2nd RK + FB stage

    ttic = time.time()

    if cnfg.fb_weight:
        BETA = cnfg.fb_weight[1] * ("FB" in cnfg.integrate)
    else:
        BETA = (2.0 / 3.0) * ("FB" in cnfg.integrate)

    hm_cell = 0.5 * hh_cell + 0.5 * h1_cell
    um_edge = 0.5 * uu_edge + 0.5 * u1_edge

    rh_cell = rhs_all_h(
        mesh, trsk, flow, cnfg, h1_cell, u1_edge)

    h2_cell = (
        hm_cell - 1.0 / 2.0 * cnfg.time_step * rh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    hb_cell = h2_cell * (0.0 + 0.5 * BETA) + \
              h1_cell * (0.5 - 0.5 * BETA) + \
              hh_cell * (0.5)

    ru_edge, \
    ke_cell, ke_dual, ke_bias, \
    rv_cell, pv_cell, \
    rv_dual, pv_dual, pv_bias = rhs_all_u(
        mesh, trsk, flow, cnfg, hb_cell, um_edge)

    u2_edge = (
        uu_edge - 1.0 / 1.0 * cnfg.time_step * ru_edge
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

#-- 1st RK + FB stage

    ttic = time.time()
    
    if cnfg.fb_weight:
        print('no def')
        BETA = cnfg.fb_weight[0] * ("FB" in cnfg.integrate)
    else:
        BETA = (1.0 / 3.0) * ("FB" in cnfg.integrate)


    rh_cell = rhs_all_h(
        mesh, trsk, flow, cnfg, hh_cell, uu_edge)

    h1_cell = (
        hh_cell - 1.0 / 3.0 * cnfg.time_step * rh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    hb_cell = h1_cell * (0.0 + 1.0 * BETA) + \
              hh_cell * (1.0 - 1.0 * BETA)

    ru_edge, \
    ke_cell, ke_dual, ke_bias, \
    rv_cell, pv_cell, \
    rv_dual, pv_dual, pv_bias = rhs_all_u(
        mesh, trsk, flow, cnfg, hb_cell, uu_edge)

    u1_edge = (
        uu_edge - 1.0 / 3.0 * cnfg.time_step * ru_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 2nd RK + FB stage

    ttic = time.time()

    if cnfg.fb_weight:
        BETA = cnfg.fb_weight[1] * ("FB" in cnfg.integrate)
    else:
        BETA = (1.0 / 2.0) * ("FB" in cnfg.integrate)
   
    rh_cell = rhs_all_h(
        mesh, trsk, flow, cnfg, h1_cell, u1_edge)

    h2_cell = (
        hh_cell - 1.0 / 2.0 * cnfg.time_step * rh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    # centred at 1/4 for 1/2 step?
    hb_cell = h2_cell * (0.0 + 1.0 * BETA) + \
              hh_cell * (1.0 - 1.0 * BETA)

    ru_edge, \
    ke_cell, ke_dual, ke_bias, \
    rv_cell, pv_cell, \
    rv_dual, pv_dual, pv_bias = rhs_all_u(
        mesh, trsk, flow, cnfg, hb_cell, u1_edge)

    u2_edge = (
        uu_edge - 1.0 / 2.0 * cnfg.time_step * ru_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 3rd RK + FB stage

    ttic = time.time()

    if cnfg.fb_weight:
        BETA = cnfg.fb_weight[2] * ("FB" in cnfg.integrate)
    else:
        BETA = (89.0 / 300.0) * ("FB" in cnfg.integrate)

    rh_cell = rhs_all_h(
        mesh, trsk, flow, cnfg, h2_cell, u2_edge)

    h3_cell = (
        hh_cell - 1.0 / 1.0 * cnfg.time_step * rh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    # centred at 1/2 for 1/1 step?
    hb_cell = h3_cell * (0.0 + 1.0 * BETA) + \
              h2_cell * (1.0 - 2.0 * BETA) + \
              hh_cell * (0.0 + 1.0 * BETA)

    ru_edge, \
    ke_cell, ke_dual, ke_bias, \
    rv_cell, pv_cell, \
    rv_dual, pv_dual, pv_bias = rhs_all_u(
        mesh, trsk, flow, cnfg, hb_cell, u2_edge)

    u3_edge = (
        uu_edge - 1.0 / 1.0 * cnfg.time_step * ru_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

    return h3_cell, u3_edge, ke_cell, ke_dual, \
           rv_cell, pv_cell, \
           rv_dual, pv_dual, ke_bias, pv_bias


def step_SP33(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

#-- standard SSP-RK(3,3) method:
#-- S. Gottlieb (2005): On high order strong stability preserving 
#-- runge-kutta and multi step time discretizations
#-- doi.org/10.1007/BF02728985

#-- 1st RK + FB stage

    ttic = time.time()

    rh_cell = rhs_all_h(
        mesh, trsk, flow, cnfg, hh_cell, uu_edge)

    h1_cell = (
        hh_cell - 1.0 / 1.0 * cnfg.time_step * rh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    ru_edge, \
    ke_cell, ke_dual, ke_bias, \
    rv_cell, pv_cell, \
    rv_dual, pv_dual, pv_bias = rhs_all_u(
        mesh, trsk, flow, cnfg, hh_cell, uu_edge)

    u1_edge = (
        uu_edge - 1.0 / 1.0 * cnfg.time_step * ru_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 2nd RK + FB stage

    ttic = time.time()

    rh_cell = rhs_all_h(
        mesh, trsk, flow, cnfg, h1_cell, u1_edge)

    h2_cell = (
        3.0 / 4.0 * hh_cell + 
        1.0 / 4.0 * h1_cell - 
            1.0 / 4.0 * cnfg.time_step * rh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    ru_edge, \
    ke_cell, ke_dual, ke_bias, \
    rv_cell, pv_cell, \
    rv_dual, pv_dual, pv_bias = rhs_all_u(
        mesh, trsk, flow, cnfg, h1_cell, u1_edge)

    u2_edge = (
        3.0 / 4.0 * uu_edge +
        1.0 / 4.0 * u1_edge - 
            1.0 / 4.0 * cnfg.time_step * ru_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 3rd RK + FB stage

    ttic = time.time()

    rh_cell = rhs_all_h(
        mesh, trsk, flow, cnfg, h2_cell, u2_edge)

    h3_cell = (
        1.0 / 3.0 * hh_cell +
        2.0 / 3.0 * h2_cell - 
            2.0 / 3.0 * cnfg.time_step * rh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    ru_edge, \
    ke_cell, ke_dual, ke_bias, \
    rv_cell, pv_cell, \
    rv_dual, pv_dual, pv_bias = rhs_all_u(
        mesh, trsk, flow, cnfg, h2_cell, u2_edge)

    u3_edge = (
        1.0 / 3.0 * uu_edge +
        2.0 / 3.0 * u2_edge - 
            2.0 / 3.0 * cnfg.time_step * ru_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

    return h3_cell, u3_edge, ke_cell, ke_dual, \
           rv_cell, pv_cell, \
           rv_dual, pv_dual, ke_bias, pv_bias


