

def step_BT22(mesh, trsk, flow, cnfg, hh_cell, uu_edge, 
                                      ru_edge) :

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

    rh_cell = rhs_btr_h(
        mesh, trsk, flow, cnfg, hh_cell, uu_edge)

    h1_cell = (
        hh_cell - 1.0 / 1.0 * cnfg._btr_step * rh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    hb_cell = h1_cell * (0.0 + 1.0 * BETA) + \
              hh_cell * (1.0 - 1.0 * BETA)

    hz_cell =(hb_cell + zb_cell) * flow.grav

    hz_grad = trsk.edge_grad_norm * hz_cell

    u1_edge = uu_edge - 1.0 / 1.0 * cnfg._btr_step * (
        hz_grad - ru_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 2nd RK + FB stage

    ttic = time.time()

    BETA = (2.0 / 3.0) * ("FB" in cnfg.integrate)

    hm_cell = 0.5 * hh_cell + 0.5 * h1_cell
    um_edge = 0.5 * uu_edge + 0.5 * u1_edge

    rh_cell = rhs_all_h(
        mesh, trsk, flow, cnfg, h1_cell, u1_edge)

    h2_cell = (
        hm_cell - 1.0 / 2.0 * cnfg._btr_step * rh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    hb_cell = h2_cell * (0.0 + 0.5 * BETA) + \
              h1_cell * (0.5 - 0.5 * BETA) + \
              hh_cell * (0.5)

    hz_cell =(hb_cell + zb_cell) * flow.grav

    hz_grad = trsk.edge_grad_norm * hz_cell

    u2_edge = uu_edge - 1.0 / 1.0 * cnfg._btr_step * (
        hz_grad - ru_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

    return h2_cell, u2_edge, hz_grad


def step_SE22(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

#-- A split version of the RK22-FB scheme, with the surface waves
#-- evaluated within each parent RK22-FB stage, using the RK22-FB
#-- scheme on a sub-timestep. 

    ff_cell = flow.ff_cell
    ff_edge = flow.ff_edge
    ff_dual = flow.ff_vert

    zb_cell = flow.zb_cell

    cnfg._btr_subs = 3
    cnfg._btr_step = cnfg.time_step / cnfg._btr_subs

#-- 1st-stage: (n+0/1)
#-- assemble explicit momentum forcings - btr terms

    ttic = time.time()

    ru_edge = rhs_slw_u()

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- run btr loop; advance to h1 and mean(btr) terms

    ru_edge = uu_damp - qh_flux - ke_grad
    
    bu_edge = uu_edge[:]
    bh_cell = hh_cell[:]
    bt_grad = ke_grad[:] * 0.0
    
    for isub in range(cnfg._btr_subs):

        bh_cell, bu_edge, \
        hz_grad = step_BT22(
            mesh, trsk,
            flow, cnfg, bh_cell, bu_edge, ru_edge)

        bt_grad+= hz_grad / cnfg._btr_subs

    hk_grad = ke_grad + bt_grad

#-- correct onto long time-step; advance to (h1,u1)

    h1_cell = bh_cell
    u1_edge = uu_edge - 1.0 / 1.0 * cnfg.time_step * (
        hk_grad + qh_flux - uu_damp
    )

#-- 2nd-stage: (n+1/2)
#-- assemble explicit momentum forcings - btr terms

    ttic = time.time()

    rk_bias = 1./300.  # tiny offcentre

    hm_cell = (0.5 - rk_bias) * hh_cell + \
              (0.5 + rk_bias) * h1_cell
    um_edge = (0.5 - rk_bias) * uu_edge + \
              (0.5 + rk_bias) * u1_edge

    vm_edge = trsk.edge_lsqr_perp * um_edge

    hm_dual, \
    hm_edge = compute_H(mesh, trsk, cnfg, hm_cell, um_edge)

    uh_edge = um_edge * hm_edge

    ke_dual, ke_cell, ke_bias = computeKE(
        mesh, trsk, cnfg, 
        hm_cell, hm_edge, hm_dual, 
        um_edge, vm_edge,
        +1.0 / 2.0 * cnfg.time_step)

    ke_grad = trsk.edge_grad_norm * ke_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, trsk, cnfg, 
        hm_cell, hm_edge, hm_dual, um_edge, vm_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, um_edge)

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- run btr loop; advance to h2 and mean(btr) terms

    ru_edge = uu_damp - qh_flux - ke_grad
    
    bu_edge = uu_edge[:]
    bh_cell = hh_cell[:]
    bt_grad = ke_grad[:] * 0.0

    for isub in range(cnfg._btr_subs):

        bh_cell, bu_edge, \
        hz_grad = step_BT22(
            mesh, trsk,
            flow, cnfg, bh_cell, bu_edge, ru_edge)

        bt_grad+= hz_grad / cnfg._btr_subs

    hk_grad = ke_grad + bt_grad

#-- correct onto long time-step; advance to (h2,u2)

    h2_cell = bh_cell

    u2_edge = uu_edge - 1.0 / 1.0 * cnfg.time_step * (
        hk_grad + qh_flux - uu_damp
    )

    return h2_cell, u2_edge, ke_cell, ke_dual, \
           rv_cell, pv_cell, \
           rv_dual, pv_dual, ke_bias, pv_bias


def step_BT32(mesh, trsk, flow, cnfg, hh_cell, uu_edge, 
                                      ru_edge) :

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
    
    rh_cell = rhs_all_h(
        mesh, trsk, flow, cnfg, hh_cell, uu_edge)

    h1_cell = (
        hh_cell - 1.0 / 3.0 * cnfg._btr_step * rh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    hb_cell = h1_cell * (0.0 + 1.0 * BETA) + \
              hh_cell * (1.0 - 1.0 * BETA)

    hz_cell =(hb_cell + zb_cell) * flow.grav

    hz_grad = trsk.edge_grad_norm * hz_cell

    u1_edge = uu_edge - 1.0 / 3.0 * cnfg._btr_step * (
        hz_grad - ru_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 2nd RK + FB stage

    ttic = time.time()

    BETA = (1.0 / 2.0) * ("FB" in cnfg.integrate)
   
    rh_cell = rhs_all_h(
        mesh, trsk, flow, cnfg, h1_cell, u1_edge)

    h2_cell = (
        hh_cell - 1.0 / 2.0 * cnfg._btr_step * rh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    # centred at 1/4 for 1/2 step?
    hb_cell = h2_cell * (0.0 + 1.0 * BETA) + \
              hh_cell * (1.0 - 1.0 * BETA)

    hz_cell =(hb_cell + zb_cell) * flow.grav

    hz_grad = trsk.edge_grad_norm * hz_cell

    u2_edge = uu_edge - 1.0 / 2.0 * cnfg._btr_step * (
        hz_grad - ru_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- 3rd RK + FB stage

    ttic = time.time()

    BETA = (89. / 300) * ("FB" in cnfg.integrate)
   
    rh_cell = rhs_all_h(
        mesh, trsk, flow, cnfg, h2_cell, u2_edge)

    h3_cell = (
        hh_cell - 1.0 / 1.0 * cnfg._btr_step * rh_cell
    )

    ttoc = time.time()
    tcpu.thickness = tcpu.thickness + (ttoc - ttic)

    ttic = time.time()

    # centred at 1/2 for 1/1 step?
    hb_cell = h3_cell * (0.0 + 1.0 * BETA) + \
              h2_cell * (1.0 - 2.0 * BETA) + \
              hh_cell * (0.0 + 1.0 * BETA)

    hz_cell =(hb_cell + zb_cell) * flow.grav

    hz_grad = trsk.edge_grad_norm * hz_cell

    u3_edge = uu_edge - 1.0 / 1.0 * cnfg._btr_step * (
        hz_grad - ru_edge
    )

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

    return h3_cell, u3_edge, hz_grad


def step_SE32(mesh, trsk, flow, cnfg, hh_cell, uu_edge):

#-- A split version of the RK32-FB scheme, with the surface waves
#-- evaluated within each parent RK32-FB stage, using the RK32-FB
#-- scheme on a sub-timestep.  

    ff_cell = flow.ff_cell
    ff_edge = flow.ff_edge
    ff_dual = flow.ff_vert

    zb_cell = flow.zb_cell

    bt_grad = np.zeros(
        (mesh.edge.size), dtype=np.float64)

#-- 1st-stage: (n+0/1)
#-- assemble explicit momentum forcings - btr terms

    ttic = time.time()

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

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, trsk, cnfg, 
        hh_cell, hh_edge, hh_dual, 
        uu_edge, vv_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, uu_edge)

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- run btr loop; advance to h1 and mean(btr) terms

    cnfg._btr_step = cnfg.time_step / 6.0

    bt_grad = bt_grad * 0.0
    ru_edge = uu_damp - qh_flux - ke_grad

    bh_cell, bu_edge, hz_grad = step_BT32(
        mesh, trsk, flow, cnfg , hh_cell, uu_edge, ru_edge)

    bt_grad+= hz_grad * 1.0 / 6.0

    bh_cell, bu_edge, hz_grad = step_BT32(
        mesh, trsk, flow, cnfg , bh_cell, bu_edge, ru_edge)

    bt_grad+= hz_grad * 1.0 / 6.0
    
#-- correct onto long time-step; advance to (h1,u1)

    h1_cell = bh_cell
    u1_edge = uu_edge - 1.0 / 3.0 * cnfg.time_step * (
        3.0 * bt_grad + 
            ke_grad + qh_flux - uu_damp
    )
    u1_edge = 0.50 * (u1_edge + bu_edge)

#-- 2nd-stage: (n+1/3)
#-- assemble explicit momentum forcings - btr terms

    ttic = time.time()

    v1_edge = trsk.edge_lsqr_perp * u1_edge

    h1_dual, \
    h1_edge = compute_H(mesh, trsk, cnfg, h1_cell, u1_edge)

    uh_edge = u1_edge * h1_edge

    ke_dual, ke_cell, ke_bias = computeKE(
        mesh, trsk, cnfg, 
        h1_cell, h1_edge, h1_dual, 
        u1_edge, v1_edge,
        +1.0 / 2.0 * cnfg.time_step)

    ke_grad = trsk.edge_grad_norm * ke_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, trsk, cnfg, 
        h1_cell, h1_edge, h1_dual, 
        u1_edge, v1_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, u1_edge)

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- run btr loop; advance to h2 and mean(btr) terms

    cnfg._btr_step = cnfg.time_step / 4.0

    bt_grad = bt_grad * 0.0
    ru_edge = uu_damp - qh_flux - ke_grad
    
    bh_cell, bu_edge, hz_grad = step_BT32(
        mesh, trsk, flow, cnfg , hh_cell, uu_edge, ru_edge)

    bt_grad+= hz_grad * 1.0 / 4.0

    bh_cell, bu_edge, hz_grad = step_BT32(
        mesh, trsk, flow, cnfg , bh_cell, bu_edge, ru_edge)

    bt_grad+= hz_grad * 1.0 / 4.0

#-- correct onto long time-step; advance to (h2,u2)

    h2_cell = bh_cell
    u2_edge = uu_edge - 1.0 / 2.0 * cnfg.time_step * (
        2.0 * bt_grad + 
            ke_grad + qh_flux - uu_damp
    )
    u2_edge = 0.50 * (u2_edge + bu_edge)

#-- 3rd-stage: (n+1/2)
#-- assemble explicit momentum forcings - btr terms

    ttic = time.time()

    v2_edge = trsk.edge_lsqr_perp * u2_edge

    h2_dual, \
    h2_edge = compute_H(mesh, trsk, cnfg, h2_cell, u2_edge)

    uh_edge = u2_edge * h2_edge

    ke_dual, ke_cell, ke_bias = computeKE(
        mesh, trsk, cnfg, 
        h2_cell, h2_edge, h2_dual, 
        u2_edge, v2_edge,
        +1.0 / 2.0 * cnfg.time_step)

    ke_grad = trsk.edge_grad_norm * ke_cell

    rv_dual, pv_dual, rv_cell, pv_cell, \
    pv_edge, pv_bias = computePV(
        mesh, trsk, cnfg, 
        h2_cell, h2_edge, h2_dual, 
        u2_edge, v2_edge, 
        ff_dual, ff_edge, ff_cell, 
        +1.0 / 2.0 * cnfg.time_step)

    qh_flux = advect_PV(mesh, trsk, cnfg, uh_edge, pv_edge)

    uu_damp = computeDU(mesh, trsk, cnfg, u2_edge)

    ttoc = time.time()
    tcpu.momentum_ = tcpu.momentum_ + (ttoc - ttic)

#-- run btr loop; advance to h3 and mean(btr) terms

    cnfg._btr_step = cnfg.time_step / 4.0

    bt_grad = bt_grad * 0.0
    ru_edge = uu_damp - qh_flux - ke_grad

    bh_cell, bu_edge, hz_grad = step_BT32(
        mesh, trsk, flow, cnfg , hh_cell, uu_edge, ru_edge)

    bt_grad+= hz_grad * 1.0 / 4.0

    bh_cell, bu_edge, hz_grad = step_BT32(
        mesh, trsk, flow, cnfg , bh_cell, bu_edge, ru_edge)

    bt_grad+= hz_grad * 1.0 / 4.0

    bh_cell, bu_edge, hz_grad = step_BT32(
        mesh, trsk, flow, cnfg , bh_cell, bu_edge, ru_edge)

    bt_grad+= hz_grad * 1.0 / 4.0

    bh_cell, bu_edge, hz_grad = step_BT32(
        mesh, trsk, flow, cnfg , bh_cell, bu_edge, ru_edge)

    bt_grad+= hz_grad * 1.0 / 4.0
    
#-- correct onto long time-step; advance to (h3,u3)

    h3_cell = bh_cell
    u3_edge = uu_edge - 1.0 / 1.0 * cnfg.time_step * (
        1.0 * bt_grad + 
            ke_grad + qh_flux - uu_damp
    )
    u3_edge = 0.50 * (u3_edge + bu_edge)
    
    return h3_cell, u3_edge, ke_cell, ke_dual, \
           rv_cell, pv_cell, \
           rv_dual, pv_dual, ke_bias, pv_bias

    
