
import os
import time
import numpy as np

import jigsawpy
import argparse

from msh import load_mesh, load_flow
from ops import trsk_mats


def main(name):
    """
    SWE: a very simple driver to solve the SWE via spherical 
    MPAS meshes.

    """
    # Authors: Darren Engwirda

    class base: pass

    mesh = load_mesh(name)
    trsk = trsk_mats(mesh)
    flow = load_flow(name)

    cnfg = base()
    cnfg.iter = 1000
    cnfg.step = 100.
    cnfg.freq = 25
    cnfg.grav = 9.81

    unrm = flow.unrm[0, :, 0]; uold = unrm
    hdel = flow.hdel[0, :, 0]; hold = hdel
    zbot = flow.zbot[:]

    for step in range(cnfg.iter):

    #-- a TVD-RK2 + FB scheme, a'la ROMS (but with variables
    #-- updated via the standard "monotone" TVD-RK approach)

        fcem = flow.frot["edge"][:, 0]

        BETA = 1.0 / 3.0

        divh = eval_cell_div_(trsk, mesh, unrm, hdel)

        h1st = hdel - 1.0 * cnfg.step * divh

        hbar = hdel * (1.0 - BETA) + h1st * (0.0 + BETA)

        hpgf = hbar[mesh.edge.cell[:, 1] - 1] - \
               hbar[mesh.edge.cell[:, 0] - 1]
        hpgf = hpgf / mesh.edge.clen

        uprp = trsk.edge_reco_perp * unrm

        urhs = uprp * fcem + hpgf * cnfg.grav

        u1st = unrm - 1.0 * cnfg.step * urhs
        

        htmp = hdel * 0.5 + h1st * 0.5
        utmp = unrm * 0.5 + u1st * 0.5

        divh = eval_cell_div_(trsk, mesh, u1st, h1st)

        h2nd = htmp - 0.5 * cnfg.step * divh

        BETA = 2.0 / 3.0

        hbar = h1st * (1.0 - BETA) + h2nd * (0.0 + BETA)

        hpgf = hbar[mesh.edge.cell[:, 1] - 1] - \
               hbar[mesh.edge.cell[:, 0] - 1]
        hpgf = hpgf / mesh.edge.clen

        uprp = trsk.edge_reco_perp * u1st

        urhs = uprp * fcem + hpgf * cnfg.grav

        u2nd = utmp - 0.5 * cnfg.step * urhs
        

        eta_ = trsk.dual_curl_sums * unrm


        uold = unrm; unrm = u2nd
        hold = hdel; hdel = h2nd


        if (step % cnfg.freq == 0):

            check_conserve(cnfg, mesh, unrm, 
                           hdel, zbot)

            mout = jigsawpy.jigsaw_msh_t()
            mout.point = np.zeros(
                mesh.cell.size, dtype=mout.VERT3_t)
            mout.point["coord"][:, 0] = mesh.cell.xpos[:]
            mout.point["coord"][:, 1] = mesh.cell.ypos[:]
            mout.point["coord"][:, 2] = mesh.cell.zpos[:]
            mout.tria3 = np.zeros(
                mesh.vert.size, dtype=mout.TRIA3_t)
            mout.tria3["index"] = mesh.vert.cell[:] - 1
            mout.value = hdel[:]

            jigsawpy.savevtk(os.path.join(
                "out", "flow" + str(step)), mout)

    return


def eval_cell_div_(trsk, mesh, unrm, hdel):

#-- eval div(uh) term: finite-volume flux integration around
#-- cell contours.

    icel = mesh.edge.cell[:, 0] - 1
    jcel = mesh.edge.cell[:, 1] - 1

    if True:
 
        hmid = (hdel[icel] + hdel[jcel]) * 0.5
    
    else:

        wind = unrm >= 0.
        hmid = hdel[jcel]
        hmid[wind] = hdel[icel[wind]]

    divh = trsk.cell_flux_sums * (unrm * hmid)

    return divh / mesh.cell.area


def check_conserve(cnfg, mesh, unrm, hdel, 
                   zbot):

#-- calc. energy, etc, see Ringler et al: A unified approach 
#-- to energy conservation and potential vorticity dynamics 
#-- for arbitrarily structured C-grids, J. Comp. Phys.: 229 
#-- (9), 3065-3090, 2010

    hmid = hdel[mesh.edge.cell[:, 0] - 1] + \
           hdel[mesh.edge.cell[:, 1] - 1]
    hmid = hmid * 0.5

    pe = cnfg.grav * (+0.5 * hdel - zbot) * \
         mesh.cell.area * hdel

    ke = +0.5 * unrm ** 2 * hmid * \
         mesh.edge.vlen * mesh.edge.clen

    print(np.sum(ke) + np.sum(pe))

    return


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--mpas-file", dest="mpas_file", type=str,
        required=True, help="Path to user MPAS file.")

    args = parser.parse_args()

    main(name=args.mpas_file)
