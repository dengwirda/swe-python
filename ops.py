
import numpy as np

from scipy.sparse import csr_matrix


def init_mesh_sign(base, mesh):

    ivec = np.zeros(
        np.sum(mesh.cell.topo), dtype=int)
    jvec = np.zeros(
        np.sum(mesh.cell.topo), dtype=int)
    xvec = np.zeros(
        np.sum(mesh.cell.topo), dtype=int)

    next = 0

    for cell in range(mesh.cell.size):
        for edge in range(mesh.cell.topo[cell]):

            vidx = mesh.cell.vert[cell, edge] - 1
            eidx = mesh.cell.edge[cell, edge] - 1

            vone = mesh.edge.vert[eidx, 0] - 1

            if (vidx != vone):
                ivec[next] = cell
                jvec[next] = eidx
                xvec[next] = +1
            else:
                ivec[next] = cell
                jvec[next] = eidx
                xvec[next] = -1

            next = next + 1

    mesh.etoc = csr_matrix((xvec, (ivec, jvec)))

    return mesh


def eval_cell_div_(base, mesh, unrm, hdel):

    icel = mesh.edge.cell[:, 0] - 1
    jcel = mesh.edge.cell[:, 1] - 1

    if True:
 
        hmid =(hdel[icel] + hdel[jcel]) * 0.5
    
    else:

        wind = unrm >= 0.
        hmid = hdel[jcel]
        hmid[wind] = hdel[icel[wind]]

    flux = mesh.edge.vlen * unrm * hmid

    divh = mesh.etoc * flux

    return divh / mesh.cell.area 


def eval_edge_hpgf(base, mesh, zlev, hdel, plev):

    zlev[:, 0] = zlev[:, 1] + hdel

    icel = mesh.edge.cell[:, 0] - 1
    jcel = mesh.edge.cell[:, 1] - 1

    zpos = np.zeros(
        (mesh.edge.size, 2), dtype=float)

    zpos[:, 0] = (                  # surf. height at u-pos
        zlev[icel, 0] + zlev[jcel, 0]) * 0.5

    zpos[:, 1] = zpos[:, 0] - (     # use same h as div(uh)
        hdel[icel] + hdel[jcel]) * 0.5

    pone = plev[icel, 0] + (
        zlev[icel, 0] - zpos[:, 1]) * base.cnfg.grav * base.cnfg.rho0
    
    ptwo = plev[jcel, 0] + (
        zlev[jcel, 0] - zpos[:, 1]) * base.cnfg.grav * base.cnfg.rho0

    hpgf = ptwo - pone

    return hpgf / mesh.edge.clen
