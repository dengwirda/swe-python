
import numpy as np
from scipy.sparse import csr_matrix, spdiags
from mat import inv_3x3


def trsk_mats(mesh):
    """
    TRSK-MATS: returns various "TRSK-style" numerical
    operators as sparse matrices:

    CELL-FLUX-SUMS: div. integration (* area)
    CELL-KITE-SUMS: dual-to-cell remapping
    CELL-WING-SUMS: edge-to-cell remapping
    CELL-EDGE-SUMS: edge-to-cell summation
    CELL-VERT-SUMS: vert-to-cell summation
    CELL-CURL-SUMS: curl integration (* area)
    CELL-DEL2-SUMS: cell del-squared (* area)

    EDGE-STUB-SUMS: dual-to-edge remapping
    EDGE-WING-SUMS: cell-to-edge remapping
    EDGE-VERT-SUMS: vert-to-edge summation
    EDGE-CELL-SUMS: cell-to-edge summation
    EDGE-GRAD-NORM: edge gradient (normal)
    EDGE-GRAD-PERP: edge gradient (perpendicular)

    DUAL-CURL-SUMS: curl integration (* area)
    DUAL-KITE-SUMS: cell-to-dual remapping
    DUAL-STUB-SUMS: edge-to-dual remapping
    DUAL-CELL-SUMS: cell-to-dual summation
    DUAL-EDGE-SUMS: edge-to-dual summation
    DUAL-DEL2-SUMS: dual del-squared (* area)

    EDGE-FLUX-PERP: reconstruct u (perpendicular)
    EDGE-LSQR-PERP: reconstruct u (perpendicular)

    (from norm. components)
    CELL-LSQR-XNRM: reconstruct U (cartesian)
    CELL-LSQR-YNRM: reconstruct Y (cartesian)
    CELL-LSQR-ZNRM: reconstruct Z (cartesian)

    (from norm. components)
    DUAL-LSQR-XNRM: reconstruct U (cartesian)
    DUAL-LSQR-YNRM: reconstruct Y (cartesian)
    DUAL-LSQR-ZNRM: reconstruct Z (cartesian)

    (from perp. components)
    DUAL-LSQR-XPRP: reconstruct U (cartesian)
    DUAL-LSQR-YPRP: reconstruct Y (cartesian)
    DUAL-LSQR-ZPRP: reconstruct Z (cartesian)

    """
    # Authors: Darren Engwirda

    class base: pass

    trsk = base()
    trsk.cell_flux_sums = cell_flux_sums(mesh)
    trsk.cell_kite_sums = cell_kite_sums(mesh)
    trsk.cell_wing_sums = cell_wing_sums(mesh)
    trsk.cell_edge_sums = cell_edge_sums(mesh)
    trsk.cell_vert_sums = cell_vert_sums(mesh)
    trsk.cell_curl_sums = cell_curl_sums(mesh)

    trsk.edge_stub_sums = edge_stub_sums(mesh)
    trsk.edge_wing_sums = edge_wing_sums(mesh)
    trsk.edge_vert_sums = edge_vert_sums(mesh)
    trsk.edge_cell_sums = edge_cell_sums(mesh)
    trsk.edge_grad_norm = edge_grad_norm(mesh)
    trsk.edge_grad_perp = edge_grad_perp(mesh)

    trsk.cell_del2_sums = trsk.cell_flux_sums \
                        * trsk.edge_grad_norm

    trsk.dual_curl_sums = dual_curl_sums(mesh)
    trsk.dual_kite_sums = dual_kite_sums(mesh)
    trsk.dual_stub_sums = dual_stub_sums(mesh)
    trsk.dual_cell_sums = dual_cell_sums(mesh)
    trsk.dual_edge_sums = dual_edge_sums(mesh)

    trsk.dual_del2_sums = dual_del2_sums(mesh)

    trsk.cell_lsqr_xnrm, \
    trsk.cell_lsqr_ynrm, \
    trsk.cell_lsqr_znrm = cell_lsqr_fxyz(mesh)
    
    trsk.dual_lsqr_xnrm, \
    trsk.dual_lsqr_ynrm, \
    trsk.dual_lsqr_znrm, \
    trsk.dual_lsqr_xprp, \
    trsk.dual_lsqr_yprp, \
    trsk.dual_lsqr_zprp = dual_lsqr_fxyz(mesh)
    
    trsk.edge_flux_perp = edge_flux_perp(mesh)


    # ensure remapping is always at worst dissipative
    # due to floating-point round-off!
    # this modifies the mesh data-structure in-place.
    mesh.vert.area = np.maximum(
        trsk.dual_kite_sums * np.ones(mesh.cell.size),
        trsk.dual_stub_sums * np.ones(mesh.edge.size)
    )

    mesh.edge.area = np.maximum(
        trsk.edge_wing_sums * np.ones(mesh.cell.size),
        trsk.edge_stub_sums * np.ones(mesh.vert.size)
    )

    mesh.cell.area = np.maximum(
        trsk.cell_wing_sums * np.ones(mesh.edge.size),
        trsk.cell_kite_sums * np.ones(mesh.vert.size)
    )


    # build LSQR-PERP from edge-wise reconstructions
    area = spdiags(1.0 / mesh.edge.area, 0, 
                   mesh.edge.size, mesh.edge.size)

    xhat = (
        mesh.vert.xpos[mesh.edge.vert[:, 1] - 1] -
        mesh.vert.xpos[mesh.edge.vert[:, 0] - 1]
    )
    yhat = (
        mesh.vert.ypos[mesh.edge.vert[:, 1] - 1] -
        mesh.vert.ypos[mesh.edge.vert[:, 0] - 1]
    )
    zhat = (
        mesh.vert.zpos[mesh.edge.vert[:, 1] - 1] -
        mesh.vert.zpos[mesh.edge.vert[:, 0] - 1]
    )

    lhat = xhat ** 2 + yhat ** 2 + zhat ** 2
    lhat = np.sqrt(lhat)

    xhat = spdiags(xhat / lhat, 0, 
                   mesh.edge.size, mesh.edge.size)
    yhat = spdiags(yhat / lhat, 0, 
                   mesh.edge.size, mesh.edge.size)
    zhat = spdiags(zhat / lhat, 0, 
                   mesh.edge.size, mesh.edge.size)

    trsk.edge_lsqr_perp = -1. * area * (
        xhat * trsk.edge_wing_sums * 
               trsk.cell_lsqr_xnrm +
        yhat * trsk.edge_wing_sums *
               trsk.cell_lsqr_ynrm +
        zhat * trsk.edge_wing_sums *
               trsk.cell_lsqr_znrm )

    # build LSQR-GRAD for apvm-style reconstructions
    xhat = (
        mesh.cell.xpos[mesh.edge.cell[:, 1] - 1] -
        mesh.cell.xpos[mesh.edge.cell[:, 0] - 1]
    )
    yhat = (
        mesh.cell.ypos[mesh.edge.cell[:, 1] - 1] -
        mesh.cell.ypos[mesh.edge.cell[:, 0] - 1]
    )
    zhat = (
        mesh.cell.zpos[mesh.edge.cell[:, 1] - 1] -
        mesh.cell.zpos[mesh.edge.cell[:, 0] - 1]
    )

    lhat = xhat ** 2 + yhat ** 2 + zhat ** 2
    lhat = np.sqrt(lhat)

    xhat = spdiags(xhat / lhat, 0, 
                   mesh.edge.size, mesh.edge.size)
    yhat = spdiags(yhat / lhat, 0, 
                   mesh.edge.size, mesh.edge.size)
    zhat = spdiags(zhat / lhat, 0, 
                   mesh.edge.size, mesh.edge.size)

    trsk.apvm_grad_norm = -1. * area * (
        xhat * trsk.edge_stub_sums * 
               trsk.dual_lsqr_xprp * 
               trsk.edge_grad_perp +
        yhat * trsk.edge_stub_sums * 
               trsk.dual_lsqr_yprp * 
               trsk.edge_grad_perp +
        zhat * trsk.edge_stub_sums * 
               trsk.dual_lsqr_zprp * 
               trsk.edge_grad_perp )

    trsk.apvm_grad_perp = trsk.edge_grad_perp

    return trsk


def cell_edge_sign(mesh):

    sign = np.zeros((mesh.cell.size, 
        np.max(mesh.cell.topo)), dtype=np.int32)

    for edge in range(np.max(mesh.cell.topo)):

        mask = mesh.cell.topo > edge

        cidx = np.argwhere(mask).ravel()

        vidx = mesh.cell.vert[mask, edge] - 1
        eidx = mesh.cell.edge[mask, edge] - 1

        v1st = mesh.edge.vert[eidx, 0] - 1

        okay = vidx != v1st
        flip = vidx == v1st

        sign[cidx[okay], edge] = +1
        sign[cidx[flip], edge] = -1

    return sign


def cell_flux_sums(mesh):

#-- CELL-FLUX-SUMS: returns SUM(l_e * F_e) via sparse matrix
#-- operator OP. Use DIV(F) = OP * F, where F is a vector of
#-- (signed) fluxes for all edges in the mesh.

    xvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for edge in range(np.max(mesh.cell.topo)):

        mask = mesh.cell.topo > edge

        cidx = np.argwhere(mask).ravel()

        vidx = mesh.cell.vert[mask, edge] - 1
        eidx = mesh.cell.edge[mask, edge] - 1

        vlen = mesh.edge.vlen[eidx]

        v1st = mesh.edge.vert[eidx, 0] - 1

        okay = vidx != v1st
        flip = vidx == v1st

        ivec = np.hstack((
            ivec, +cidx[flip], cidx[okay]))
        jvec = np.hstack((
            jvec, +eidx[flip], eidx[okay]))
        xvec = np.hstack((
            xvec, -vlen[flip], vlen[okay]))

    return csr_matrix((xvec, (ivec, jvec)))


def cell_curl_sums(mesh):

#-- CELL-CURL-SUMS: returns SUM(f_e * P_e) via sparse matrix
#-- operator OP. Use CURL(P) = OP * P where P is a vector of
#-- (perpendicular) fluxes for edges in the mesh.

    xvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for edge in range(np.max(mesh.cell.topo)):

        mask = mesh.cell.topo > edge

        cidx = np.argwhere(mask).ravel()

        vidx = mesh.cell.vert[mask, edge] - 1
        eidx = mesh.cell.edge[mask, edge] - 1

        # to ensure curl(grad(h)) = 0
        clen = mesh.edge.clen[eidx]
        area = mesh.edge.area[eidx] * +2.

        scal = area / clen

        v1st = mesh.edge.vert[eidx, 0] - 1

        okay = vidx != v1st
        flip = vidx == v1st

        ivec = np.hstack((
            ivec, +cidx[flip], cidx[okay]))
        jvec = np.hstack((
            jvec, +eidx[flip], eidx[okay]))
        xvec = np.hstack((
            xvec, -scal[flip], scal[okay]))

        """
        vlen = mesh.edge.vlen[eidx]

        v1st = mesh.edge.vert[eidx, 0] - 1

        okay = vidx != v1st
        flip = vidx == v1st

        ivec = np.hstack((
            ivec, +cidx[flip], cidx[okay]))
        jvec = np.hstack((
            jvec, +eidx[flip], eidx[okay]))
        xvec = np.hstack((
            xvec, -vlen[flip], vlen[okay]))
        """

    return csr_matrix((xvec, (ivec, jvec)))


def cell_kite_sums(mesh):

    return dual_kite_sums(mesh).transpose(copy=True).tocsr()


def cell_wing_sums(mesh):

    return edge_wing_sums(mesh).transpose(copy=True).tocsr()


def dual_stub_sums(mesh):

    return edge_stub_sums(mesh).transpose(copy=True).tocsr()


def cell_edge_sums(mesh):

    xvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for edge in range(np.max(mesh.cell.topo)):

        mask = mesh.cell.topo > edge

        cidx = np.argwhere(mask).ravel()

        eidx = mesh.cell.edge[mask, edge] - 1

        ivec = np.hstack((ivec, cidx))
        jvec = np.hstack((jvec, eidx))
        xvec = np.hstack((
            xvec, np.ones(eidx.size, dtype=np.float64)))

    return csr_matrix((xvec, (ivec, jvec)))


def cell_vert_sums(mesh):

    xvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for vert in range(np.max(mesh.cell.topo)):

        mask = mesh.cell.topo > vert

        cidx = np.argwhere(mask).ravel()

        vidx = mesh.cell.vert[mask, vert] - 1

        ivec = np.hstack((ivec, cidx))
        jvec = np.hstack((jvec, vidx))
        xvec = np.hstack((
            xvec, np.ones(vidx.size, dtype=np.float64)))

    return csr_matrix((xvec, (ivec, jvec)))


def edge_stub_sums(mesh):

    xvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for vert in range(2):

        eidx = np.arange(0, mesh.edge.size)

        vidx = mesh.edge.vert[:, vert] - 1

        stub = mesh.edge.stub[:, vert]

        ivec = np.hstack((ivec, eidx))
        jvec = np.hstack((jvec, vidx))
        xvec = np.hstack((xvec, stub))

    return csr_matrix((xvec, (ivec, jvec)))


def edge_wing_sums(mesh):

    xvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for cell in range(2):

        eidx = np.arange(0, mesh.edge.size)

        cidx = mesh.edge.cell[:, cell] - 1

        wing = mesh.edge.wing[:, cell]

        ivec = np.hstack((ivec, eidx))
        jvec = np.hstack((jvec, cidx))
        xvec = np.hstack((xvec, wing))

    return csr_matrix((xvec, (ivec, jvec)))


def edge_vert_sums(mesh):

    xvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for vert in range(2):

        eidx = np.arange(0, mesh.edge.size)

        vidx = mesh.edge.vert[:, vert] - 1

        ivec = np.hstack((ivec, eidx))
        jvec = np.hstack((jvec, vidx))
        xvec = np.hstack((
            xvec, np.ones(vidx.size, dtype=np.float64)))

    return csr_matrix((xvec, (ivec, jvec)))


def edge_cell_sums(mesh):

    xvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for cell in range(2):

        eidx = np.arange(0, mesh.edge.size)

        cidx = mesh.edge.cell[:, cell] - 1

        ivec = np.hstack((ivec, eidx))
        jvec = np.hstack((jvec, cidx))
        xvec = np.hstack((
            xvec, np.ones(cidx.size, dtype=np.float64)))

    return csr_matrix((xvec, (ivec, jvec)))


def edge_grad_norm(mesh):

#-- EDGE-GRAD-NORM: returns (Q(j)-Q(i))/lij as sparse matrix
#-- operator OP. Use GRAD(Q) = OP * Q where Q is a vector of
#-- cell-centred scalars for all cells in the mesh.

    icel = mesh.edge.cell[:, 0] - 1
    jcel = mesh.edge.cell[:, 1] - 1

    eidx = np.arange(+0, mesh.edge.size)

    # does ensuring grad = -1.0 * div ^ T help?
    vlen = mesh.edge.vlen
    area = mesh.edge.area * +2.

    ivec = np.concatenate((eidx, eidx))
    jvec = np.concatenate((icel, jcel))
    xvec = np.concatenate(
        (-1. * vlen / area, +1. * vlen / area))

    """
    clen = mesh.edge.clen

    ivec = np.concatenate((eidx, eidx))
    jvec = np.concatenate((icel, jcel))
    xvec = np.concatenate(
        (-1.E+0 / clen, +1.E+0 / clen))
    """
    
    return csr_matrix((xvec, (ivec, jvec)))


def edge_grad_perp(mesh):

#-- EDGE-GRAD-PERP: returns (V(j)-V(i))/eij as sparse matrix
#-- operator OP. Use GRAD(V) = OP * V where V is a vector of
#-- node-centred scalars for all nodes in the mesh.

    inod = mesh.edge.vert[:, 0] - 1
    jnod = mesh.edge.vert[:, 1] - 1

    eidx = np.arange(+0, mesh.edge.size)

    clen = mesh.edge.clen
    area = mesh.edge.area * +2.

    ivec = np.concatenate((eidx, eidx))
    jvec = np.concatenate((inod, jnod))
    xvec = np.concatenate(
        (-1. * clen / area, +1. * clen / area))

    """
    vlen = mesh.edge.vlen

    ivec = np.concatenate((eidx, eidx))
    jvec = np.concatenate((inod, jnod))
    xvec = np.concatenate(
        (-1.E+0 / vlen, +1.E+0 / vlen))
    """
    
    return csr_matrix((xvec, (ivec, jvec)))


def edge_flux_perp(mesh):

#-- EDGE-FLUX-PERP: returns f_perp, via the TRSK-type scheme
#-- for edges sandwiched between cells.
#-- Use f_perp = OP * f_nrm to reconstruct the perpendicular
#-- component of a vector field F.

    xvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for edge in range(np.max(mesh.edge.topo)):

        mask = mesh.edge.topo > edge

        eidx = np.argwhere(mask).ravel()

        edsh = mesh.edge.edge[mask, edge] - 1

        wmul = mesh.edge.wmul[mask, edge]

        ivec = np.hstack((ivec, eidx))
        jvec = np.hstack((jvec, edsh))
        xvec = np.hstack((xvec, wmul))

    return csr_matrix((xvec, (ivec, jvec)))


def dual_edge_sign(mesh):

    sign = np.zeros(
        (mesh.vert.size, 3), dtype=np.int32)

    for edge in range(3):

        vidx = np.arange(0, mesh.vert.size)

        eidx = mesh.vert.edge[:, edge] - 1
        cidx = mesh.vert.cell[:, edge] - 1

        c1st = mesh.edge.cell[eidx, 0] - 1

        okay = cidx != c1st
        flip = cidx == c1st

        sign[okay, edge] = +1
        sign[flip, edge] = -1

    return sign


def dual_curl_sums(mesh):

#-- DUAL-CURL-SUMS: returns SUM(lij * F_e) via sparse matrix
#-- operator OP. Use CURL(F) = OP * F where F is a vector of
#-- (signed) fluxes for all edges in the mesh.

    xvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for edge in range(3):

        vidx = np.arange(0, mesh.vert.size)

        eidx = mesh.vert.edge[:, edge] - 1
        cidx = mesh.vert.cell[:, edge] - 1

        # to ensure curl(grad(h)) = 0
        vlen = mesh.edge.vlen[eidx]
        area = mesh.edge.area[eidx] * +2.

        scal = area / vlen

        c1st = mesh.edge.cell[eidx, 0] - 1

        okay = cidx != c1st
        flip = cidx == c1st

        ivec = np.hstack((
            ivec, +vidx[flip], vidx[okay]))
        jvec = np.hstack((
            jvec, +eidx[flip], eidx[okay]))
        xvec = np.hstack((
            xvec, -scal[flip], scal[okay]))

        """
        clen = mesh.edge.clen[eidx]

        c1st = mesh.edge.cell[eidx, 0] - 1

        okay = cidx != c1st
        flip = cidx == c1st

        ivec = np.hstack((
            ivec, +vidx[flip], vidx[okay]))
        jvec = np.hstack((
            jvec, +eidx[flip], eidx[okay]))
        xvec = np.hstack((
            xvec, -clen[flip], clen[okay]))
        """

    return csr_matrix((xvec, (ivec, jvec)))


def dual_kite_sums(mesh):

    xvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for cell in range(3):

        vidx = np.arange(0, mesh.vert.size)

        cidx = mesh.vert.cell[:, cell] - 1

        kite = mesh.vert.kite[:, cell]

        ivec = np.hstack((ivec, vidx))
        jvec = np.hstack((jvec, cidx))
        xvec = np.hstack((xvec, kite))

    return csr_matrix((xvec, (ivec, jvec)))


def dual_cell_sums(mesh):

    xvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for cell in range(3):

        vidx = np.arange(0, mesh.vert.size)

        cidx = mesh.vert.cell[:, cell] - 1

        unit = np.ones(
            mesh.vert.size, dtype=np.float64)

        ivec = np.hstack((ivec, vidx))
        jvec = np.hstack((jvec, cidx))
        xvec = np.hstack((xvec, unit))

    return csr_matrix((xvec, (ivec, jvec)))


def dual_edge_sums(mesh):

    xvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for edge in range(3):

        vidx = np.arange(0, mesh.vert.size)

        eidx = mesh.vert.edge[:, edge] - 1

        unit = np.ones(
            mesh.vert.size, dtype=np.float64)

        ivec = np.hstack((ivec, vidx))
        jvec = np.hstack((jvec, eidx))
        xvec = np.hstack((xvec, unit))

    return csr_matrix((xvec, (ivec, jvec)))


def dual_del2_sums(mesh):

    xvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for edge in range(3):

        vidx = np.arange(0, mesh.vert.size)

        eidx = mesh.vert.edge[:, edge] - 1

        aidx = mesh.edge.vert[eidx, 0] - 1
        bidx = mesh.edge.vert[eidx, 1] - 1

        same = aidx != vidx
        next = aidx == vidx

        vals = mesh.edge.clen[eidx] / \
               mesh.edge.vlen[eidx]

        ivec = np.hstack((ivec, +vidx))
        jvec = np.hstack((jvec, +vidx))
        xvec = np.hstack((xvec, -vals))

        ivec = np.hstack((
            ivec, vidx[same], vidx[next]))
        jvec = np.hstack((
            jvec, aidx[same], bidx[next]))
        xvec = np.hstack((
            xvec, vals[same], vals[next]))

    return csr_matrix((xvec, (ivec, jvec)))


def dual_lsqr_fxyz(mesh):

    RSPH = mesh.rsph

    ndir = np.vstack((
        mesh.cell.xpos[mesh.edge.cell[:, 1] - 1] - 
        mesh.cell.xpos[mesh.edge.cell[:, 0] - 1],
        mesh.cell.ypos[mesh.edge.cell[:, 1] - 1] - 
        mesh.cell.ypos[mesh.edge.cell[:, 0] - 1],
        mesh.cell.zpos[mesh.edge.cell[:, 1] - 1] - 
        mesh.cell.zpos[mesh.edge.cell[:, 0] - 1]))
    ndir = ndir.T

    nlen = np.sqrt(np.sum(
        ndir ** 2, axis=1, keepdims=True))

    pdir = np.vstack((
        mesh.vert.xpos[mesh.edge.vert[:, 1] - 1] - 
        mesh.vert.xpos[mesh.edge.vert[:, 0] - 1],
        mesh.vert.ypos[mesh.edge.vert[:, 1] - 1] - 
        mesh.vert.ypos[mesh.edge.vert[:, 0] - 1],
        mesh.vert.zpos[mesh.edge.vert[:, 1] - 1] - 
        mesh.vert.zpos[mesh.edge.vert[:, 0] - 1]))
    pdir = pdir.T

    plen = np.sqrt(np.sum(
        pdir ** 2, axis=1, keepdims=True))

    ndir = ndir / nlen
    pdir = pdir / plen

    dnrm = np.vstack((
        mesh.vert.xmid, 
        mesh.vert.ymid, mesh.vert.zmid)).T
    
    dnrm = dnrm / RSPH
    
    Tmat = np.zeros(
        (4, 3, mesh.vert.size), dtype=np.float64)
    Tmat[0, :, :] = \
        ndir[mesh.vert.edge[:, 0] - 1].T
    Tmat[1, :, :] = \
        ndir[mesh.vert.edge[:, 1] - 1].T
    Tmat[2, :, :] = \
        ndir[mesh.vert.edge[:, 2] - 1].T
    Tmat[3, :, :] = np.transpose(dnrm)

    Umat = np.zeros(
        (4, 3, mesh.vert.size), dtype=np.float64)
    Umat[0, :, :] = \
        pdir[mesh.vert.edge[:, 0] - 1].T
    Umat[1, :, :] = \
        pdir[mesh.vert.edge[:, 1] - 1].T
    Umat[2, :, :] = \
        pdir[mesh.vert.edge[:, 2] - 1].T
    Umat[3, :, :] = np.transpose(dnrm)

    matT = np.transpose(Tmat, axes=(1, 0, 2))
    matU = np.transpose(Umat, axes=(1, 0, 2))

    Rmat = np.einsum(
        "ik..., kj... -> ij...", matT, Tmat)

    Rinv, Rdet = inv_3x3(Rmat)

    Smat = np.einsum(
        "ik..., kj... -> ij...", matU, Umat)

    Sinv, Sdet = inv_3x3(Smat)

    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    xnrm = np.array([], dtype=np.float64)
    ynrm = np.array([], dtype=np.float64)
    znrm = np.array([], dtype=np.float64)
    
    xprp = np.array([], dtype=np.float64)
    yprp = np.array([], dtype=np.float64)
    zprp = np.array([], dtype=np.float64)
    
    for edge in range(3):

        vidx = np.arange(0, mesh.vert.size)

        eidx = mesh.vert.edge[:, edge] - 1

        ivec = np.hstack((ivec, vidx))
        jvec = np.hstack((jvec, eidx))

        xmul = (
            Rinv[0, 0, :] * matT[0, edge, :] +
            Rinv[0, 1, :] * matT[1, edge, :] +
            Rinv[0, 2, :] * matT[2, edge, :]
        ) / Rdet

        ymul = (
            Rinv[1, 0, :] * matT[0, edge, :] +
            Rinv[1, 1, :] * matT[1, edge, :] +
            Rinv[1, 2, :] * matT[2, edge, :]
        ) / Rdet

        zmul = (
            Rinv[2, 0, :] * matT[0, edge, :] +
            Rinv[2, 1, :] * matT[1, edge, :] +
            Rinv[2, 2, :] * matT[2, edge, :]
        ) / Rdet

        xnrm = np.hstack((xnrm, xmul))
        ynrm = np.hstack((ynrm, ymul))
        znrm = np.hstack((znrm, zmul))

        xmul = (
            Sinv[0, 0, :] * matU[0, edge, :] +
            Sinv[0, 1, :] * matU[1, edge, :] +
            Sinv[0, 2, :] * matU[2, edge, :]
        ) / Sdet

        ymul = (
            Sinv[1, 0, :] * matU[0, edge, :] +
            Sinv[1, 1, :] * matU[1, edge, :] +
            Sinv[1, 2, :] * matU[2, edge, :]
        ) / Sdet

        zmul = (
            Sinv[2, 0, :] * matU[0, edge, :] +
            Sinv[2, 1, :] * matU[1, edge, :] +
            Sinv[2, 2, :] * matU[2, edge, :]
        ) / Sdet

        xprp = np.hstack((xprp, xmul))
        yprp = np.hstack((yprp, ymul))
        zprp = np.hstack((zprp, zmul))

    return csr_matrix((xnrm, (ivec, jvec))), \
           csr_matrix((ynrm, (ivec, jvec))), \
           csr_matrix((znrm, (ivec, jvec))), \
           csr_matrix((xprp, (ivec, jvec))), \
           csr_matrix((yprp, (ivec, jvec))), \
           csr_matrix((zprp, (ivec, jvec)))


def cell_lsqr_fxyz(mesh):

    cnrm = np.vstack((
        mesh.cell.xmid,
        mesh.cell.ymid, mesh.cell.zmid)).T
    
    cnrm = cnrm / mesh.rsph
    
    Tmat = np.zeros(
        (np.max(mesh.cell.topo) + 1, 3, 
            mesh.cell.size), dtype=np.float64)

    Wmat = np.zeros(
        (np.max(mesh.cell.topo) + 1,
            np.max(mesh.cell.topo) + 1,
            mesh.cell.size), dtype=np.float64)

    sign = cell_edge_sign(mesh)

    for edge in range(np.max(mesh.cell.topo) + 1):

        Wmat[edge, edge, :] = \
            1.0 / np.sqrt(mesh.cell.area).T

    for edge in range(np.max(mesh.cell.topo) + 0):

        mask = mesh.cell.topo > edge

        eidx = mesh.cell.edge[mask, edge] - 1

        edir = np.vstack((
            mesh.cell.xpos[mask] -
            mesh.edge.xpos[eidx],
            mesh.cell.ypos[mask] -
            mesh.edge.ypos[eidx],
            mesh.cell.zpos[mask] -
            mesh.edge.zpos[eidx]))
        edir = edir.T

        elen = np.sqrt(np.sum(
            edir ** 2, axis=1, keepdims=True))

        edir = edir / elen

        Wmat[edge, edge, mask] = 1.0 / elen.T

        Tmat[edge, :, mask] = \
            (sign[mask, edge] * edir.T).T
    
    Tmat[-1, :, :] = np.transpose(cnrm)

    matT = np.transpose(Tmat, axes=(1, 0, 2))

    matW = np.einsum(
        "ik..., kj... -> ij...", matT, Wmat)

    Rmat = np.einsum(
        "ik..., kj... -> ij...", matW, Tmat)

    Rinv, Rdet = inv_3x3(Rmat)

    xvec = np.array([], dtype=np.float64)
    yvec = np.array([], dtype=np.float64)
    zvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for edge in range(np.max(mesh.cell.topo) + 0):

        mask = mesh.cell.topo > edge

        cidx = np.argwhere(mask).ravel()

        eidx = mesh.cell.edge[mask, edge] - 1

        M = mask
        xmul = (
            Rinv[0, 0, M] * matW[0, edge, M] +
            Rinv[0, 1, M] * matW[1, edge, M] +
            Rinv[0, 2, M] * matW[2, edge, M]
        ) / Rdet[M]

        ymul = (
            Rinv[1, 0, M] * matW[0, edge, M] +
            Rinv[1, 1, M] * matW[1, edge, M] +
            Rinv[1, 2, M] * matW[2, edge, M]
        ) / Rdet[M]

        zmul = (
            Rinv[2, 0, M] * matW[0, edge, M] +
            Rinv[2, 1, M] * matW[1, edge, M] +
            Rinv[2, 2, M] * matW[2, edge, M]
        ) / Rdet[M]

        ivec = np.hstack((ivec, cidx))
        jvec = np.hstack((jvec, eidx))
        xvec = np.hstack((xvec, xmul))
        yvec = np.hstack((yvec, ymul))
        zvec = np.hstack((zvec, zmul))

    return csr_matrix((xvec, (ivec, jvec))), \
           csr_matrix((yvec, (ivec, jvec))), \
           csr_matrix((zvec, (ivec, jvec)))
