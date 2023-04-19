
import time
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

    DUAL-FLUX-SUMS: div. integration (* area)
    DUAL-KITE-SUMS: cell-to-dual remapping
    DUAL-STUB-SUMS: edge-to-dual remapping
    DUAL-CELL-SUMS: cell-to-dual summation
    DUAL-EDGE-SUMS: edge-to-dual summation
    DUAL-CURL-SUMS: curl integration (* area)    
    DUAL-DEL2-SUMS: dual del-squared (* area)

    QUAD-CURL-SUMS: curl integration (* area)

    EDGE-FLUX-PERP: reconstruct v (perpendicular)
    EDGE-LSQR-PERP: reconstruct v (perpendicular)

    (from norm. components)
    DUAL-LSQR-XNRM: reconstruct U (cartesian)
    DUAL-LSQR-YNRM: reconstruct Y (cartesian)
    DUAL-LSQR-ZNRM: reconstruct Z (cartesian)

    (from perp. components)
    DUAL-LSQR-XPRP: reconstruct U (cartesian)
    DUAL-LSQR-YPRP: reconstruct Y (cartesian)
    DUAL-LSQR-ZPRP: reconstruct Z (cartesian)

    (from norm. components)
    CELL-LSQR-XNRM: reconstruct U (cartesian)
    CELL-LSQR-YNRM: reconstruct Y (cartesian)
    CELL-LSQR-ZNRM: reconstruct Z (cartesian)

    (from norm. components)
    EDGE-LSQR-XNRM: reconstruct U (cartesian)
    EDGE-LSQR-YNRM: reconstruct Y (cartesian)
    EDGE-LSQR-ZNRM: reconstruct Z (cartesian)

    (piecewise linear op's)
    EDGE-DUAL-RECO: recon. F at edge from dual
    EDGE-CELL-RECO: recon. F at edge from cell

    """
    # Authors: Darren Engwirda

    class base: pass

    ttic = time.time()

    trsk = base()
   #trsk.cell_edge_sign = cell_edge_sign(mesh)
   #trsk.dual_edge_sign = dual_edge_sign(mesh)

    trsk.cell_flux_sums = cell_flux_sums(mesh)
    trsk.cell_kite_sums = cell_kite_sums(mesh)
    trsk.cell_wing_sums = cell_wing_sums(mesh)
    trsk.cell_edge_sums = cell_edge_sums(mesh)
    trsk.cell_vert_sums = cell_vert_sums(mesh)
   #trsk.cell_curl_sums = cell_curl_sums(mesh)
    trsk.cell_curl_sums = trsk.cell_flux_sums  # equiv.

    trsk.edge_stub_sums = edge_stub_sums(mesh)
    trsk.edge_wing_sums = edge_wing_sums(mesh)
    trsk.edge_vert_sums = edge_vert_sums(mesh)
    trsk.edge_cell_sums = edge_cell_sums(mesh)
    trsk.edge_grad_norm = edge_grad_norm(mesh)
    trsk.edge_grad_perp = edge_grad_perp(mesh)

    trsk.cell_del2_sums = trsk.cell_flux_sums \
                        * trsk.edge_grad_norm

    trsk.dual_flux_sums = dual_flux_sums(mesh)
    trsk.dual_kite_sums = dual_kite_sums(mesh)
    trsk.dual_stub_sums = dual_stub_sums(mesh)
    trsk.dual_cell_sums = dual_cell_sums(mesh)
    trsk.dual_edge_sums = dual_edge_sums(mesh)
   #trsk.dual_curl_sums = dual_curl_sums(mesh)
    trsk.dual_curl_sums = trsk.dual_flux_sums  # equiv.

   #trsk.dual_del2_sums = trsk.dual_flux_sums \
   #                    * trsk.edge_grad_perp

    # take curl on rhombi, a'la Gassmann
    trsk.quad_curl_sums = trsk.edge_vert_sums \
                        * trsk.dual_curl_sums

    ttoc = time.time()
   #print("mats:", ttoc - ttic)
    
    ttic = time.time()

    # least-squares vector reconstruction operators
    trsk.dual_lsqr_xnrm, \
    trsk.dual_lsqr_ynrm, \
    trsk.dual_lsqr_znrm, \
    trsk.dual_lsqr_xprp, \
    trsk.dual_lsqr_yprp, \
    trsk.dual_lsqr_zprp = dual_lsqr_fxyz(mesh)

    trsk.cell_lsqr_xnrm, \
    trsk.cell_lsqr_ynrm, \
    trsk.cell_lsqr_znrm = cell_lsqr_fxyz(mesh)

    trsk.edge_lsqr_xnrm, \
    trsk.edge_lsqr_ynrm, \
    trsk.edge_lsqr_znrm = edge_lsqr_fxyz(mesh)

    ttoc = time.time()
   #print("lsqr:", ttoc - ttic)
    
    ttic = time.time()

    # ensure flux reconstruction operator is exactly
    # skew-symmetric. Per Ringler et al, 2010, W_prp
    # is required to be anti-symmetric to ensure
    # energetically neutral PV fluxes: W_ij = -W_ji.
    # Due to floating-point round-off!
    trsk.edge_flux_perp = edge_flux_perp(mesh)

    lmat = spdiags(
        1./mesh.edge.vlen, 
        0, mesh.edge.size, mesh.edge.size)

    dmat = spdiags(
        1.*mesh.edge.dlen, 
        0, mesh.edge.size, mesh.edge.size)

    wmat = dmat * trsk.edge_flux_perp * lmat

    wmat = 0.5 * (wmat - wmat.transpose())

    lmat = spdiags(
        1.*mesh.edge.vlen,
        0, mesh.edge.size, mesh.edge.size)

    dmat = spdiags(
        1./mesh.edge.clen, 
        0, mesh.edge.size, mesh.edge.size)

    trsk.edge_flux_perp = dmat * wmat * lmat

    ttoc = time.time()
   #print("wnrm:", ttoc - ttic)

    ttic = time.time()

    # ensure remapping is always at worst dissipative
    # due to floating-point round-off!
    # this modifies the mesh data-structure in-place.
    crhs = np.ones(
        mesh.cell.size, dtype=np.float64)
    erhs = np.ones(
        mesh.edge.size, dtype=np.float64)
    vrhs = np.ones(
        mesh.vert.size, dtype=np.float64)

    mesh.vert.area = (
        0.5 * trsk.dual_kite_sums * crhs +
        0.5 * trsk.dual_stub_sums * erhs
    )

    mesh.edge.area = (
        0.5 * trsk.edge_wing_sums * crhs +
        0.5 * trsk.edge_stub_sums * vrhs
    )

    mesh.cell.area = (
        0.5 * trsk.cell_wing_sums * erhs +
        0.5 * trsk.cell_kite_sums * vrhs
    )

    ttoc = time.time()
   #print("area:", ttoc - ttic)
    
    ttic = time.time()

    mesh.quad = base()
    mesh.quad.area = trsk.edge_vert_sums \
                   * mesh.vert.area

    # build LSQR-<OP> from edge-wise reconstructions
    trsk.edge_lsqr_perp = edge_lsqr_perp(mesh, trsk)
   #trsk.edge_lsqr_norm = edge_lsqr_norm(mesh, trsk)

    # operators for piecewise linear reconstructions
    # fe = fi + (xe - xi) * grad(f)
    trsk.edge_dual_reco = edge_dual_reco(mesh, trsk)
   #trsk.edge_cell_reco = edge_cell_reco(mesh, trsk)

    ttoc = time.time()
   #print("reco:", ttoc - ttic)
   
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

    clen = mesh.edge.clen

    ivec = np.concatenate((eidx, eidx))
    jvec = np.concatenate((icel, jcel))
    xvec = np.concatenate(
        (-1.E+0 / clen, +1.E+0 / clen))
    
    return csr_matrix((xvec, (ivec, jvec)))


def edge_grad_perp(mesh):

#-- EDGE-GRAD-PERP: returns (V(j)-V(i))/vij as sparse matrix
#-- operator OP. Use GRAD(V) = OP * V where V is a vector of
#-- node-centred scalars for all nodes in the mesh.

    inod = mesh.edge.vert[:, 0] - 1
    jnod = mesh.edge.vert[:, 1] - 1

    eidx = np.arange(+0, mesh.edge.size)

    vlen = mesh.edge.vlen

    ivec = np.concatenate((eidx, eidx))
    jvec = np.concatenate((inod, jnod))
    xvec = np.concatenate(
        (-1.E+0 / vlen, +1.E+0 / vlen))
    
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


def dual_flux_sums(mesh):

    xvec = np.array([], dtype=np.float64)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    for edge in range(3):

        vidx = np.arange(0, mesh.vert.size)

        eidx = mesh.vert.edge[:, edge] - 1
        cidx = mesh.vert.cell[:, edge] - 1

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
        
    return csr_matrix((xvec, (ivec, jvec)))


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


def dual_lsqr_mats(mesh):

#-- lsqr matrices for nrm. + prp. cell reconstructions

    ndir = np.vstack((
        mesh.edge.xnrm,
        mesh.edge.ynrm, mesh.edge.znrm)).T
    
    pdir = np.vstack((
        mesh.edge.xprp,
        mesh.edge.yprp, mesh.edge.zprp)).T

    dnrm = np.vstack((
        mesh.vert.xmid, 
        mesh.vert.ymid, mesh.vert.zmid)).T
    
    dnrm = dnrm / mesh.rsph
    
    Amat = np.zeros(
        (4, 3, mesh.vert.size), dtype=np.float64)
    Amat[0, :, :] = \
        ndir[mesh.vert.edge[:, 0] - 1].T
    Amat[1, :, :] = \
        ndir[mesh.vert.edge[:, 1] - 1].T
    Amat[2, :, :] = \
        ndir[mesh.vert.edge[:, 2] - 1].T
    Amat[3, :, :] = np.transpose(dnrm)

    Bmat = np.zeros(
        (4, 3, mesh.vert.size), dtype=np.float64)
    Bmat[0, :, :] = \
        pdir[mesh.vert.edge[:, 0] - 1].T
    Bmat[1, :, :] = \
        pdir[mesh.vert.edge[:, 1] - 1].T
    Bmat[2, :, :] = \
        pdir[mesh.vert.edge[:, 2] - 1].T
    Bmat[3, :, :] = np.transpose(dnrm)

    matA = np.transpose(Amat, axes=(1, 0, 2))
    matB = np.transpose(Bmat, axes=(1, 0, 2))

    Rmat = np.einsum(
        "ik..., kj... -> ij...", matA, Amat)
    Smat = np.einsum(
        "ik..., kj... -> ij...", matB, Bmat)

    Rinv, Rdet = inv_3x3(Rmat)
    Sinv, Sdet = inv_3x3(Smat)
    
    return Rinv, Rdet, matA, Sinv, Sdet, matB


def dual_lsqr_fxyz(mesh):

#-- dual reconstruction via "small" dual-based stencil

    Rinv, Rdet, matR, \
    Sinv, Sdet, matS = dual_lsqr_mats(mesh)
    
    xnrm = []; ynrm = []; znrm = []
    xprp = []; yprp = []; zprp = []
    ivec = []; jvec = []

    for edge in range(3):

        vidx = np.arange(0, mesh.vert.size)

        eidx = mesh.vert.edge[:, edge] - 1

        mask = eidx >= 0

        ivec.append(vidx); jvec.append(eidx)

        xmul = Rinv[0, 0, :] * matR[0, edge, :]
        xmul+= Rinv[0, 1, :] * matR[1, edge, :]
        xmul+= Rinv[0, 2, :] * matR[2, edge, :]
        xmul/= Rdet
        
        ymul = Rinv[1, 0, :] * matR[0, edge, :]
        ymul+= Rinv[1, 1, :] * matR[1, edge, :]
        ymul+= Rinv[1, 2, :] * matR[2, edge, :]
        ymul/= Rdet
        
        zmul = Rinv[2, 0, :] * matR[0, edge, :]
        zmul+= Rinv[2, 1, :] * matR[1, edge, :]
        zmul+= Rinv[2, 2, :] * matR[2, edge, :]
        zmul/= Rdet

        xnrm.append(xmul[mask])
        ynrm.append(ymul[mask])
        znrm.append(zmul[mask])
        
        xmul = Sinv[0, 0, :] * matS[0, edge, :]
        xmul+= Sinv[0, 1, :] * matS[1, edge, :]
        xmul+= Sinv[0, 2, :] * matS[2, edge, :]
        xmul/= Sdet
        
        ymul = Sinv[1, 0, :] * matS[0, edge, :]
        ymul+= Sinv[1, 1, :] * matS[1, edge, :]
        ymul+= Sinv[1, 2, :] * matS[2, edge, :]
        ymul/= Sdet
        
        zmul = Sinv[2, 0, :] * matS[0, edge, :]
        zmul+= Sinv[2, 1, :] * matS[1, edge, :]
        zmul+= Sinv[2, 2, :] * matS[2, edge, :]
        zmul/= Sdet

        xprp.append(xmul[mask])
        yprp.append(ymul[mask])
        zprp.append(zmul[mask])
        
    ivec = np.concatenate(ivec)
    jvec = np.concatenate(jvec)
    
    xnrm = np.concatenate(xnrm)
    ynrm = np.concatenate(ynrm)
    znrm = np.concatenate(znrm)
    xprp = np.concatenate(xprp)
    yprp = np.concatenate(yprp)
    zprp = np.concatenate(zprp)

    return csr_matrix((xnrm, (ivec, jvec))), \
           csr_matrix((ynrm, (ivec, jvec))), \
           csr_matrix((znrm, (ivec, jvec))), \
           csr_matrix((xprp, (ivec, jvec))), \
           csr_matrix((yprp, (ivec, jvec))), \
           csr_matrix((zprp, (ivec, jvec)))


def cell_lsqr_mats(mesh):

#-- lsqr matrices for nrm. + prp. cell reconstructions

    edir = np.vstack((
        mesh.edge.xnrm,
        mesh.edge.ynrm, mesh.edge.znrm)).T

    cnrm = np.vstack((
        mesh.cell.xmid,
        mesh.cell.ymid, mesh.cell.zmid)).T
    
    cnrm = cnrm / mesh.rsph
       
    Amat = np.zeros(
        (np.max(mesh.cell.topo) + 1, 3, 
         mesh.cell.size), dtype=np.float64)

    Wmat = np.zeros(
        (np.max(mesh.cell.topo) + 1,
         np.max(mesh.cell.topo) + 1,
         mesh.cell.size), dtype=np.float64)

    for edge in range(np.max(mesh.cell.topo) + 1):

        Wmat[edge, edge, :] = mesh.cell.area.T

    for edge in range(np.max(mesh.cell.topo) + 0):

        mask = mesh.cell.topo > edge

        eidx = mesh.cell.edge[mask, edge] - 1

        area = mesh.edge.area[eidx].T

        Wmat[edge, edge, mask] = area

        Amat[edge,    :, mask] = edir[eidx]
    
    Amat[-1, :, :] = np.transpose(cnrm)

    matA = np.transpose(Amat, axes=(1, 0, 2))

    matA = np.einsum(
        "ik..., kj... -> ij...", matA, Wmat)

    Rmat = np.einsum(
        "ik..., kj... -> ij...", matA, Amat)

    Rinv, Rdet = inv_3x3(Rmat)
    
    return Rinv, Rdet, matA


def cell_lsqr_fxyz(mesh):

#-- cell reconstruction via "large" cell-based stencil

    Rinv, Rdet, matR = cell_lsqr_mats(mesh)
    
    xnrm = []; ynrm = []; znrm = []
    xprp = []; yprp = []; zprp = []
    ivec = []; jvec = []

    for edge in range(np.max(mesh.cell.topo) + 0):

        mask = mesh.cell.topo > edge

        cidx = np.argwhere(mask).ravel()

        eidx = mesh.cell.edge[mask, edge] - 1

        ivec.append(cidx); jvec.append(eidx)

        xmul = Rinv[0, 0, :] * matR[0, edge, :]
        xmul+= Rinv[0, 1, :] * matR[1, edge, :]
        xmul+= Rinv[0, 2, :] * matR[2, edge, :]
        xmul/= Rdet
        
        ymul = Rinv[1, 0, :] * matR[0, edge, :]
        ymul+= Rinv[1, 1, :] * matR[1, edge, :]
        ymul+= Rinv[1, 2, :] * matR[2, edge, :]
        ymul/= Rdet
        
        zmul = Rinv[2, 0, :] * matR[0, edge, :]
        zmul+= Rinv[2, 1, :] * matR[1, edge, :]
        zmul+= Rinv[2, 2, :] * matR[2, edge, :]
        zmul/= Rdet

        xnrm.append(xmul[mask])
        ynrm.append(ymul[mask])
        znrm.append(zmul[mask])
        
    ivec = np.concatenate(ivec)
    jvec = np.concatenate(jvec)
    
    xnrm = np.concatenate(xnrm)
    ynrm = np.concatenate(ynrm)
    znrm = np.concatenate(znrm)

    return csr_matrix((xnrm, (ivec, jvec))), \
           csr_matrix((ynrm, (ivec, jvec))), \
           csr_matrix((znrm, (ivec, jvec)))


def edge_lsqr_mats(mesh):

#-- lsqr matrices for nrm. + prp. edge reconstructions

    ndir = np.vstack((
        mesh.edge.xnrm,
        mesh.edge.ynrm, mesh.edge.znrm)).T
    
    pdir = np.vstack((
        mesh.edge.xprp,
        mesh.edge.yprp, mesh.edge.zprp)).T

    enrm = np.vstack((
        mesh.edge.xpos,
        mesh.edge.ypos, mesh.edge.zpos)).T
    
    enrm = enrm / mesh.rsph

    Amat = np.zeros(
        (np.max(mesh.edge.topo) + 1, 3,
         mesh.edge.size), dtype=np.float64)

    Bmat = np.zeros(
        (np.max(mesh.edge.topo) + 1, 3,
         mesh.edge.size), dtype=np.float64)

    Wmat = np.zeros(
        (np.max(mesh.edge.topo) + 1, 
         np.max(mesh.edge.topo) + 1, 
         mesh.edge.size), dtype=np.float64)

    for edge in range(np.max(mesh.edge.topo) + 1):

        Wmat[edge, edge, :] = mesh.edge.area.T

    for edge in range(np.max(mesh.edge.topo) + 0):

        mask = mesh.edge.topo > edge

        eidx = mesh.edge.edge[mask, edge] - 1

        area = mesh.edge.area[eidx].T

        Wmat[edge, edge, mask] = area

        Amat[edge,    :, mask] = ndir[eidx]
        Bmat[edge,    :, mask] = pdir[eidx]
    
    Amat[-1, :, :] = np.transpose(enrm)
    Bmat[-1, :, :] = np.transpose(enrm)
    
    matA = np.transpose(Amat, axes=(1, 0, 2))
    matB = np.transpose(Bmat, axes=(1, 0, 2))

    matA = np.einsum(
        "ik..., kj... -> ij...", matA, Wmat)
    Rmat = np.einsum(
        "ik..., kj... -> ij...", matA, Amat)

    matB = np.einsum(
        "ik..., kj... -> ij...", matB, Wmat)
    Smat = np.einsum(
        "ik..., kj... -> ij...", matB, Bmat)

    Rinv, Rdet = inv_3x3(Rmat)
    Sinv, Sdet = inv_3x3(Smat)
    
    return Rinv, Rdet, matA, Sinv, Sdet, matB


def edge_lsqr_fxyz(mesh):

#-- edge reconstruction via "large" cell-based stencil

    Rinv, Rdet, matR, \
    Sinv, Sdet, matS = edge_lsqr_mats(mesh)
    
    xnrm = []; ynrm = []; znrm = []
    xprp = []; yprp = []; zprp = []
    ivec = []; jvec = []
    
    for edge in range(np.max(mesh.edge.topo) + 0):

        mask = mesh.edge.topo > edge

        enum = np.argwhere(mask).ravel()

        eidx = mesh.edge.edge[mask, edge] - 1

        ivec.append(enum); jvec.append(eidx)

        xmul = Rinv[0, 0, :] * matR[0, edge, :]
        xmul+= Rinv[0, 1, :] * matR[1, edge, :]
        xmul+= Rinv[0, 2, :] * matR[2, edge, :]
        xmul/= Rdet
        
        ymul = Rinv[1, 0, :] * matR[0, edge, :]
        ymul+= Rinv[1, 1, :] * matR[1, edge, :]
        ymul+= Rinv[1, 2, :] * matR[2, edge, :]
        ymul/= Rdet
        
        zmul = Rinv[2, 0, :] * matR[0, edge, :]
        zmul+= Rinv[2, 1, :] * matR[1, edge, :]
        zmul+= Rinv[2, 2, :] * matR[2, edge, :]
        zmul/= Rdet

        xnrm.append(xmul[mask])
        ynrm.append(ymul[mask])
        znrm.append(zmul[mask])

    ivec = np.concatenate(ivec)
    jvec = np.concatenate(jvec)
    
    xnrm = np.concatenate(xnrm)
    ynrm = np.concatenate(ynrm)
    znrm = np.concatenate(znrm)

    return csr_matrix((xnrm, (ivec, jvec))), \
           csr_matrix((ynrm, (ivec, jvec))), \
           csr_matrix((znrm, (ivec, jvec)))


def edge_lsqr_perp(mesh, trsk):

    xprp = mesh.edge.xprp
    xprp = spdiags(
        xprp, 0, mesh.edge.size, mesh.edge.size)
    yprp = mesh.edge.yprp
    yprp = spdiags(
        yprp, 0, mesh.edge.size, mesh.edge.size)
    zprp = mesh.edge.zprp
    zprp = spdiags(
        zprp, 0, mesh.edge.size, mesh.edge.size)
  
    return (
        +1.000 * xprp * trsk.edge_lsqr_xnrm +
        +1.000 * yprp * trsk.edge_lsqr_ynrm +
        +1.000 * zprp * trsk.edge_lsqr_znrm 
    )
    
    
def edge_lsqr_norm(mesh, trsk):

    xnrm = mesh.edge.xnrm
    xnrm = spdiags(
        xnrm, 0, mesh.edge.size, mesh.edge.size)
    ynrm = mesh.edge.ynrm   
    ynrm = spdiags(
        ynrm, 0, mesh.edge.size, mesh.edge.size)
    znrm = mesh.edge.znrm
    znrm = spdiags(
        znrm, 0, mesh.edge.size, mesh.edge.size)

    return (
        +1.000 * xnrm * trsk.edge_lsqr_xprp +
        +1.000 * ynrm * trsk.edge_lsqr_yprp +
        +1.000 * znrm * trsk.edge_lsqr_zprp 
    )


def edge_dual_reco(mesh, trsk):

#-- EDGE-DUAL-RECO: returns .5 * (xe - xv) * grad(F) part of
#-- edge reconstruction operator,
#-- with gradients estimated using "2-ring" stencil on duals.

    vrt1 = mesh.edge.vert[:, 0] - 1
    xev1 = mesh.edge.xpos - mesh.vert.xmid[vrt1]
    yev1 = mesh.edge.ypos - mesh.vert.ymid[vrt1]
    zev1 = mesh.edge.zpos - mesh.vert.zmid[vrt1]

    vrt2 = mesh.edge.vert[:, 1] - 1
    xev2 = mesh.edge.xpos - mesh.vert.xmid[vrt2]
    yev2 = mesh.edge.ypos - mesh.vert.ymid[vrt2]
    zev2 = mesh.edge.zpos - mesh.vert.zmid[vrt2]

    eidx = np.arange(0, mesh.edge.size)

    ivec = np.hstack((eidx, eidx))
    jvec = np.hstack((vrt1, vrt2))

    xmat = csr_matrix((
        np.hstack((xev1, xev2)), (ivec, jvec)))
    
    ymat = csr_matrix((
        np.hstack((yev1, yev2)), (ivec, jvec)))
    
    zmat = csr_matrix((
        np.hstack((zev1, zev2)), (ivec, jvec)))

    return (
        +0.500 * xmat * (trsk.dual_lsqr_xprp * 
                         trsk.edge_grad_perp) +
        +0.500 * ymat * (trsk.dual_lsqr_yprp * 
                         trsk.edge_grad_perp) +
        +0.500 * zmat * (trsk.dual_lsqr_zprp * 
                         trsk.edge_grad_perp)
    )


def edge_cell_reco(mesh, trsk):

#-- EDGE-CELL-RECO: returns .5 * (xe - xc) * grad(F) part of
#-- edge reconstruction operator,
#-- with gradients estimated using "2-ring" stencil on cells.

    cel1 = mesh.edge.cell[:, 0] - 1
    xec1 = mesh.edge.xpos - mesh.cell.xmid[cel1]
    yec1 = mesh.edge.ypos - mesh.cell.ymid[cel1]
    zec1 = mesh.edge.zpos - mesh.cell.zmid[cel1]

    cel2 = mesh.edge.cell[:, 1] - 1
    xec2 = mesh.edge.xpos - mesh.cell.xmid[cel2]
    yec2 = mesh.edge.ypos - mesh.cell.ymid[cel2]
    zec2 = mesh.edge.zpos - mesh.cell.zmid[cel2]

    eidx = np.arange(0, mesh.edge.size)

    ivec = np.hstack((eidx, eidx))
    jvec = np.hstack((cel1, cel2))

    xmat = csr_matrix((
        np.hstack((xec1, xec2)), (ivec, jvec)))
    
    ymat = csr_matrix((
        np.hstack((yec1, yec2)), (ivec, jvec)))
    
    zmat = csr_matrix((
        np.hstack((zec1, zec2)), (ivec, jvec)))

    return (
        +0.500 * xmat * (trsk.cell_lsqr_xnrm * 
                         trsk.edge_grad_norm) +
        +0.500 * ymat * (trsk.cell_lsqr_ynrm * 
                         trsk.edge_grad_norm) +
        +0.500 * zmat * (trsk.cell_lsqr_znrm * 
                         trsk.edge_grad_norm)
    )
