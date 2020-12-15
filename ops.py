
import numpy as np
from scipy.sparse import csr_matrix


def trsk_mats(mesh):
    """
    TRSK-MATS: returns the various TRiSK-style numerical 
    operators as sparse matrices:

    CELL-FLUX-SUMS: div. integration (x area)
    CELL-KITE-SUMS: dual-to-cell remapping
    CELL-EDGE-SUMS: edge-to-cell summation
    CELL-CURL-SUMS: curl integration (x area)
    CELL-DEL2-SUMS: cell del-squared (x area)
    
    EDGE-KITE-SUMS: dual-to-edge remapping
    EDGE-GRAD-NORM: edge gradient (normal)
    EDGE-GRAD-PERP: edge gradient (perpendicular)
    EDGE-RECO-PERP: reconstruct u (perpendicular)

    DUAL-CURL-SUMS: curl integration (x area)
    DUAL-KITE-SUMS: cell-to-dual remapping
    DUAL-CELL-SUMS: cell-to-dual summation
    DUAL-EDGE-SUMS: edge-to-dual summation

    """
    # Authors: Darren Engwirda

    class base: pass

    trsk = base()
    trsk.cell_flux_sums = cell_flux_sums(mesh)
    trsk.cell_kite_sums = cell_kite_sums(mesh)
    trsk.cell_edge_sums = cell_edge_sums(mesh)
    trsk.cell_curl_sums = cell_curl_sums(mesh)
    
    trsk.edge_kite_sums = edge_kite_sums(mesh)
    trsk.edge_grad_norm = edge_grad_norm(mesh)
    trsk.edge_grad_perp = edge_grad_perp(mesh)
    
    trsk.cell_del2_sums = trsk.cell_flux_sums \
                        * trsk.edge_grad_norm
    
    trsk.edge_reco_perp = edge_reco_perp(mesh)

    trsk.dual_curl_sums = dual_curl_sums(mesh)
    trsk.dual_kite_sums = dual_kite_sums(mesh)
    trsk.dual_cell_sums = dual_cell_sums(mesh)

    return trsk


def cell_flux_sums(mesh):

#-- CELL-FLUX-SUMS: returns SUM(l_e * F_e) via sparse matrix
#-- operator OP. Use DIV(F) = OP * F, where F is a vector of
#-- (signed) fluxes for all edges in the mesh.

    xvec = np.array([], dtype=float)
    ivec = np.array([], dtype=int)
    jvec = np.array([], dtype=int)
    
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

    xvec = np.array([], dtype=float)
    ivec = np.array([], dtype=int)
    jvec = np.array([], dtype=int)
    
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

    return dual_kite_sums(mesh).transpose(copy=True)


def cell_edge_sums(mesh):

    xvec = np.array([], dtype=float)
    ivec = np.array([], dtype=int)
    jvec = np.array([], dtype=int)
    
    for edge in range(np.max(mesh.cell.topo)):

        mask = mesh.cell.topo > edge

        cidx = np.argwhere(mask).ravel()

        eidx = mesh.cell.edge[mask, edge] - 1

        ivec = np.hstack((ivec, cidx))
        jvec = np.hstack((jvec, eidx))
        xvec = np.hstack((
            xvec, np.ones(eidx.size, dtype=float)))

    return csr_matrix((xvec, (ivec, jvec)))


def edge_kite_sums(mesh):

    xvec = np.array([], dtype=float)
    ivec = np.array([], dtype=int)
    jvec = np.array([], dtype=int)

    for vert in range(2):

        eidx = np.arange(0, mesh.edge.size)

        vidx = mesh.edge.vert[:, vert] - 1

        kite = mesh.edge.kite[:, vert]

        ivec = np.hstack((ivec, eidx))
        jvec = np.hstack((jvec, vidx))
        xvec = np.hstack((xvec, kite))
    
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

#-- EDGE-GRAD-PERP: returns (V(j)-V(i))/eij as sparse matrix
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


def edge_reco_perp(mesh):

#-- EDGE-RECO-PERP: returns f_perp, via the TRSK-type scheme
#-- for edges sandwiched between cells. 
#-- Use f_perp = OP * f_nrm to reconstruct the perpendicular
#-- component of a vector field F.

    xvec = np.array([], dtype=float)
    ivec = np.array([], dtype=int)
    jvec = np.array([], dtype=int)
    
    for edge in range(np.max(mesh.edge.topo)):

        mask = mesh.edge.topo > edge

        eidx = np.argwhere(mask).ravel()

        edsh = mesh.edge.edge[mask, edge] - 1

        wmul = mesh.edge.wmul[mask, edge]

        ivec = np.hstack((ivec, eidx))
        jvec = np.hstack((jvec, edsh))
        xvec = np.hstack((xvec, wmul))

    return csr_matrix((xvec, (ivec, jvec)))


def dual_curl_sums(mesh):

#-- DUAL-CURL-SUMS: returns SUM(lij * F_e) via sparse matrix
#-- operator OP. Use CURL(F) = OP * F where F is a vector of
#-- (signed) fluxes for all edges in the mesh.

    xvec = np.array([], dtype=float)
    ivec = np.array([], dtype=int)
    jvec = np.array([], dtype=int)

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

    xvec = np.array([], dtype=float)
    ivec = np.array([], dtype=int)
    jvec = np.array([], dtype=int)

    for cell in range(3):

        vidx = np.arange(0, mesh.vert.size)

        cidx = mesh.vert.cell[:, cell] - 1

        kite = mesh.vert.kite[:, cell]

        ivec = np.hstack((ivec, vidx))
        jvec = np.hstack((jvec, cidx))
        xvec = np.hstack((xvec, kite))
    
    return csr_matrix((xvec, (ivec, jvec)))


def dual_cell_sums(mesh):

    xvec = np.array([], dtype=float)
    ivec = np.array([], dtype=int)
    jvec = np.array([], dtype=int)

    for cell in range(3):

        vidx = np.arange(0, mesh.vert.size)

        cidx = mesh.vert.cell[:, cell] - 1

        unit = np.ones(
            mesh.vert.size, dtype=float)

        ivec = np.hstack((ivec, vidx))
        jvec = np.hstack((jvec, cidx))
        xvec = np.hstack((xvec, unit))
    
    return csr_matrix((xvec, (ivec, jvec)))


def dual_edge_sums(mesh):

    xvec = np.array([], dtype=float)
    ivec = np.array([], dtype=int)
    jvec = np.array([], dtype=int)

    for edge in range(3):

        vidx = np.arange(0, mesh.vert.size)

        eidx = mesh.vert.edge[:, edge] - 1

        unit = np.ones(
            mesh.vert.size, dtype=float)

        ivec = np.hstack((ivec, vidx))
        jvec = np.hstack((jvec, eidx))
        xvec = np.hstack((xvec, unit))
    
    return csr_matrix((xvec, (ivec, jvec)))
