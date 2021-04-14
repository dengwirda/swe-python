
import numpy as np
from netCDF4 import Dataset


def load_mesh(name, rsph=None):
    """
    LOAD-MESH: load the NAME.nc MPAS mesh file into a
    local mesh data structure.

    """
    # Authors: Darren Engwirda, Sara Calandrini

    class base: pass

    data = Dataset(name, "r")

    mesh = base()
    mesh.rsph = float(data.sphere_radius)

    if (rsph is not None):
        scal = rsph / mesh.rsph
        mesh.rsph = mesh.rsph * scal
    else:
        scal = 1.0E+00

    mesh.cell = base()
    mesh.cell.size = int(data.dimensions["nCells"].size)
    mesh.cell.xpos = np.array(data.variables["xCell"]) * scal
    mesh.cell.ypos = np.array(data.variables["yCell"]) * scal
    mesh.cell.zpos = np.array(data.variables["zCell"]) * scal
    mesh.cell.xlon = np.array(data.variables["lonCell"])
    mesh.cell.ylat = np.array(data.variables["latCell"])
    mesh.cell.area = \
        np.array(data.variables["areaCell"]) * scal ** 2
    mesh.cell.vert = \
        np.array(data.variables["verticesOnCell"])
    mesh.cell.edge = \
        np.array(data.variables["edgesOnCell"])
    mesh.cell.cell = \
        np.array(data.variables["cellsOnCell"])
    mesh.cell.topo = \
        np.array(data.variables["nEdgesOnCell"])

    mesh.edge = base()
    mesh.edge.size = int(data.dimensions["nEdges"].size)
    mesh.edge.xpos = np.array(data.variables["xEdge"]) * scal
    mesh.edge.ypos = np.array(data.variables["yEdge"]) * scal
    mesh.edge.zpos = np.array(data.variables["zEdge"]) * scal
    mesh.edge.xlon = np.array(data.variables["lonEdge"])
    mesh.edge.ylat = np.array(data.variables["latEdge"])
    mesh.edge.vlen = np.array(data.variables["dvEdge"]) * scal
    mesh.edge.clen = np.array(data.variables["dcEdge"]) * scal
    mesh.edge.vert = \
        np.array(data.variables["verticesOnEdge"])
    mesh.edge.wmul = \
        np.array(data.variables["weightsOnEdge"])
    mesh.edge.cell = \
        np.array(data.variables["cellsOnEdge"])
    mesh.edge.edge = \
        np.array(data.variables["edgesOnEdge"])
    mesh.edge.topo = \
        np.array(data.variables["nEdgesOnEdge"])

    mesh.vert = base()
    mesh.vert.size = int(data.dimensions["nVertices"].size)
    mesh.vert.xpos = np.array(data.variables["xVertex"]) * scal
    mesh.vert.ypos = np.array(data.variables["yVertex"]) * scal
    mesh.vert.zpos = np.array(data.variables["zVertex"]) * scal
    mesh.vert.xlon = np.array(data.variables["lonVertex"])
    mesh.vert.ylat = np.array(data.variables["latVertex"])
    mesh.vert.area = \
        np.array(data.variables["areaTriangle"]) * scal ** 2
    mesh.vert.kite = \
        np.array(data.variables["kiteAreasOnVertex"]) * scal ** 2
    mesh.vert.edge = \
        np.array(data.variables["edgesOnVertex"])
    mesh.vert.cell = \
        np.array(data.variables["cellsOnVertex"])


    mesh.cell.xmid = \
        cell_quad(mesh, mesh.cell.xpos, mesh.vert.xpos)
    mesh.cell.ymid = \
        cell_quad(mesh, mesh.cell.ypos, mesh.vert.ypos)
    mesh.cell.zmid = \
        cell_quad(mesh, mesh.cell.zpos, mesh.vert.zpos)

    mesh.cell.xmid, mesh.cell.ymid, mesh.cell.zmid, \
    mesh.cell.mlon, mesh.cell.mlat = to_sphere(
        mesh, mesh.cell.xmid, mesh.cell.ymid, mesh.cell.zmid)

    mesh.edge.xmid = \
        edge_quad(mesh, mesh.cell.xpos, mesh.vert.xpos)
    mesh.edge.ymid = \
        edge_quad(mesh, mesh.cell.ypos, mesh.vert.ypos)
    mesh.edge.zmid = \
        edge_quad(mesh, mesh.cell.zpos, mesh.vert.zpos)

    mesh.edge.xmid, mesh.edge.ymid, mesh.edge.zmid, \
    mesh.edge.mlon, mesh.edge.mlat = to_sphere(
        mesh, mesh.edge.xmid, mesh.edge.ymid, mesh.edge.zmid)

    mesh.vert.xmid = \
        dual_quad(mesh, mesh.cell.xpos, mesh.vert.xpos)
    mesh.vert.ymid = \
        dual_quad(mesh, mesh.cell.ypos, mesh.vert.ypos)
    mesh.vert.zmid = \
        dual_quad(mesh, mesh.cell.zpos, mesh.vert.zpos)

    mesh.vert.xmid, mesh.vert.ymid, mesh.vert.zmid, \
    mesh.vert.mlon, mesh.vert.mlat = to_sphere(
        mesh, mesh.vert.xmid, mesh.vert.ymid, mesh.vert.zmid)


    mesh.edge.stub = np.zeros(
        (mesh.edge.size, 2), dtype=np.float64)
    mesh.edge.stub[:, 0] = tria_area(
        mesh.rsph,
        np.vstack((mesh.vert.xlon[mesh.edge.vert[:, 0] - 1],
                   mesh.vert.ylat[mesh.edge.vert[:, 0] - 1])).T,
        np.vstack((mesh.cell.xlon[mesh.edge.cell[:, 1] - 1],
                   mesh.cell.ylat[mesh.edge.cell[:, 1] - 1])).T,
        np.vstack((mesh.cell.xlon[mesh.edge.cell[:, 0] - 1],
                   mesh.cell.ylat[mesh.edge.cell[:, 0] - 1])).T
    )
    mesh.edge.stub[:, 1] = tria_area(
        mesh.rsph,
        np.vstack((mesh.vert.xlon[mesh.edge.vert[:, 1] - 1],
                   mesh.vert.ylat[mesh.edge.vert[:, 1] - 1])).T,
        np.vstack((mesh.cell.xlon[mesh.edge.cell[:, 0] - 1],
                   mesh.cell.ylat[mesh.edge.cell[:, 0] - 1])).T,
        np.vstack((mesh.cell.xlon[mesh.edge.cell[:, 1] - 1],
                   mesh.cell.ylat[mesh.edge.cell[:, 1] - 1])).T
    )

    mesh.edge.wing = np.zeros(
        (mesh.edge.size, 2), dtype=np.float64)
    mesh.edge.wing[:, 0] = tria_area(
        mesh.rsph,
        np.vstack((mesh.vert.xlon[mesh.edge.vert[:, 1] - 1],
                   mesh.vert.ylat[mesh.edge.vert[:, 1] - 1])).T,
        np.vstack((mesh.cell.xlon[mesh.edge.cell[:, 0] - 1],
                   mesh.cell.ylat[mesh.edge.cell[:, 0] - 1])).T,
        np.vstack((mesh.vert.xlon[mesh.edge.vert[:, 0] - 1],
                   mesh.vert.ylat[mesh.edge.vert[:, 0] - 1])).T
    )
    mesh.edge.wing[:, 1] = tria_area(
        mesh.rsph,
        np.vstack((mesh.vert.xlon[mesh.edge.vert[:, 0] - 1],
                   mesh.vert.ylat[mesh.edge.vert[:, 0] - 1])).T,
        np.vstack((mesh.cell.xlon[mesh.edge.cell[:, 1] - 1],
                   mesh.cell.ylat[mesh.edge.cell[:, 1] - 1])).T,
        np.vstack((mesh.vert.xlon[mesh.edge.vert[:, 1] - 1],
                   mesh.vert.ylat[mesh.edge.vert[:, 1] - 1])).T
    )

    mesh.edge.area = np.sum(mesh.edge.stub, axis=1) + \
                     np.sum(mesh.edge.wing, axis=1) 
    mesh.edge.area = mesh.edge.area * 0.5E+00
   
    return mesh


def load_flow(name):
    """
    LOAD-FLOW: load the NAME.nc MPAS mesh file into a
    local flow data structure.

    """

    class base: pass

    data = Dataset(name, "r")

    step = int(data.dimensions["Time"].size)
    ncel = int(data.dimensions["nCells"].size)
    nedg = int(data.dimensions["nEdges"].size)
    nvrt = int(data.dimensions["nVertices"].size)
    nlev = int(data.dimensions["nVertLevels"].size)

    flow = base()

    if ("config_gravity" in data.ncattrs()):
        flow.grav = float(data.config_gravity)
    else:
        flow.grav = 9.80616

    flow.hh_cell = np.zeros((step, ncel, nlev), dtype=float)

    flow.zb_cell = np.zeros((ncel), dtype=float)
    flow.ff_cell = np.zeros((ncel), dtype=float)
    flow.ff_edge = np.zeros((nedg), dtype=float)
    flow.ff_vert = np.zeros((nvrt), dtype=float)

    flow.uu_edge = np.zeros((step, nedg, nlev), dtype=float)
    flow.vv_edge = np.zeros((step, nedg, nlev), dtype=float)

    flow.ke_cell = np.zeros((step, ncel, nlev), dtype=float)

    flow.rv_cell = np.zeros((step, ncel, nlev), dtype=float)
    flow.pv_cell = np.zeros((step, ncel, nlev), dtype=float)
    flow.rv_dual = np.zeros((step, nvrt, nlev), dtype=float)
    flow.pv_dual = np.zeros((step, nvrt, nlev), dtype=float)

    if ("h" in data.variables.keys()):
        flow.hh_cell = np.array(data.variables["h"])
    if ("h_s" in data.variables.keys()):
        flow.zb_cell = np.array(data.variables["h_s"])

    if ("fCell" in data.variables.keys()):
        flow.ff_cell = np.array(data.variables["fCell"])
    if ("fEdge" in data.variables.keys()):
        flow.ff_edge = np.array(data.variables["fEdge"])
    if ("fVertex" in data.variables.keys()):
        flow.ff_vert = np.array(data.variables["fVertex"])
    
    if ("u" in data.variables.keys()):
        flow.uu_edge = np.array(data.variables["u"])
    if ("v" in data.variables.keys()):
        flow.vv_edge = np.array(data.variables["v"])

    if ("hh_cell" in data.variables.keys()):
        flow.hh_cell = np.array(data.variables["hh_cell"])
    if ("zb_cell" in data.variables.keys()):
        flow.zb_cell = np.array(data.variables["zb_cell"])

    if ("ff_cell" in data.variables.keys()):
        flow.ff_cell = np.array(data.variables["ff_cell"])
    if ("ff_edge" in data.variables.keys()):
        flow.ff_edge = np.array(data.variables["ff_edge"])
    if ("ff_vert" in data.variables.keys()):
        flow.ff_vert = np.array(data.variables["ff_vert"])

    if ("uu_edge" in data.variables.keys()):
        flow.uu_edge = np.array(data.variables["uu_edge"])
    if ("vv_edge" in data.variables.keys()):
        flow.vv_edge = np.array(data.variables["vv_edge"])

    if ("ke_cell" in data.variables.keys()):
        flow.ke_cell = np.array(data.variables["ke_cell"])

    if ("rv_cell" in data.variables.keys()):
        flow.rv_cell = np.array(data.variables["rv_cell"])
    if ("pv_cell" in data.variables.keys()):
        flow.pv_cell = np.array(data.variables["pv_cell"])

    if ("rv_dual" in data.variables.keys()):
        flow.rv_dual = np.array(data.variables["rv_dual"])
    if ("pv_dual" in data.variables.keys()):
        flow.pv_dual = np.array(data.variables["pv_dual"])

    return flow


def tria_area(rs, pa, pb, pc):

    lena = circ_dist(1., pa, pb)
    lenb = circ_dist(1., pb, pc)
    lenc = circ_dist(1., pc, pa)

    slen = 0.5 * (lena + lenb + lenc)

    tana = np.tan(0.5 * (slen - lena))
    tanb = np.tan(0.5 * (slen - lenb))
    tanc = np.tan(0.5 * (slen - lenc))

    edel = 4.0 * np.arctan(np.sqrt(
        np.tan(0.5 * slen) * tana * tanb * tanc))

    return edel * rs ** 2


def circ_dist(rs, pa, pb):

    dlon = .5 * (pa[:, 0] - pb[:, 0])
    dlat = .5 * (pa[:, 1] - pb[:, 1])

    dist = 2. * rs * np.arcsin(np.sqrt(
        np.sin(dlat) ** 2 +
        np.sin(dlon) ** 2 * np.cos(pa[:, 1]) * np.cos(pb[:, 1])
    ))

    return dist


def cell_quad(mesh, fcel, fvrt):
    
    pcel = np.vstack(
        (mesh.cell.xlon, mesh.cell.ylat)).T
    pvrt = np.vstack(
        (mesh.vert.xlon, mesh.vert.ylat)).T

    abar = np.zeros(mesh.cell.size, dtype=np.float64)
    fbar = np.zeros(mesh.cell.size, dtype=np.float64)

    rsph = mesh.rsph

    for epos in range(np.max(mesh.cell.topo)):

        mask = mesh.cell.topo > epos

        cidx = np.argwhere(mask).ravel()

        ifac = mesh.cell.edge[mask, epos] - 1

        ivrt = mesh.edge.vert[ifac, 0] - 1
        jvrt = mesh.edge.vert[ifac, 1] - 1

        atri = tria_area(
            rsph, pcel[cidx], pvrt[ivrt], pvrt[jvrt])

        ftri = (fcel[cidx] + fvrt[ivrt] + fvrt[jvrt])

        abar[cidx] += atri
        fbar[cidx] += atri * ftri / 3.0

    return fbar / abar


def edge_quad(mesh, fcel, fvrt):

    pcel = np.vstack(
        (mesh.cell.xlon, mesh.cell.ylat)).T
    pvrt = np.vstack(
        (mesh.vert.xlon, mesh.vert.ylat)).T

    abar = np.zeros(mesh.edge.size, dtype=np.float64)
    fbar = np.zeros(mesh.edge.size, dtype=np.float64)

    rsph = mesh.rsph

    for epos in range(1):

        eidx = np.arange(0, mesh.edge.size)

        ivrt = mesh.edge.vert[eidx, 0] - 1
        jvrt = mesh.edge.vert[eidx, 1] - 1

        icel = mesh.edge.cell[eidx, 0] - 1
        jcel = mesh.edge.cell[eidx, 1] - 1

        atri = tria_area(
            rsph, pvrt[ivrt], pcel[icel], pcel[jcel])

        ftri = (fvrt[ivrt] + fcel[icel] + fcel[jcel])

        abar[eidx] += atri
        fbar[eidx] += atri * ftri / 3.0

        atri = tria_area(
            rsph, pvrt[jvrt], pcel[jcel], pcel[icel])

        ftri = (fvrt[jvrt] + fcel[jcel] + fcel[icel])

        abar[eidx] += atri
        fbar[eidx] += atri * ftri / 3.0

    return fbar / abar


def dual_quad(mesh, fcel, fvrt):

    pcel = np.vstack(
        (mesh.cell.xlon, mesh.cell.ylat)).T
    pvrt = np.vstack(
        (mesh.vert.xlon, mesh.vert.ylat)).T

    abar = np.zeros(mesh.vert.size, dtype=np.float64)
    fbar = np.zeros(mesh.vert.size, dtype=np.float64)

    rsph = mesh.rsph

    for epos in range(3):

        vidx = np.arange(0, mesh.vert.size)

        ifac = mesh.vert.edge[vidx, epos] - 1

        icel = mesh.edge.cell[ifac, 0] - 1
        jcel = mesh.edge.cell[ifac, 1] - 1

        atri = tria_area(
            rsph, pvrt[vidx], pcel[icel], pcel[jcel])

        ftri = (fvrt[vidx] + fcel[icel] + fcel[jcel])

        abar[vidx] += atri
        fbar[vidx] += atri * ftri / 3.0

    return fbar / abar


def to_sphere(mesh, xpos, ypos, zpos):

    radii = mesh.rsph * np.ones((3), dtype=np.float64)
    
    xmid = 0.5 * xpos
    ymid = 0.5 * ypos
    zmid = 0.5 * zpos

    ax = xmid ** 1 / radii[0] ** 1
    ay = ymid ** 1 / radii[1] ** 1
    az = zmid ** 1 / radii[2] ** 1

    aa = ax ** 2 + ay ** 2 + az ** 2

    bx = xmid ** 2 / radii[0] ** 2
    by = ymid ** 2 / radii[1] ** 2
    bz = zmid ** 2 / radii[2] ** 2

    bb = bx * 2. + by * 2. + bz * 2.

    cx = xmid ** 1 / radii[0] ** 1
    cy = ymid ** 1 / radii[1] ** 1
    cz = zmid ** 1 / radii[2] ** 1

    cc = cx ** 2 + cy ** 2 + cz ** 2
    cc = cc - 1.0

    ts = bb * bb - 4. * aa * cc

    ok = ts >= .0

    AA = aa[ok]; BB = bb[ok]; CC = cc[ok]; TS = ts[ok]

    t1 = (-BB + np.sqrt(TS)) / AA / 2.0
    t2 = (-BB - np.sqrt(TS)) / AA / 2.0

    tt = np.maximum(t1, t2)
    
    xprj = np.zeros(xpos.shape, dtype=np.float64)
    xprj[ok] = (1. + tt) * xmid[ok]
    
    yprj = np.zeros(ypos.shape, dtype=np.float64)
    yprj[ok] = (1. + tt) * ymid[ok]    

    zprj = np.zeros(zpos.shape, dtype=np.float64)
    zprj[ok] = (1. + tt) * zmid[ok]

    xrad = xprj * radii[1]
    yrad = yprj * radii[0]
    zrad = zprj / radii[2]

    zrad = np.maximum(np.minimum(zrad, +1.), -1.)

    ylat = np.arcsin(zrad)
    xlon = np.arctan2(yrad, xrad)

    return xprj, yprj, zprj, xlon, ylat
