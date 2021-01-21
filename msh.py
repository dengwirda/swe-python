
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

    mesh.edge.kite = np.zeros(
        (mesh.edge.size, 2), dtype=float)
    mesh.edge.kite[:, 0] = tria_area(
        mesh.rsph, 
        np.vstack((mesh.vert.xlon[mesh.edge.vert[:, 0] - 1], 
                   mesh.vert.ylat[mesh.edge.vert[:, 0] - 1])).T,
        np.vstack((mesh.cell.xlon[mesh.edge.cell[:, 1] - 1], 
                   mesh.cell.ylat[mesh.edge.cell[:, 1] - 1])).T,
        np.vstack((mesh.cell.xlon[mesh.edge.cell[:, 0] - 1], 
                   mesh.cell.ylat[mesh.edge.cell[:, 0] - 1])).T
    )
    mesh.edge.kite[:, 1] = tria_area(
        mesh.rsph, 
        np.vstack((mesh.vert.xlon[mesh.edge.vert[:, 1] - 1], 
                   mesh.vert.ylat[mesh.edge.vert[:, 1] - 1])).T,
        np.vstack((mesh.cell.xlon[mesh.edge.cell[:, 0] - 1], 
                   mesh.cell.ylat[mesh.edge.cell[:, 0] - 1])).T,
        np.vstack((mesh.cell.xlon[mesh.edge.cell[:, 1] - 1], 
                   mesh.cell.ylat[mesh.edge.cell[:, 1] - 1])).T
    )

    mesh.edge.area = np.sum(mesh.edge.kite, axis=1)
   #mesh.edge.area = mesh.edge.vlen * mesh.edge.clen * .5

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
