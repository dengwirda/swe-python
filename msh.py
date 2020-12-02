
import numpy as np
from netCDF4 import Dataset


def load_mesh(name):
    """
    LOAD-MESH: load the NAME.nc MPAS mesh file into a
    local mesh data structure.

    """
    # Authors: Darren Engwirda

    class base: pass

    data = Dataset(name, "r")

    mesh = base()
    mesh.rsph = float(data.sphere_radius)

    mesh.cell = base()
    mesh.cell.size = int(data.dimensions["nCells"].size)
    mesh.cell.xpos = np.array(data.variables["xCell"])
    mesh.cell.ypos = np.array(data.variables["yCell"])
    mesh.cell.zpos = np.array(data.variables["zCell"])
    mesh.cell.xlon = np.array(data.variables["lonCell"])
    mesh.cell.ylat = np.array(data.variables["latCell"])
    mesh.cell.area = np.array(data.variables["areaCell"])   
    mesh.cell.vert = \
        np.array(data.variables["verticesOnCell"])
    mesh.cell.edge = \
        np.array(data.variables["edgesOnCell"])
    mesh.cell.topo = \
        np.array(data.variables["nEdgesOnCell"])

    mesh.edge = base()
    mesh.edge.size = int(data.dimensions["nEdges"].size)
    mesh.edge.xpos = np.array(data.variables["xEdge"])
    mesh.edge.ypos = np.array(data.variables["yEdge"])
    mesh.edge.zpos = np.array(data.variables["zEdge"])
    mesh.edge.xlon = np.array(data.variables["lonEdge"])
    mesh.edge.ylat = np.array(data.variables["latEdge"])
    mesh.edge.vlen = np.array(data.variables["dvEdge"])
    mesh.edge.clen = np.array(data.variables["dcEdge"])
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
    mesh.vert.xpos = np.array(data.variables["xVertex"])
    mesh.vert.ypos = np.array(data.variables["yVertex"])
    mesh.vert.zpos = np.array(data.variables["zVertex"])
    mesh.vert.xlon = np.array(data.variables["lonVertex"])
    mesh.vert.ylat = np.array(data.variables["latVertex"])
    mesh.vert.area = \
        np.array(data.variables["areaTriangle"])
    mesh.vert.kite = \
        np.array(data.variables["kiteAreasOnVertex"])
    mesh.vert.edge = \
        np.array(data.variables["edgesOnVertex"])
    mesh.vert.cell = \
        np.array(data.variables["cellsOnVertex"])

    return mesh


def load_flow(name):
    """
    LOAD-FLOW: load the NAME.nc MPAS mesh file into a
    local flow data structure.

    """

    class base: pass

    data = Dataset(name, "r")

    flow = base()
    flow.hdel = np.array(data.variables["layerThickness"])
    flow.unrm = np.array(data.variables["normalVelocity"])
    flow.zbot = np.array(data.variables["bottomDepth"])
    flow.frot = {"vert": [], "edge": [], "cell": []}
    flow.frot["vert"] = np.array(data.variables["fVertex"])
    flow.frot["edge"] = np.array(data.variables["fEdge"])
    flow.frot["cell"] = np.array(data.variables["fCell"])

    return flow
