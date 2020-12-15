
import os
import time
import numpy as np
import xarray
import argparse

from msh import load_mesh, load_flow
from ops import trsk_mats


def main(args):
    """
    SWE: a very simple driver to solve the SWE via spherical 
    MPAS meshes.

    """
    # Authors: Darren Engwirda

    class base: pass

    cnfg = base()
    cnfg.iter = args.iteration
    cnfg.step = args.time_step
    cnfg.freq = args.save_freq
    cnfg.grav = 9.80616

    name = args.mpas_file
    path, file = os.path.split(name)
    save = os.path.join(path, "out_" + file)

    mesh = load_mesh(name)
    flow = load_flow(name)

    trsk = trsk_mats(mesh)

    xout = []

    uu_edge = flow.uu_edge[0, :, 0]
    hh_cell = flow.hh_cell[0, :, 0]
    ff_cell = flow.ff_cell


    mesh.edge.mask = base()
    mesh.edge.mask.self = []
    mesh.edge.mask.dash = []

    for edge in range(np.max(mesh.edge.topo)):

        mesh.edge.mask.size = edge + 1

        mask = mesh.edge.topo > edge

        self = np.argwhere(mask).ravel()

        dash = mesh.edge.edge[mask, edge] - 1
    
        mesh.edge.mask.self.append(self)
        mesh.edge.mask.dash.append(dash)


    ttic = time.time()

    for step in range(cnfg.iter + 1):

    #-- a TVD-RK2 + FB scheme, a'la ROMS (but with variables
    #-- updated via the standard "monotone" TVD-RK approach)

        ff_dual = trsk.dual_cell_sums * ff_cell
        ff_dual = ff_dual / 3.0

    #-- 1st RK + FB stage

        BETA = 1.0 / 3.0

        hh_edge = hh_cell[mesh.edge.cell[:, 0] - 1] + \
                  hh_cell[mesh.edge.cell[:, 1] - 1]
        hh_edge = hh_edge * 0.5

        uh_edge = uu_edge * hh_edge

        dd_cell = trsk.cell_flux_sums * uh_edge
        dd_cell = dd_cell / mesh.cell.area
        
        h1_cell = hh_cell - 1.0 * cnfg.step * dd_cell


        hb_cell = hh_cell * (1.0 - BETA) + \
                  h1_cell * (0.0 + BETA)

        gp_edge = hb_cell[mesh.edge.cell[:, 1] - 1] - \
                  hb_cell[mesh.edge.cell[:, 0] - 1]
        gp_edge = gp_edge / mesh.edge.clen

        #ke_edge = mesh.edge.area * .50 * u1_edge ** 2
        ke_edge = mesh.edge.vlen * 
                  mesh.edge.clen * .25 * uu_edge ** 2
        ke_cell = trsk.cell_edge_sums * ke_edge
        ke_cell = ke_cell / mesh.cell.area

        gk_edge = ke_cell[mesh.edge.cell[:, 1] - 1] - \
                  ke_cell[mesh.edge.cell[:, 0] - 1]
        gk_edge = gk_edge / mesh.edge.clen

        hh_dual = trsk.dual_kite_sums * hb_cell
        hh_dual = hh_dual / mesh.vert.area                 

        rv_dual = trsk.dual_curl_sums * uu_edge
        rv_dual = rv_dual / mesh.vert.area

        av_dual = rv_dual + ff_dual
        pv_dual = av_dual / hh_dual

        av_cell = trsk.cell_kite_sums * av_dual
        av_cell = av_cell / mesh.cell.area
     
        pv_cell = av_cell / hb_cell        
        rv_cell = av_cell - ff_cell

        '''
        pv_edge = pv_dual[mesh.edge.vert[:, 0] - 1] + \
                  pv_dual[mesh.edge.vert[:, 1] - 1]
        pv_edge = pv_edge * 0.5
        '''

        pv_edge = trsk.edge_kite_sums * pv_dual
        pv_edge = pv_edge / mesh.edge.area
        
        qh_edge = perp_reco(mesh, uh_edge, pv_edge)
       
        ur_edge = gp_edge * cnfg.grav + gk_edge + qh_edge

        u1_edge = uu_edge - 1.0 * cnfg.step * ur_edge

        
    #-- 2nd RK + FB stage

        BETA = 2.0 / 3.0

        ht_cell = hh_cell * 0.5 + h1_cell * 0.5
        ut_edge = uu_edge * 0.5 + u1_edge * 0.5

        h1_edge = h1_cell[mesh.edge.cell[:, 0] - 1] + \
                  h1_cell[mesh.edge.cell[:, 1] - 1]
        h1_edge = h1_edge * 0.5

        uh_edge = u1_edge * h1_edge

        dd_cell = trsk.cell_flux_sums * uh_edge
        dd_cell = dd_cell / mesh.cell.area
        
        h2_cell = ht_cell - 0.5 * cnfg.step * dd_cell


        hb_cell = h1_cell * (1.0 - BETA) + \
                  h2_cell * (0.0 + BETA)

        gp_edge = hb_cell[mesh.edge.cell[:, 1] - 1] - \
                  hb_cell[mesh.edge.cell[:, 0] - 1]
        gp_edge = gp_edge / mesh.edge.clen

       #ke_edge = mesh.edge.area * .50 * u1_edge ** 2
        ke_edge = mesh.edge.vlen * 
                  mesh.edge.clen * .25 * u1_edge ** 2
        ke_cell = trsk.cell_edge_sums * ke_edge
        ke_cell = ke_cell / mesh.cell.area

        gk_edge = ke_cell[mesh.edge.cell[:, 1] - 1] - \
                  ke_cell[mesh.edge.cell[:, 0] - 1]
        gk_edge = gk_edge / mesh.edge.clen

        hh_dual = trsk.dual_kite_sums * hb_cell
        hh_dual = hh_dual / mesh.vert.area

        rv_dual = trsk.dual_curl_sums * u1_edge
        rv_dual = rv_dual / mesh.vert.area

        av_dual = rv_dual + ff_dual
        pv_dual = av_dual / hh_dual

        av_cell = trsk.cell_kite_sums * av_dual
        av_cell = av_cell / mesh.cell.area

        pv_cell = av_cell / hb_cell
        rv_cell = av_cell - ff_cell        

        '''
        pv_edge = pv_dual[mesh.edge.vert[:, 0] - 1] + \
                  pv_dual[mesh.edge.vert[:, 1] - 1]
        pv_edge = pv_edge * 0.5
        '''

        pv_edge = trsk.edge_kite_sums * pv_dual
        pv_edge = pv_edge / mesh.edge.area

        qh_edge = perp_reco(mesh, uh_edge, pv_edge)
       
        ur_edge = gp_edge * cnfg.grav + gk_edge + qh_edge

        u2_edge = ut_edge - 0.5 * cnfg.step * ur_edge


        uu_edge = u2_edge
        hh_cell = h2_cell


        if (step % cnfg.freq == 0):

            print("step:", step)

            xnow = xarray.Dataset()
            xnow["uu_edge"] = (("nEdges", "nVertLevels"), 
                np.reshape(uu_edge, (mesh.edge.size, 1)))
            xnow["hh_cell"] = (("nCells", "nVertLevels"), 
                np.reshape(hh_cell, (mesh.cell.size, 1)))
            xnow["pv_dual"] = (("nVertices", "nVertLevels"), 
                np.reshape(pv_dual, (mesh.vert.size, 1)))
            xnow["pv_cell"] = (("nCells", "nVertLevels"), 
                np.reshape(pv_cell, (mesh.cell.size, 1)))
            xnow["rv_dual"] = (("nVertices", "nVertLevels"), 
                np.reshape(rv_dual, (mesh.vert.size, 1)))
            xnow["rv_cell"] = (("nCells", "nVertLevels"), 
                np.reshape(rv_cell, (mesh.cell.size, 1)))
            xnow["ke_cell"] = (("nCells", "nVertLevels"), 
                np.reshape(ke_cell, (mesh.cell.size, 1)))

            xout.append(xnow)

    ttoc = time.time()

    print("tcpu:", ttoc - ttic)
    

    data = xarray.Dataset()
    data = xarray.merge(
        (data, xarray.concat(xout, dim="Time")))
    data.attrs["on_a_sphere"] = "YES"
    data.attrs["sphere_radius"] = mesh.rsph
    data.attrs["is_periodic"] = "NO"
    data.attrs["mesh_id"] = "swe-python"
    data["lonCell"] = (("nCells"), mesh.cell.xlon)
    data["latCell"] = (("nCells"), mesh.cell.ylat)
    data["xCell"] = (("nCells"), mesh.cell.xpos)
    data["yCell"] = (("nCells"), mesh.cell.ypos)
    data["zCell"] = (("nCells"), mesh.cell.zpos)
    data["areaCell"] = (("nCells"), mesh.cell.area)
    data["verticesOnCell"] = (
        ("nCells", "maxEdges"), mesh.cell.vert)
    data["edgesOnCell"] = (
        ("nCells", "maxEdges"), mesh.cell.edge)
    data["cellsOnCell"] = (
        ("nCells", "maxEdges"), mesh.cell.cell)
    data["nEdgesOnCell"] = (("nCells"), mesh.cell.topo)

    data["lonEdge"] = (("nEdges"), mesh.edge.xlon)
    data["latEdge"] = (("nEdges"), mesh.edge.ylat)
    data["xEdge"] = (("nEdges"), mesh.edge.xpos)
    data["yEdge"] = (("nEdges"), mesh.edge.ypos)
    data["zEdge"] = (("nEdges"), mesh.edge.zpos)
    data["dvEdge"] = (("nEdges"), mesh.edge.vlen)    
    data["dcEdge"] = (("nEdges"), mesh.edge.clen)
    data["verticesOnEdge"] = (
        ("nEdges", "TWO"), mesh.edge.vert)
    data["weightsOnEdge"] = (
        ("nEdges", "maxEdges2"), mesh.edge.wmul)
    data["cellsOnEdge"] = (
        ("nEdges", "TWO"), mesh.edge.cell)
    data["edgesOnEdge"] = (
        ("nEdges", "maxEdges2"), mesh.edge.edge)
    data["nEdgesOnEdge"] = (("nEdges"), mesh.edge.topo)

    data["lonVertex"] = (("nVertices"), mesh.vert.xlon)
    data["latVertex"] = (("nVertices"), mesh.vert.ylat)
    data["xVertex"] = (("nVertices"), mesh.vert.xpos)
    data["yVertex"] = (("nVertices"), mesh.vert.ypos)
    data["zVertex"] = (("nVertices"), mesh.vert.zpos)
    data["areaTriangle"] = (
        ("nVertices"), mesh.vert.area)
    data["kiteAreasOnVertex"] = (
        ("nVertices", "vertexDegree"), mesh.vert.kite)
    data["edgesOnVertex"] = (
        ("nVertices", "vertexDegree"), mesh.vert.edge)
    data["cellsOnVertex"] = (
        ("nVertices", "vertexDegree"), mesh.vert.cell)

    data.to_netcdf(
        save, format="NETCDF3_64BIT_OFFSET")

    return


def perp_reco(mesh, uh_edge, pv_edge):

    pv_flux = np.zeros(mesh.edge.size, dtype=float)

    for edge in range(mesh.edge.mask.size):

        self = mesh.edge.mask.self[edge]
        dash = mesh.edge.mask.dash[edge]

        pv_flux[self] -= mesh.edge.wmul[self, edge] * \
            uh_edge[dash] * (pv_edge[self] + pv_edge[dash])

    return 0.5 * pv_flux


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--mpas-file", dest="mpas_file", type=str,
        required=True, help="Path to user MPAS file.")

    parser.add_argument(
        "--time-step", dest="time_step", type=float,
        required=True, help="Length of time steps.")

    parser.add_argument(
        "--num-steps", dest="iteration", type=int,
        required=True, help="Number of time steps.")

    parser.add_argument(
        "--save-freq", dest="save_freq", type=int,
        required=True, help="Save each FREQ-th step.")

    main(parser.parse_args())
