
import numpy as np


def setunrm(mesh):

    unrm = np.zeros((mesh.edge.size, 1), dtype=float)

    return unrm


def setzlev(mesh):

    zlev = np.zeros((mesh.cell.size, 2), dtype=float)

   #zlev[:, 0] = 0.             # surface
    zlev[:, 0] = 10. * np.exp(-25. * (
        (mesh.cell.xlon - np.pi * 1.) ** 2 + 
        (mesh.cell.ylat - np.pi * 0.) ** 2
    ))

    zlev[:, 1] = -1.            # bottom

    return zlev


def setcnfg(mesh):

    class base: pass

    cnfg = base()
    cnfg.step = 7500.
    cnfg.iter = 2500
    cnfg.freq = 50
    cnfg.grav = 9.81
    cnfg.rho0 = 1000.

    return cnfg
