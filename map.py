
import subprocess
import os
import numpy as np
import netCDF4 as nc
import argparse

#-- Output MPAS-style *.vtk, requires MPAS-Tools

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--mpas-file", dest="mpas_file", type=str,
        required=True, help="Path to user MPAS file.")

    opts = parser.parse_args()


    data = nc.Dataset(opts.mpas_file, "r+")

    if ("xtime" not in data.variables.keys()):
        data.createVariable("xtime", "f8", ("Time"))

    data["xtime"][:] = \
        np.arange(0, data.dimensions["Time"].size)

    data.close()


    args = ["paraview_vtk_field_extractor.py",
            "-3",
            "-l",
            "-d", "maxEdges=0",
            "-v", "allOnCells",
            "-f", os.path.join(opts.mpas_file),
            "-o", "sol_vtk"]
    print("")
    print("running:", " ".join(args))
    subprocess.check_call(args, env=os.environ.copy())
