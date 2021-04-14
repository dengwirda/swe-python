## `swe-python`

A simple sandbox for unstructured-mesh spherical shallow water equation solvers.

To run the shallow-water solver:

    python3 swe.py 
    --mpas-file="path+name-to-mpas-mesh+init-file"
    --num-steps=number-of-time-steps
    --time-step=delta_t
    --save-freq=output-freq-th-steps

Solver output is saved to an MPAS-format netCDF file that can be visualised via
paraview.

Input files for various cases can be built from MPAS-format mesh files using the 
following utilities. A set of example icosahedral grids are attached to releases
of this repository. 

The barotopic jet of Galewesky et al:

    python3 jet.py
    --mesh-file="path+name-to-mpas-mesh-file"
    --init-file="path+name-to-mpas-init-file"
    --with-pert=True
    --radius=6371220.

Various Williamson et al SWE config.:

    python3 wtc.py
    --mesh-file="path+name-to-mpas-mesh-file"
    --init-file="path+name-to-mpas-init-file"
    --radius=6371220.
    --test-case=2, 5

For example, to build + run the barotropic jet test case using the CVT-optimised 
'level-7' icosahedral mesh:

    python3 jet.py
    --mesh-file="mesh_cvt_7ref.nc"
    --init-file="jet_cvt_7ref.nc"
    --with-pert=True
    --radius=6371220.

    python3 swe.py
    --mpas-file="jet_cvt_7ref.nc"
    --num-steps=4320
    --time-step=120.
    --save-freq=270
    --operators="TRSK-CV"

Output is saved to the 'out_jet_cvt_7ref.nc' file, which can be opened for visualisation 
in paraview.
