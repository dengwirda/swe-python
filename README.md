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

Various Williamson et al SWE configs:

    python3 wtc.py
    --mesh-file="path+name-to-mpas-mesh-file"
    --init-file="path+name-to-mpas-init-file"
    --radius=6371220.
    --test-case=1..5
