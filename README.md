## `swe-python`

A 'sandbox' for unstructured-mesh spherical shallow water equation solver development.

To run the shallow-water solver:

    python3 swe.py
    --mpas-file="path+name-to-mpas-mesh+init-file"
    --num-steps=number-of-time-steps
    --time-step=delta_t
    --save-freq=output-freq-th-steps

    python3 swe.py --help  # for detail

Solver output is saved to an MPAS-format netCDF file that can be visualised via
e.g. [paraview](https://www.paraview.org/).

Input files for various cases can be built from MPAS-format mesh files using the 
included utilities. A set of example icosahedral grids are attached to 
[releases](https://github.com/dengwirda/swe-python/releases) of this repository. 

The barotopic jet of [Galewsky et al](https://doi.org/10.3402/tellusa.v56i5.14436):

    python3 jet.py
    --mesh-file="path+name-to-mpas-mesh-file"
    --init-file="path+name-to-mpas-init-file"
    --with-pert=True
    --radius=6371220.

Various [Williamson et al](https://doi.org/10.1016/S0021-9991(05)80016-6) SWE config.'s:

    python3 wtc.py
    --mesh-file="path+name-to-mpas-mesh-file"
    --init-file="path+name-to-mpas-init-file"
    --radius=6371220.
    --test-case=N

Surface gravity wave and tsunami cases:

    python3 ltc.py
    --mesh-file="path+name-to-mpas-mesh-file"
    --init-file="path+name-to-mpas-init-file"
    --radius=6371220.
    --test-case=N
    --wave-xmid=165.
    --wave-ymid=40.0

For example, to build + run the barotropic jet test case using the CVT-optimised 
'level-7' icosahedral mesh:

    # create the jet_cvt_7.nc initial conditions file from a mesh
    python3 jet.py \
    --mesh-file="mesh_cvt_7.nc" \
    --init-file="jet_cvt_7.nc" \
    --with-pert=True \
    --radius=6371220.

    # run this config. in the sandbox with various solver param.'s
    python3 swe.py \
    --mpas-file="jet_cvt_7.nc" \
    --num-steps=2592 \
    --time-step=200. \
    --save-freq=432 \
    --stat-freq=108 \
    --integrate="RK32-FB"

Output is saved to the `out_jet_cvt_7.nc` file, which can be opened for visualisation 
in paraview.
