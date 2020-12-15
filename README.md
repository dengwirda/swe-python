## `swe-python`

A simple sandbox for unstructured-mesh spherical shallow water equation solvers.

    python3 swe.py 
    --mpas-file="path+name-to-mpas-mesh+init-file"
    --num-steps=number-of-time-steps 
    --time-step=delta_t 
    --save-freq=output-freq-th-steps

    python3 jet.py
    --mesh-file="path+name-to-mpas-mesh-file"
    --init-file="path+name-to-mpas-init-file"
    --with-pert=True
    --radius=6371220.

    python3 wtc.py
    --mesh-file="path+name-to-mpas-mesh-file"
    --init-file="path+name-to-mpas-init-file"
    --radius=6371220.
    --test-case=1..5
