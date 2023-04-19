
import numpy as np
import argparse

from msh import load_mesh, load_flow

# Numerical error metrics
# Authors: Darren Engwirda

def errs_cell(mesh, func):

    return np.sum(mesh.cell.area * func) / np.sum(mesh.cell.area)


def errs_edge(mesh, func):

    return np.sum(mesh.edge.area * func) / np.sum(mesh.edge.area)


def errs_dual(mesh, func):

    return np.sum(mesh.vert.area * func) / np.sum(mesh.vert.area)


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--test-file", dest="test_file", type=str,
        required=True, help="File containing approx.")

    parser.add_argument(
        "--base-file", dest="base_file", type=str,
        required=True, help="File containing *exact.")

    args = parser.parse_args()

    m1st = load_mesh(args.test_file)
    m2nd = load_mesh(args.base_file)

    print("space(m):", np.mean(m1st.edge.clen))

    f1st = load_flow(args.test_file)
    f2nd = load_flow(args.base_file)

    h1_cell = f1st.hh_cell[-1, :, 0]
    h2_cell = f2nd.hh_cell[-1, :, 0]

    l_two = np.sqrt(errs_cell(m1st, (h1_cell - h2_cell) ** 2)) / \
            np.sqrt(errs_cell(m1st, h2_cell ** 2))

    l_inf = np.max(np.abs(h1_cell - h2_cell)) / np.max(np.abs(h2_cell))

    print("l_two(hh):", l_two)
    print("l_inf(hh):", l_inf)

    u1_edge = f1st.uu_edge[-1, :, 0]
    u2_edge = f2nd.uu_edge[-1, :, 0]

    l_two = np.sqrt(errs_edge(m1st, (u1_edge - u2_edge) ** 2)) / \
            np.sqrt(errs_edge(m1st, u2_edge ** 2))

    l_inf = np.max(np.abs(u1_edge - u2_edge)) / np.max(np.abs(u2_edge))

    print("l_two(uu):", l_two)
    print("l_inf(uu):", l_inf)

    r1_dual = f1st.rv_dual[-1, :, 0]
    r2_dual = f2nd.rv_dual[-1, :, 0]

    l_two = np.sqrt(errs_dual(m1st, (r1_dual - r2_dual) ** 2)) / \
            np.sqrt(errs_dual(m1st, r2_dual ** 2))

    l_inf = np.max(np.abs(r1_dual - r2_dual)) / np.max(np.abs(r2_dual))

    print("l_two(rv):", l_two)
    print("l_inf(rv):", l_inf)

    p1_dual = f1st.pv_dual[-1, :, 0]
    p2_dual = f2nd.pv_dual[-1, :, 0]

    l_two = np.sqrt(errs_dual(m1st, (p1_dual - p2_dual) ** 2)) / \
            np.sqrt(errs_dual(m1st, p2_dual ** 2))

    l_inf = np.max(np.abs(p1_dual - p2_dual)) / np.max(np.abs(p2_dual))

    print("l_two(pv):", l_two)
    print("l_inf(pv):", l_inf)


