
import numpy as np
import argparse

from msh import load_mesh, load_flow


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

    f1st = load_flow(args.test_file)
    f2nd = load_flow(args.base_file)

    h1_cell = f1st.hh_cell[-1, :, 0]
    h2_cell = f2nd.hh_cell[-1, :, 0]

   #l_two

    l_inf = np.max(np.abs(h1_cell - h2_cell)) / np.max(np.abs(h2_cell))

    print("l_inf(h):", l_inf)
    
