#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import textwrap
import datetime
from lib.Simplification_embedding import *
from scipy.io import savemat

def main():

    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = textwrap.dedent("""
        """))

    parser.add_argument("-i", "--input", required=True,
                         help = "Matrix like structure representing the graph to be simplified")
    parser.add_argument("--cpu_cores", default=3, type=int, help = "Number of cpu cores to use.")

    args = parser.parse_args()

    current_path = os.getcwd()

    data_out_path = current_path + "/Data"
    selected_edge_path = data_out_path +"/selected_edges.pkl"
    distance_matrix_path = data_out_path + "/distance_matrix.mat"
    plots_path = current_path + "/Results"
    graph_path_sempl = data_out_path + "/graph_sempl_metadata.mat"

    # Semplifing the OCL

    all_links = load_list(where=args.input)

    # print(all_links)

    # print(f"Args: {args}")

    edges, distance_matrix = edge_selection(edges=all_links, cpu=args.cpu_cores)

    print(edges)

    save_list(data=edges, where=selected_edge_path)

    savemat(distance_matrix_path, mdict = {"Distance_matrix":distance_matrix}, do_compression = False)

if __name__ == "__main__":
    main()