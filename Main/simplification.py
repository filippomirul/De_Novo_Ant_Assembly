import os
import argparse
import textwrap
import datetime
from lib.Simplification_embedding import *
from scipy.io import savemat, loadmat


def main():

    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = textwrap.dedent("""
        """))

    parser.add_argument("-i", "--input", required=True,
                         help = "Matrix like structure representing the graph to be simplified")
    parser.add_argument("--cpu_cores", default=2, )

    args = parser.parse_args()

    current_path = os.getcwd()

    data_out_path = current_path + "/Data"
    plots_path = current_path + "Results"

    graph_path_sempl = data_out_path + "/graph_sempl_metadata.mat"

    # Semplifing the OCL

    print(f"[{datetime.datetime.now()}]: Starting with the simplification of the graph")

    graph = graph_semplification(graph=loadmat(args.input)["data"], cores = args.cpu_cores)
    num_links = eval_nonzeros(graph)/2

    print(f"[{datetime.datetime.now()}]: Finished the reduction of the data structure. Graph has {num_links} edges")
    print(f"[{datetime.datetime.now()}]: The problem dimension is {len(graph)}x{len(graph)}")

    # print(graph)
    data = {"data_semplified":graph}
    savemat(graph_path_sempl, mdict = data, do_compression = False, appendmat=True)

if __name__ == "__main__":
    main()