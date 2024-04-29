from lib.ants import *
from lib.colony import *
import subprocess
import shlex
import argparse
import textwrap
from joblib import Parallel, delayed
import datetime 
from scipy.io import savemat, loadmat


def incipit():
    print("""
         _       __    _   __________ 
        / \     |   \ | | |___   ____|
       / _ \    | |\ \| |     | |     
      / /_\ \   | | \   |     | |       
     /  ___  \  | |  \  |     | |     ____|^-^|  ____|^-^|          ____|^-^|
    /_/     \_\ |_|   \_|     |_|     /\ /\      /\ /\              /\ /\ 

    Author: Filippo A. Mirolo, 2024
    """)
    return None

def main():

    print(f"[{datetime.datetime.now()}]")

    incipit()

    # Argument Parsing

    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = textwrap.dedent("""
        """))


    ## Real-Program

    parser.add_argument("-i", "--input", type = str, help = "The input must be a fasta or fastq file", nargs="+")
    parser.add_argument("-o", "--output_directory", type = str, help = "Directory of output", default = "./standard_output")


    # These 3 are useless for now
    # parser.add_argument("-ont", "--nanopore_long_reads", type = str, help = "Nanoporo long reads")
    # parser.add_argument("-hifi", "--Pacbio_long_reads", type = str, help = "Hifi Pacbio reads")
    parser.add_argument("--test", help = "This is for testing", nargs="?", const="test", type=str)
    
    parser.add_argument("-p", "--population_size", type = int, default = 80,
                        help = "")
    parser.add_argument("-e", "--evaporation_rate", type = float, default = 0.2,
                         help="Internal parameter of the ant colony system")
    parser.add_argument("-r", "--learning_rate", type = float, default = 0.4,
                        help = "Internal parameter for the ant colony system")
    parser.add_argument("-v", "--verbose", type = bool, default = False,
                        help = "Prints and return more information on how the process is developing")
    parser.add_argument("-cpus", "--cpu_cores", type = int,
                        help = "Number of cpu to use; default = 2", default = 7)
    parser.add_argument("-g", "--max_generation", default = 10, help = "Number of iterations/generatios of the ant colony algorithm")
    parser.add_argument("-L", "--ipothetical_length", default = 10000,
                         help = "For a better reconstruction of the genome an ipotetical lenght of the sequence to rebuild is fondamental for retriving good results")

    args = parser.parse_args()


    current_path = os.getcwd()
    data_out_path = "/".join(current_path.split("\\")[:-1]) + "/Data"

    graph_path = data_out_path + "/graph_metadata.mat"
    final_array_path = data_out_path + "/final_array.mat"


    # Reads simulation

    if args.test == "test":
        args.input = "C:\\Users\\filoa\\Desktop\\Programming_trials\\Assembler\\Data\\GCA_014117465.1_ASM1411746v1_genomic.fna"
        reads_length = 200
        coverage = 8

        sequence = extracting_sequence(args.input, limit = args.ipothetical_length)
        print(f"[{datetime.datetime.now()}]: Lenght of the sequence is : {len(sequence)}")

        reads = custom_reads(seq = sequence, coverage = coverage, length_reads = reads_length)
    
    else:

        reads = args.input 
        raise NotImplemented # need to be a way to covert file into list of read
        # length_reads = mean(args.input)

    print(f"[{datetime.datetime.now()}]: Starting the encoding of the reads")
    reads = Parallel(n_jobs=args.cpu_cores)(delayed(uni_code)(i)for i in reads)

    print(f"[{datetime.datetime.now()}]: Reads has been successfully converted!")

    # Building the Overlap-Layout Consensus (OLC) graph

    print(f"[{datetime.datetime.now()}]: Building the Overlap-Layout Consensus (OLC) graph")
    graph = eval_allign_np(reads = reads)
    
    num_links = eval_nonzeros(graph)/2
    print(f"[{datetime.datetime.now()}]: Finished the building of the data structure. Graph has {num_links} links")

    print(f"[{datetime.datetime.now()}]: Starting with the simplification of the graph")

    graph = graph_semplification(graph=graph, cores = args.cpu_cores)
    num_links = eval_nonzeros(graph)/2
    print(f"[{datetime.datetime.now()}]: Finished the building of the data structure. Graph has {num_links} links")

    print(f"[{datetime.datetime.now()}]: Simplification has been completed!!")
    print(f"[{datetime.datetime.now()}]: The problem dimension is {len(graph)}x{len(graph)}")

    # print(graph)
    data = {"data":graph}
    savemat(graph_path, mdict = data, do_compression = False, appendmat=True)

    # Here goes the travel scipt    

    if args.test == "test":

        command_shell_test = f"python travel.py -i {graph_path}"

        # print(shlex.split(command_shell_test))

        subprocess.run(shlex.split(command_shell_test), check=True)

    else:
        possible_arguments_travel = ["--input", "--output_directory", "--population_size", "--max_generation",
                           "--evaporation_rate", "--learning_rate", "--verbose", "--cpu_cores", "--ipothetical_length"]

        command_shell = "python travel.py"

        for argument in vars(args).keys():
            value = str(vars(args)[argument])

            if (value != "None") and (argument in possible_arguments_travel):
                command_shell = command_shell + f" --{argument} {value}"


        subprocess.run(shlex.split(command_shell), check=True)

    best_ACS = loadmat(final_array_path)["Best_ACS"]

    cov = coverage + int(coverage/10)

    print(f"[{datetime.datetime.now()}]: Building up the consesus matrix")
    # TODO check
    final_recons = final_consensus(best_ACS, reads, positions = graph, max_coverage = cov, length = args.ipothetical_length)

    print(f"[{datetime.datetime.now()}]: Retriving additional information and statistics")

    final_recons = join_consensus_sequence(consensus_matrix=final_recons, cpus=args.cpu_cores)

    # This last part will be implemented, will contain the output writing and addition information regarding the
    # results of all the processes in between 

    return print(final_recons)

if __name__ == "__main__":
    main()