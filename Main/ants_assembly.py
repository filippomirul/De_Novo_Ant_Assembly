from lib.ants import *
from lib.colony import *
from inspyred import swarm, ec
import subprocess
import argparse
import textwrap
from joblib import Parallel, delayed
import time 
from scipy.io import savemat


def incipit():
    print("""
         _       __    _   __________ 
        / \     |   \ | | |___   ____|
       / _ \    | |\ \| |     | |     
      / /_\ \   | | \   |     | |       
     /  ___  \  | |  \  |     | |     ___|^-^| ___|^-^|   
    /_/     \_\ |_|   \_|     |_|     /\ /\    /\ /\ 

    Author: Filippo A. Mirolo, 2024
    """)
    return None

def main():

    incipit()

    # Argument Parsing

    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = textwrap.dedent("""
        """))

    parser.add_argument("-i", "--input", type = str, help = "The input must be a fasta or fastq file") # required = True
    parser.add_argument("-o", "--output_directory", type = str, help = "Directory of output", default="./")

    parser.add_argument("-ont", "--nanopore_long_reads", type = str, help = "Nanoporo long reads")
    parser.add_argument("-hifi", "--Pacbio_long_reads", type = str, help = "Hifi Pacbio reads")
    
    parser.add_argument("-l", "--reads_length", type = int, help = "Lenght of the custum reads", default = 200) # required=True
    parser.add_argument("-cov", "--coverage", type = int, default = 12,
                         help = "In the simulation of the reads, resamble the theoretical sequencing covearge")
    parser.add_argument("-p", "--population_size", type = int, default = 80,
                        help = "")
    parser.add_argument("-e", "--evaporation_rate", type = float, default = 0.2,
                         help="Internal parameter of the ant colony system")
    parser.add_argument("-r", "--learning_rate", type = float, default = 0.4,
                        help = "Internal parameter for the ant colony system")
    parser.add_argument("-v", "--verbose", type = bool, default = False,
                        help = "Prints and return more information on how the process is developing")
    parser.add_argument("-cpus", "--cpus_cores", type = int,
                        help = "Number of cpu to use; default = 2", default = 2)
    parser.add_argument("-g", "--max_generation", default = 10, help = "Number of iterations/generatios of the ant colony algorithm")
    parser.add_argument("-s", "--simulation", default = True, help="This is development only!")
    parser.add_argument("--ipothetical_length", default = 500,
                         help = "For a better reconstruction of the genome an ipotetical lenght of the sequence to rebuild is fondamental for retriving good results")

    args = parser.parse_args()

    # Reads simulation

    if args.input == None:
        args.input = "C:\\Users\\filoa\\Desktop\\Programming_trials\\Assembler\\Data\\GCA_014117465.1_ASM1411746v1_genomic.fna"

    sequence = extracting_sequence(args.input, limit = args.ipothetical_length)

    reads = custom_reads(seq = sequence, coverage = args.coverage, length_reads = args.reads_length)

    reads = Parallel(n_jobs=args.cpus)(delayed(uni_code)(i)for i in reads)

    # Building the Overlap-Layout Consensus (OLC) graph

    graph = eval_allign_np(reads = reads)

    data = {"data":graph}

    savemat("data", data, do_compression=True)

# Here goes the travel scipt

    subprocess.run("...")

    cov = args.coverage + (args.coverage/10)

    final_recons = final_consensus(best_ACS, reads, positions = graph, max_coverage = cov, length = args.ipothetical_length)

    return print(final_consensus)

if __name__ == "__main__":
    main()