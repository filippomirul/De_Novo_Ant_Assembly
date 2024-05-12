#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lib.colony import *
from lib.seq_dealing import *
import subprocess
import os
import shlex
import argparse
import textwrap
import datetime 
from scipy.io import savemat, loadmat


def incipit():
    print("""
      
    De Novo Ant - Assembly
     ______     __    _       _                     _     
    |  __  \   |   \ | |     / \                   / \     
    | |  |  |  | |\ \| |    / _ \      _____      / _ \    
    | |  |  |  | | \   |   / /_\ \    |_____|    / /_\ \   
    | |__|  |  | |  \  |  /  ___  \             /  ___  \       ____|^-^|  ____|^-^|          ____|^-^|
    | ____ /   |_|   \_| /_/     \_\           /_/     \_\      /\ /\      /\ /\              /\ /\ 
    
    Author: Filippo A. Mirolo, 2024     

    """)
    return None

def main():

    print(f"[{datetime.datetime.now()}]: Benvenidos!!")

    incipit()

    # Argument Parsing

    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = textwrap.dedent("""
        """))


    ## Real-Program

    parser.add_argument("-i", "--input", type = str, help = "The input must be a fasta or fastq file", nargs="+")
    parser.add_argument("-o", "--output_directory", type = str, help = "Directory of output", default = "/standard_output")


    # These 3 are useless for now
    # parser.add_argument("-ont", "--nanopore_long_reads", type = str, help = "Nanoporo long reads")
    # parser.add_argument("-hifi", "--Pacbio_long_reads", type = str, help = "Hifi Pacbio reads")
    parser.add_argument("--test", help = "This is for testing", nargs="?", const="test", type=str)
    
    parser.add_argument("-p", "--population_size", type = int, default = 150,
                        help = "")
    parser.add_argument("-e", "--evaporation_rate", type = float, default = 0.2,
                         help="Internal parameter of the ant colony system")
    parser.add_argument("-r", "--learning_rate", type = float, default = 0.4,
                        help = "Internal parameter for the ant colony system")
    parser.add_argument("-v", "--verbose", type = bool, default = True,
                        help = "Prints and return more information on how the process is developing")
    parser.add_argument("-cpus", "--cpu_cores", type = int,
                        help = "Number of cpu to use; default = 2", default = 2)
    parser.add_argument("-g", "--max_generation", default = 8, help = "Number of iterations/generatios of the ant colony algorithm")
    parser.add_argument("-L", "--ipothetical_length", default = 10000,
                         help = "For a better reconstruction of the genome an ipotetical lenght of the sequence to rebuild is fondamental for retriving good results")
    parser.add_argument("--ester_egg", type=int, default=1)
    args = parser.parse_args()

    # print(f"[{datetime.datetime.now()}]: {arg} with value: {args.arg}")
    print(args)


    current_path = os.getcwd()
    current_path = "/".join(current_path.split("\\"))
    # print(current_path)

    out_dir = current_path + args.output_directory

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_out_path = current_path + "/Data"

    graph_path = data_out_path + "/graph_metadata.mat"
    final_array_path = data_out_path + "/final_array.mat"
    reads_path = data_out_path + "/reads.mat"


    # Reads simulation

    if args.test == "test":
        args.input = "C:\\Users\\filoa\\Desktop\\Programming_trials\\Assembler\\Main\\Data\\GCA_014117465.1_ASM1411746v1_genomic.fna"
        reads_length = 200
        coverage = 12

        sequence = extracting_sequence_from_data(args.input, limit = args.ipothetical_length)
        print(f"[{datetime.datetime.now()}]: Lenght of the sequence is : {len(sequence)}")

        reads = custom_reads(seq = sequence, coverage = coverage, length_reads = reads_length,
                              res_path=args.output_directory, verbose=args.verbose)
    
    else:

        # Problem with the args: ipotetical lenght and reads lenght
        if len(args.input) > 1:

            r = Parallel(n_jobs=args.cpu_cores)(delayed(extract_reads)(i) for i in args.input)

        else:

            print(f"[{datetime.datetime.now()}]: Accessing input files:")

            phred_score_reads = extract_reads(args.input[0]) 
            reads = phred_score_reads[0][:int(len(phred_score_reads[0])/args.ester_egg)]

            print(f"[{datetime.datetime.now()}]: {len(reads)} ")
            phred_score_reads = phred_score_reads[1]


    print(f"[{datetime.datetime.now()}]: Starting the encoding of the reads")

    reads = parallel_coding(reads=reads, number_cpus=args.cpu_cores)

    # print(f"reads: {reads[0]}")

    rr = {"reads":reads}
    savemat(reads_path, mdict = rr, do_compression = False, appendmat=True)


    print(f"[{datetime.datetime.now()}]: Reads has been successfully converted!")

    # Building the Overlap-Layout Consensus (OLC) graph

    print(f"[{datetime.datetime.now()}]: Building the Overlap-Layout Consensus (OLC) graph")
    graph = eval_allign_np(reads = reads)

    # print(f"graph: {graph[0]}")
    
    num_links = eval_nonzeros(graph)/2
    print(f"[{datetime.datetime.now()}]: Finished the building of the data structure. Graph has {num_links} edges")

    # print(graph_path)

    data = {"data":graph}
    savemat(graph_path, mdict = data, do_compression = False, appendmat=True)

    command_shell_simpl = f"python simplification.py -i {graph_path}"

    subprocess.run(shlex.split(command_shell_simpl), check=True)

    graph_data_sempl= data_out_path + "/graph_sempl_metadata.mat"

    if args.test == "test":

        command_shell_test = f"python travel.py -i {graph_data_sempl}"

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

    print(f"best_ACS: {best_ACS}")

    print(f"[{datetime.datetime.now()}]: Building up the consesus matrix")
    # TODO check
    final_recons = final_consensus(best_ACS, reads, positions = graph, length = args.ipothetical_length)

    print(f"[{datetime.datetime.now()}]: Retriving additional information and statistics")

    final_recons = join_consensus_sequence(consensus_matrix=final_recons, cpus=args.cpu_cores)

    if args.test:

        # eff = efficiency(reference=sequence, recostructed_sequence=final_recons)

        # print(f"[{datetime.datetime.now()}]: The efficierncy in recostrcting the sequernce is: {eff}%")

        
        out_files(ref = sequence, reconstructed_seq=final_recons, out_path=out_dir)
        
        print(f"[{datetime.datetime.now()}]: Writing the output files...")



    return print(f"The Assembly: {final_recons} \nLength: {len(final_recons)}")

if __name__ == "__main__":
    main()

    # TODO list:
    #       Final_reconstructor
    #       Semplification
    #       Output writing

