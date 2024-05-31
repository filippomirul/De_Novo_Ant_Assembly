#C:\Users\filoa\AppData\Local\Programs\Python\Python311\ python


from lib.colony import *
from lib.seq_dealing import *
import subprocess
import os
import shlex
import argparse
import textwrap
import datetime 
from scipy.io import savemat, loadmat

from scipy.linalg import get_blas_funcs, triu


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

    print(f"[{datetime.datetime.now()}]: Benvienidos!!")

    incipit()

    # Argument Parsing

    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = textwrap.dedent("""
        """))

    parser.add_argument("-i", "--input", type = str, help = "The input must be a fasta or fastq file", nargs="+")
    parser.add_argument("-o", "--output_directory", type = str, help = "Directory of output", default = "/standard_output")
    parser.add_argument("--tmp", help="Temporary directory where to store temporary files. If not passed tmp files will be saved in the current directory."
                        , type = str, default="./")

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
    parser.add_argument("-L", "--ipothetical_length", default = 100000,
                         help = "For a better reconstruction of the genome an ipotetical lenght of the sequence to rebuild is fondamental for retriving good results")

    args = parser.parse_args()

    # Asserting location
    current_path = os.getcwd()
    current_path = "/".join(current_path.split("\\"))
    # print(current_path)

    out_dir = current_path + args.output_directory

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.tmp != None:
        tmp_path = args.tmp
    else:
        tmp_path = current_path

    data_out_path = current_path + "/Data"
    graph_path = data_out_path + "/graph_metadata.txt"
    final_array_path = data_out_path + "/final_array.mat"
    reads_path = data_out_path + "/reads.txt"
    phred_path = current_path +"/phred.txt"

    print(f"[{datetime.datetime.now()}]: Output directory : {out_dir}")
    print(f"[{datetime.datetime.now()}]: Number of cpus given : {args.cpu_cores}")
    print(f"[{datetime.datetime.now()}]: Parameters passed to the Ant coony system algorithm")
    print(f"[{datetime.datetime.now()}]: Number of ants generations: {args.max_generation}")
    print(f"[{datetime.datetime.now()}]: Number of ants at each generation: {args.population_size}")
    print(f"[{datetime.datetime.now()}]: Evaporation rate: {args.evaporation_rate}")
    print(f"[{datetime.datetime.now()}]: Learning rate: {args.learning_rate}\n")



    # Reads simulation

    if args.test == "test":
        args.input = "C:\\Users\\filoa\\Desktop\\Programming_trials\\Assembler\\Main\\Data\\GCA_014117465.1_ASM1411746v1_genomic.fna"
        reads_length = 2000
        coverage = 12
        args.verbose = True

        sequence = extracting_sequence_from_data(args.input, limit = args.ipothetical_length)
        print(f"[{datetime.datetime.now()}]: Lenght of the sequence is : {len(sequence)}")
        print(f"[{datetime.datetime.now()}]: Coverage: {coverage}")
        print(f"[{datetime.datetime.now()}]: Lenght of the reads: {reads_length}")

        reads = custom_reads(seq = sequence, coverage = coverage, length_reads = reads_length,
                              res_path=out_dir, verbose=args.verbose)
        print(f"[{datetime.datetime.now()}]: Number of reads : {len(reads)}")

    
    else:

        # Problem with the args: ipotetical lenght and reads lenght
        if len(args.input) > 1:

            r = Parallel(n_jobs=args.cpu_cores)(delayed(extract_reads)(i) for i in args.input)

        else:

            print(f"[{datetime.datetime.now()}]: Accessing input files:")

            phred_score_reads = extract_reads(args.input[0]) 
            reads = phred_score_reads[0][:int(len(phred_score_reads[0])/1)]

            print(f"[{datetime.datetime.now()}]: There are {len(reads)} reads given as input! ")
            # savemat(phred_path, mdict= {"phred_reads":phred_score_reads[1]}, do_compression=True, appendmat=True)


    print(f"[{datetime.datetime.now()}]: Starting the encoding of the reads")

    reads = parallel_coding(reads=reads, number_cpus=args.cpu_cores)

    # with open(reads_path, "w") as file:


    # print(f"reads: {reads[0]}")
    # can't save with .mat because reads have different length


    print(f"[{datetime.datetime.now()}]: Reads has been successfully converted!")

    # Building the Overlap-Layout Consensus (OLC) graph

    print(f"[{datetime.datetime.now()}]: Building the Overlap-Layout Consensus (OLC) graph")

    links = Parallel(n_jobs=args.cpu_cores)(delayed(split_align)(i)for i in [(reads,j) for j in range(len(reads))])

    graph = assemble_matrix(links, len(reads))

    # print(f"graph: {graph[0]}")
    
    num_links = eval_nonzeros(graph)/2 # change this with something similar
    print(f"[{datetime.datetime.now()}]: Finished the building of the data structure. Graph has {num_links} edges")

    # print(graph_path)

    # savemat(graph_path, mdict = {"data":graph}, do_compression = False, appendmat=True)

    command_shell_simpl = f"python simplification.py -i {graph_path}"  # Add things

    subprocess.run(shlex.split(command_shell_simpl), check=True)

    graph_data_sempl= data_out_path + "/graph_sempl_metadata.mat"

    if args.test == "test":

        command_shell_test = f"python travel.py -i {graph_data_sempl} --reads_lenght {reads_length} -L {args.ipothetical_length}"

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

    # print(f"best_ACS: {best_ACS}")

    print(f"[{datetime.datetime.now()}]: Building up the consesus matrix")
    
    final_recons = final_consensus(best_ACS, reads, positions = graph, length = args.ipothetical_length)

    print(f"[{datetime.datetime.now()}]: Retriving additional information and statistics")

    final_recons = join_consensus_sequence(consensus_matrix=final_recons, cpus=args.cpu_cores)

    if args.test:

        # eff = efficiency(reference=sequence, recostructed_sequence=final_recons)

        # print(f"[{datetime.datetime.now()}]: The efficierncy in recostrcting the sequernce is: {eff}%")

        print(f"[{datetime.datetime.now()}]: Writing the output files...")
        
        out_files(ref = sequence, reconstructed_seq=final_recons, out_path=out_dir)

    return print(f"The Assembly: {final_recons} \nLength: {len(final_recons)}")

if __name__ == "__main__":
    main()

    # TODO list:
    #       Output writing
    #       Embeddings
    #       Clustering
    #       Cross-over
    #       Phred score in consensus sequence
    #       linkage between variants

