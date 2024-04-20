from numba import njit
from lib.ants import *
from lib.Ant_algorithm import *
import argparse
import textwrap
import subprocess 

def main():
    print("""
            _       __    _   __________ 
            / \     |   \ | | |___   ____|
        / _ \    | |\ \| |     | |     
        / /_\ \   | | \   |     | |       
        /  ___  \  | |  \  |     | |     ___|^-^| ___|^-^|   
        /_/     \_\ |_|   \_|     |_|     /\ /\    /\ /\ 

    Author: Filippo A. Mirolo, 2024
    """)

    subprocess.run("echo Hello!")
    print("\nMotherfucker")

    # Argument Parsing

    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = textwrap.dedent("""
        """))

    parser.add_argument("-cpus", "--cpus_cores", type = int,
                        help = "Number of cpu to use; default = 2", default = 2)
    parser.add_argument("-i", "--input", type = str, help = "The input must be a fasta or fastq file")
    parser.add_argument("-o", "--output_directory", type = str, help = "Directory of output")
    parser.add_argument("-ont", "--nanopore_long_reads", type = str, help = "Nanoporo long reads")
    parser.add_argument("-hifi", "--Pacbio_long_reads", type = str, help = "Hifi Pacbio reads")
    parser.add_argument()
    parser.add_argument()

    args = parser.parse_args()


    raise NotImplemented

if __name__ == "main":
    main()