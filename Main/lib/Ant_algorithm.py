from Bio import SeqIO
from Bio import pairwise2   # Biopython version <= 1.79
import datetime
import argparse
import textwrap
import os
import yaml
from Bio.Seq import Seq
import numpy as np
import random
import matplotlib.pyplot as plt
from random import Random
import time
from tqdm import tqdm
from itertools import permutations
import collections
from numba import jit
from itertools import combinations

# from joblib import Parallel, delayed
# TODO try parallelization, parser


def custom_reads(seq: str, length_reads:int = 160, coverage:int = 5, verbose = False) -> list:
    """The function splits the sequence in input into reads.
    The splitting is done using random numbers, and the number of reads is given by: (len(seq)/length_read)*coverage.
    """
    len_sequence = (len(seq))
    number_of_reads = int(len_sequence/length_reads) * coverage
    starting_pos = random.sample(range(0, len_sequence - (length_reads/2) + 1), number_of_reads)
    reads = []

    for num in starting_pos:
        if num+length_reads> len_sequence:
            reads.append(seq[num:])
        else:
            reads.append(seq[num:num + length_reads])

    if verbose:
        y = [0 for _ in range(0, len(seq) + 1)]
        for i in starting_pos:
            for j in range(i, i + length_reads + 1):
                y[j] += 1

        plt.plot(y)
        plt.xlabel("Position")
        plt.ylabel("Coverage")
        plt.title("Coverage Plot")
        plt.savefig("C:\\Users\\filoa\\OneDrive\\Desktop\\Programming_trials\\Assembler")

        print(f"There are {y.count(0)} bases that have 0 coverage.")

    return reads

@jit(nopython = True)
def np_score(align_list: np.ndarray, zeros = True)->int:
    """This function is a replacement for the np function np.count_nonzero(), since inside the np_eval_function was needed to count the number of zeros (=matches).
    However this function raise an error when run with the numba decorator.
    """
    length = len(align_list)
    cnt = 0

    for i in align_list:
        if i == 0:
            cnt += 1

    if zeros:
        return cnt
    else:
        return length-cnt

@jit(nopython=True)
def np_align_func(seq_one:np.ndarray, seq_two:np.ndarray, match:int = 3, mismatch:int = -2):
    """This function is a replacement for the align function pirwise2.align.localms of the Bio library. This substitution has the aim of tackling the computational time of the
    eval_alignment function. In order to decrease the time, there was the need to create a compilable function with numba, which was also capable of being parallelised.
    As you can clearly see the function takes in input only the match and mismatch, because in this usage the gap introduction is useless.

    seq_one, seq_two = input sequences already trasformed in byte
    match, mismatch = integer value for the alignment

    Note: the mismatch should be negative
    """

    # Initialization of output, and since the the func return only one vaue for each the check will be if the saved value is greater or not
    score = 0
    diff = 0
    switch = False

    # This is neceesary since knowing which one in the longest is also needed. seq_one is now scurely the longest
    if seq_one.shape[0] >= seq_two.shape[0]:
        max_lenght_seq = seq_one.shape[0]
        min_length_seq = seq_two.shape[0]

    else:
        switch = True
        max_lenght_seq = seq_two.shape[0]
        min_length_seq = seq_two[0]
        seq_one, seq_two = seq_two, seq_one
    
    # Number of iterations
    num_iteration_int = (max_lenght_seq + min_length_seq - 1) // 2
    num_iteration = (max_lenght_seq + min_length_seq - 1) / 2
    alone = False

    if num_iteration > num_iteration_int:
        alone = True

    for i in range(num_iteration_int):
        if i < min_length_seq:

            align_forw = seq_one[:(i+1)] - seq_two[-(i+1):]
            align_back = seq_two[:(i+1)] - seq_one[-(i+1):]

            cnt = 0
            for j in align_forw, align_back:
                part_score = np_score(j)*match + np_score(j, zeros=False)*mismatch

                if part_score > score:
                    score = part_score
                    if cnt > 0:
                        diff = max_lenght_seq -i -1
                    else:
                        diff = -(min_length_seq -i -1)
                cnt += 1
        
        if i >= min_length_seq:
            align_forw = seq_one[i-min_length_seq+1:(i+1)] - seq_two[-(i+1):]
            align_back = seq_one[-(i+1):-(i-min_length_seq+1)] - seq_two[:(i+1)]

            cnt = 0
            for j in align_forw, align_back:
                part_score = np_score(j)*match + np_score(j, zeros=False)*mismatch

                
                if part_score > score:
                    score = part_score
                    if cnt > 0:
                        diff = max_lenght_seq -i -1
                    else:
                        diff = -(min_length_seq -i -1)
                cnt += 1 

        if i == (num_iteration_int-1) and alone:
            i += 1

            align_forw = seq_one[i-min_length_seq+1:(i+1)] - seq_two[-(i+1):]
            part_score = np_score(j)*match + np_score(j, zeros=False)*mismatch

            if part_score > score:
                score = part_score
                diff = max_lenght_seq -i -1


    return (score, diff, switch)

def eval_allign_np(reads:list, par:list = [3, -2]) -> np.ndarray:
    """Funtion that evaulate the alignment

    reads: list of DNA sequences, each of the read is a string

    par: list of parameters to performe the alignment
    es (the examples represent the defoult parameters):
    match_score = 2,
    mismatch_penalty = -5,
    gap_open_penalty = -22,
    gap_extension_penalty = -5

    output:
    Matrix with the weigts (distances) between the reads (nodes)
    In this matrix there are both the scores of the alignment, recognizable for the tipical integer score and
    a flot number which is needed after to recompose the sequence, it indicates the overlapping bases.
    Ex:
        allignment score -> 2.0, 13.0, ...
        overlapping number -> 12.24, 1.6, 19.56, ...
            the number before the . is the starting base, while the one after is the ending base. To avoid problem later
            with 0 a 1 digit is added for then remove it. So 12.30 become 12.301 but the corret indices are 12 and 30.

        These two numbers are link by the position in the matrix which are the trasposition
        Score 14.0 in position (1,5) --> 12.34 in position (5,1). Only the score position is referred
        to the direction of the edge.
        1 ---> 5 with allignment score 14 and read_1 is overlapped with read_5 in positions from 12 to 34 (both included)

    Example of a matrix with three reads:

        | 1    | 2    | 3    
     1  | 0    |3.0   |12.231 
     2  |34.601|  0   | 23.0
     3  | 18.0 |45.701|  0
    """
    length = len(reads)
    # initialization of the matrices
    weigth_matrix = np.zeros((length, length))

    # The score of the allingment of read[1] to read[2] is the same of the opposite (read[2] to read[1])
    # So when the function found the diretionality of the allignment put the score in rigth spot and a 0 in the wrong one.
    visited = collections.deque([j for j in range(length)])
    # comb = combinations(range(len(reads)),2)

    
    # for i,j in comb:

    for i in tqdm(range(length)):

        for j in visited:

            if i == j:
                # the diagonal of the matrix has 0 score because we are allinging the same sequence
                continue
            else:
                # pairwise must return a positive score, if there is no one it return None
                reads_1 = np.array([ord(c) for c in reads[i]])
                reads_2 = np.array([ord(c) for c in reads[j]])
                alignment = np_align_func(reads_1, reads_2, match = par[0], mismatch = par[1])

                if alignment[2]:
                    if alignment[1] > 0:
                        weigth_matrix[j, i] = alignment[0]
                        weigth_matrix[i, j] = float(f"{0}.{abs(alignment[1])}")
                    
                    else:
                        weigth_matrix[i, j] = alignment[0]
                        weigth_matrix[j, i] = float(f"{0}.{abs(alignment[1])}")

                else:
                    if alignment[1] > 0:
                        weigth_matrix[i, j] = alignment[0]
                        weigth_matrix[j, i] = float(f"{0}.{abs(alignment[1])}")
                    
                    else:
                        weigth_matrix[j, i] = alignment[0]
                        weigth_matrix[i, j] = float(f"{0}.{abs(alignment[1])}")

                    
        visited.popleft()
    print(f"Done matrix {len(weigth_matrix)}x{len(weigth_matrix)}")
    return weigth_matrix


    
def __set_up__(time:time) -> tuple:
        

    #Extracting the sequence from the fasta and selecting the lenght:
    seq = ""
    len_seq = 0
    for seq_record in SeqIO.parse(PATH_IN, format="fasta"):
        seq += seq_record.seq.upper()
        len_seq += len(seq_record)
        if len_seq > NUM_BASE:
            continue
    seq = seq[:NUM_BASE]

    # Producing the reads:
    reads = custom_reads(seq, length_reads = LENGHT_READS, coverage = COVERAGE, verbose=VERBOSE)
    print(f"[{time}]: Number of reads: {len(reads)}")

    # Constructing the matrix:
    weigths = eval_allign(reads)

    problem = Assembly_problem(matrix = weigths, approximate_length = len(seq), reads_length = LENGHT_READS)


    return (problem, reads, seq)


# def main():

#     # Starting time
#     now = datetime.datetime.now()
#     start = time.time()

#     problem , reads, seq = __set_up__(time = now)

#     # Partial for the matrix:
#     partial = time.time()
#     print(f"[{time}]: Time for matrix:  {partial - start}")
    
#     args = {}
#     args["fig_title"] = "ACS"

#     prng = Random(SEED)
#     args = {}
#     args["fig_title"] = "ACS"

#     # Problem and ACS:
#     ac = inspyred.swarm.ACS(prng, problem.components)
#     ac.terminator = inspyred.ec.terminators.generation_termination

#     if VERBOSE:
#         display = True
#         ac.observer = inspyred.ec.observers.stats_observer
#     else:
#         display = False

#     final_pop = ac.evolve(generator = problem.constructor,
#                         evaluator = inspyred.ec.evaluators.parallel_evaluation_mp, 
#                         mp_evaluator = problem.evaluator, 
#                         bounder = problem.bounder,
#                         maximize = problem.maximize,
#                         mp_nprocs = CPUS,
#                         pop_size = POP_SIZE,
#                         max_generations = MAX_GENERATIONS,
#                         evaporation_rate = EVAPORATION_RATE,
#                         learning_rate = LEARNING_RATE,
#                         **args)
#     best_ACS = max(ac.archive)

#     # Final results and final consensus sequence
#     c = [(i.element[0], i.element[1]) for i in best_ACS.candidate]
#     d = final_consensus(c, reads, length = 5000, positions = problem.weights)
#     al = pairwise2.align.localms(d, seq, 3,-1,-5,-5)[0]

#     # Writing the results:
#     ll = []
#     ll.append("The first line is the reconstructed seq, while the second is the real sequence:\n")
#     cnt=0
#     for i in range(50,len(al[0]),50):
#         ll.append(str(al[0][cnt:i]))
#         ll.append("\n")
#         ll.append(str(al[1][cnt:i]))
#         ll.append("\n\n")
#         cnt += 50

#     ll.append("\n")
#     ll.append("Score of the allignment after the reconstruction:\n")
#     ll.append(str(al[2]))
#     ll.append("\nThe percentage of macht in the allignment is:")
#     ll.append("\n")

#     cnt = 0
#     for i in range(len(al[0])):
#         if al[0][i] == al[1][i]:
#             cnt += 1
#     ll.append(str(cnt/len(seq))) 

#     if not os.path.exists(PATH_OUT):
#         os.makedirs

#     new_file = open(PATH_OUT, "w")
#     new_file.writelines(ll)
#     new_file.close()

#     stop = time.time()
#     print(f"[{now}] Time: {stop - start}")

################################################

# ./config_file.yaml

# with open(args.configuration_file, "r") as file:

#     file = yaml.safe_load(file)

#     PATH_IN = file["data_path_file"]
#     PATH_OUT = file["directory_to_save"]
#     COVERAGE = file["coverage"]
#     LENGHT_READS = file["custom_reads_lenght"]
#     NUM_BASE = file["num_of_bases"]
#     POP_SIZE = file["population_size"]
#     MAX_GENERATIONS = file["num_of_maximum_generations"]
#     SEED = file["seed"]
#     EVAPORATION_RATE = file["evaporation rate"]
#     LEARNING_RATE = file["learning_rate"]
#     CPUS = file["cpus"]
#     VERBOSE = file["verbose"]

# main()

# if __name__== "main":
#     main()
