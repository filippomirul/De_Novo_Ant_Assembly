import collections
from itertools import combinations
from numba import njit
from Bio import pairwise2
import random
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from multiprocessing import Pool
from joblib import Parallel, delayed
import seaborn as sns
import torch as tr
import time

seq = ""
for seq_record in SeqIO.parse("C:\\Users\\filoa\\Desktop\\Programming_trials\\Assembler\\GCA_014117465.1_ASM1411746v1_genomic.fna", format = "fasta"):
    seq += str(seq_record.seq.upper())

def comstum_reads(seq: str, length_reads = 10, coverage = 5, verbose = False) -> list:
    
    """The function split the sequence in input in reads.
    The splitting is done using random numbers, the amount of reds is given by: (len(seq)/length_read)*coverage.
    """

    number_of_reads = int(len(seq)/length_reads) * coverage
    starting_pos = random.sample(range(0, len(seq)-length_reads+1), number_of_reads)
    reads = []

    for num in starting_pos:
        reads.append(seq[num:num+length_reads])

    if verbose == True:
        # This part has the only aim to show some stats on the reads
        y = [0 for i in range(0,len(seq)+1)]
        for i in starting_pos:
            for j in range(i, i+length_reads+1):
                y[j] += 1 
        sns.set_theme(style="darkgrid")
        sns.lineplot(y)
        print(f"There are {y.count(0)} bases that have 0 coverage.")

    return reads

@njit
def np_score(align_list: list, zeros = True)->int:
    length = len(align_list)
    cnt = 0

    for i in align_list:
        if i == 0:
            cnt += 1

    if zeros:
        return cnt
    else:
        return length-cnt

# @njit
def np_align_func(seq_tuple:tuple, match:int = 3, mismatch:int = -2):
    """Do something
    """

    # Initialization of output, and since the the func return only one vaue for each the check will be if the saved value is greater or not
    score = 0
    diff = 0
    switch = False

    # String are transponed in number using the ord() func, tha aim is to use mathematichs instead of comparison between strigns.
    seq_one = READS[seq_tuple[0]]
    seq_one = np.array([ord(c) for c in seq_one])
    # print(f"sequence one inside th enp_align_func {seq_one}")
    seq_two = READS[seq_tuple[1]]
    seq_two = np.array([ord(c) for c in seq_two])
    # print(f"sequence two inside th enp_align_func {seq_two}")

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

def tensor_score(ord_sequence:list, zeros:True)->int:
    raise NotImplemented

def tensor_align(sequence_one:str, sequence_two:str):
    raise NotImplemented

def parallel_eval(num_reads:int):
    comb = list(combinations(range(num_reads),2))
    print(comb)

    with Pool(processes=5)as pool:
        results = pool.imap(np_align_func, comb)
    
    return results

def eval_allign_np(reads:list, par:list = [3, -2, -40, -40]) -> list:
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

    # with Pool() as pool:
    #     results = pool.imap_unordered(np_align_func, (reads_1, reads_2))
    
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
                        weigth_matrix[j, i] = alignment[0]
                        weigth_matrix[i, j] = float(f"{0}.{abs(alignment[1])}")
                    
                    else:
                        weigth_matrix[i, j] = alignment[0]
                        weigth_matrix[j, i] = float(f"{0}.{abs(alignment[1])}")

                    
        visited.popleft()
    print(f"Done matrix {len(weigth_matrix)}x{len(weigth_matrix)}")
    return weigth_matrix

def f__(tuple):
    return tuple[0]*tuple[1]

# def main():

#     READS = comstum_reads(seq[:300], length_reads = 100, coverage = 3)

#     comb = list(combinations(range(len(READS)),2))
#     print(f"Combinations: {comb}")

#     with Pool(processes=3) as p:
#         results = p.starmap(f__, comb)
#         p.close()
#         p.join()

#     print(results)
#     return None 

READS = comstum_reads(seq[:300], length_reads = 100, coverage = 3)
comb = list(combinations(range(len(READS)),2))

res = Parallel(n_jobs=-1)(delayed(np_align_func)(i)for i in comb)
print(res)

