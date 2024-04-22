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

def extracting_sequence(input_path:str, limit = 5000, format="fasta")->str:

    #Extracting the sequence from the fasta and selecting the lenght:
    seq = ""
    len_seq = 0
    for seq_record in SeqIO.parse(input_path, format="fasta"):
        seq += seq_record.seq.upper()
        len_seq += len(seq_record)
        if len_seq > limit:
            continue
    seq = seq[:limit]

    return str(seq)

def de_code(read:np.ndarray)->str:
    return "".join([chr(c) for c in read])

@jit(nopytho = True)
def uni_code(read:str)->np.ndarray:
    return np.array([ord(c) for c in read])

def custom_reads(seq: str, length_reads:int = 160, coverage:int = 5, verbose = False) -> list:
    """The function splits the sequence in input into reads.
    The splitting is done using random numbers, and the number of reads is given by: (len(seq)/length_read)*coverage.
    """
    len_sequence = (len(seq))
    number_of_reads = int(len_sequence/length_reads) * coverage
    starting_pos = random.sample(range(0, len_sequence - int(length_reads/2) + 1), number_of_reads)
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
def np_score(align_list: np.ndarray, zeros = True)->float:
    """This function is a replacement for the np function np.count_nonzero(), since inside the np_eval_function was needed to count the number of zeros (=matches).
    However this function raise an error when run with the numba decorator.
    """
    length = len(align_list)
    cnt = 0

    for i in align_list:
        if i == 0:
            cnt += 1

    if zeros:
        return float(cnt)
    else:
        return float(length-cnt)

@jit(nopython=True)
def np_align_func(seq_one:np.ndarray, seq_two:np.ndarray, match:int = 3, mismatch:int = -2) -> tuple:
    """This function is a replacement for the align function pirwise2.align.localms of the Bio library. This substitution has the aim of tackling the computational time of the
    eval_alignment function. In order to decrease the time, there was the need to create a compilable function with numba, which was also capable of being parallelised.
    As you can clearly see the function takes in input only the match and mismatch, because in this usage the gap introduction is useless.

    seq_one, seq_two = input sequences already trasformed in byte
    match, mismatch = integer value for the alignment

    Note: the mismatch should be negative
    Ex output: (12.0, 34, True)
    """

    # Initialization of outputs, the output is a tuple that contains: score of the alignment, a number indicating how the two reads align
    # and if the two sequences have been inverted in order during the process
    score = float(0)
    diff = 0
    switch = False

    # Since knowing which one is the longest is needed 
    if seq_one.shape[0] >= seq_two.shape[0]:
        max_lenght_seq = seq_one.shape[0]
        min_length_seq = seq_two.shape[0]

    else:
        switch = True
        max_lenght_seq = seq_two.shape[0]
        min_length_seq = seq_one.shape[0]
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

                if part_score >= score:
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

                
                if part_score >= score:
                    score = part_score
                    if cnt > 0:
                        diff = max_lenght_seq -i -1
                    else:
                        diff = -(min_length_seq -i -1)
                cnt += 1 

        if i == (num_iteration_int - 1) and alone:
            i += 1

            align_forw = seq_one[i-min_length_seq+1:(i+1)] - seq_two[-(i+1):]
            part_score = np_score(j)*match + np_score(j, zeros=False)*mismatch

            if part_score >= score:
                score = part_score
                diff = max_lenght_seq -i -1


    return (score, diff, switch)

def eval_allign_np(reads:list, par:list = [3, -2]) -> np.ndarray:
    """Funtion that evaulate the alignment

    reads: list of DNA sequences, each of the read is a string

    par: list of parameters to performe the alignment
    es (the examples represent the defoult parameters):
    match_score = 3,
    mismatch_penalty = -2,

    output:
    Matrix with the weigts (distances) between the reads (nodes)
    In this matrix there are both the scores of the alignment, recognizable for the tipical integer score (even if is a float point) and
    a flaot number (like 0.23) which is needed after to recompose the sequence; it indicates the overlapping bases.
    Ex:
        allignment score -> 2.0, 13.0, ...
        overlapping number -> 0.241, 0.61, 0.561, ...
            To avoid problem later with 0 a 1 digit is added for then remove it. So 12.30 become 12.301 but the corret indices are 12 and 30.

        These two numbers are link by the position in the matrix which are the trasposition
        Score 14.0 in position (1,5) --> 0.34 in position (5,1). Only the score position is referred
        to the direction of the edge.
        1 ---> 5 with allignment score 14 and read_1 is overlapped with read_5 in positions 34 (both included)

    Example of a matrix with three reads:

        | 1    | 2    | 3    
     1  | 0    |3.0   | 0.231 
     2  | 0.601|  0   | 23.0
     3  | 18.0 | 0.701|  0
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
                # print(f"Read_1 : {reads_1}")
                reads_2 = np.array([ord(p) for p in reads[j]])
                # print(f"Read_2 : {reads_2}")
                alignment = np_align_func(reads_1, reads_2, match = par[0], mismatch = par[1])

                if alignment[2]:
                    if alignment[1] > 0:
                        weigth_matrix[j, i] = alignment[0]
                        weigth_matrix[i, j] = float(f"{0}.{abs(alignment[1])}1")
                    
                    else:
                        weigth_matrix[i, j] = alignment[0]
                        weigth_matrix[j, i] = float(f"{0}.{abs(alignment[1])}1")

                else:
                    if alignment[1] > 0:
                        weigth_matrix[i, j] = alignment[0]
                        weigth_matrix[j, i] = float(f"{0}.{abs(alignment[1])}1")
                    
                    else:
                        weigth_matrix[j, i] = alignment[0]
                        weigth_matrix[i, j] = float(f"{0}.{abs(alignment[1])}1")

                    
        visited.popleft()
    print(f"Done matrix {len(weigth_matrix)}x{len(weigth_matrix)}")
    return weigth_matrix
