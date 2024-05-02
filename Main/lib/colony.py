from Bio import SeqIO
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import collections
from numba import jit, prange

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


def __de_code__(read:np.ndarray)->str:
    return "".join([chr(c) for c in read])


def __uni_code__(read:str)->np.ndarray:
    return np.array([ord(c) for c in read])


def parallel_coding(reads:list, number_cpus:int, uni_coding=True):
    if uni_coding:
        reads = Parallel(n_jobs=number_cpus)(delayed(__uni_code__)(i)for i in reads)
        return reads
    else:
        reads = Parallel(n_jobs=number_cpus)(delayed(__de_code__)(i)for i in reads)
        return reads


def custom_reads(seq: str, res_path:str, length_reads:int = 160, coverage:int = 5, verbose = False) -> list:
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

    y = [0 for _ in range(0, len_sequence + 1)]
    for i in starting_pos:
        if i < len_sequence- length_reads:
            for j in range(i, i + length_reads + 1):
                y[j] += 1
        else:
            for j in range(i, len_sequence + 1):
                y[j] += 1

    print(f"There are {y.count(0)} bases that have 0 coverage.")


    if verbose:

        plt.plot(y)
        plt.xlabel("Position")
        plt.ylabel("Coverage")
        plt.title("Coverage Plot")
        plt.savefig(res_path)

    return reads


@jit(nopython = True)
def __np_score__(align_list: np.ndarray, zeros = True)-> int:
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
def __np_align_func__(seq_one:np.ndarray, seq_two:np.ndarray, match:int = 3, mismatch:int = -2) -> tuple:
    """This function is a replacement for the align function pirwise2.align.localms of the Bio library. This substitution has the aim of tackling the computational time of the
    eval_alignment function. In order to decrease the time, there was the need to create a compilable function with numba, which was also capable of being parallelised.
    As you can clearly see the function takes in input only the match and mismatch, because in this usage the gap introduction is useless (for the moment).
    This function return only the BEST alignment.

    seq_one, seq_two = input sequences already trasformed in integers by ord function
    match, mismatch = integer value for the alignment

    Note: the mismatch should be negative
    Output: A tuple with the alignment score, a number that resamble the shift of the alignemnt, and a boolean which indicates if the order
        of the input has been inverted or not. This last element is essential to retrive the order, so which of the two will be place before the other one.
    Ex output: (12.0, 34, True)
    """

    # Initialization of outputs, the output is a tuple that contains: score of the alignment, a number indicating how the two reads align
    # and if the two sequences have been inverted in order during the process
    score = 0
    diff = 0
    switch = False

    # Since knowing which one is the longest is needed for the epoch
    if seq_one.shape[0] >= seq_two.shape[0]:
        max_lenght_seq = seq_one.shape[0]
        min_length_seq = seq_two.shape[0]

    else:
        switch = True
        max_lenght_seq = seq_two.shape[0]
        min_length_seq = seq_one.shape[0]
        seq_one, seq_two = seq_two, seq_one
    
    # Number of iterations (N + n -1)/2 because each iteration is producing two alignment one confronting the sequenvces from the forward
    # and from the backward
    num_iteration_int = (max_lenght_seq + min_length_seq - 1) // 2
    num_iteration = (max_lenght_seq + min_length_seq - 1) / 2
    alone = False # There could be needed an extra iteration if this (N + n -1)/2 is odd 

    if num_iteration > num_iteration_int:
        alone = True

    for i in range(num_iteration_int):
        if i < min_length_seq:

            # Back/Forward alignments, only overlapping bases are being used
            align_forw = seq_one[:(i+1)] - seq_two[-(i+1):]
            align_back = seq_two[:(i+1)] - seq_one[-(i+1):]

            cnt = 0
            for j in align_forw, align_back:
                part_score = __np_score__(j)*match + __np_score__(j, zeros=False)*mismatch

                if part_score >= score:
                    score = part_score
                    if cnt > 0:
                        # If the diff value is positive the first sequence is upstream
                        diff = max_lenght_seq -i -1
                    else:
                        # If the diff value is negative the second sequence if the one upstream
                        diff = -(min_length_seq -i -1)
                cnt += 1
        
        if i >= min_length_seq:
            align_forw = seq_one[i-min_length_seq+1:(i+1)] - seq_two[-(i+1):]
            align_back = seq_one[-(i+1):-(i-min_length_seq+1)] - seq_two[:(i+1)]

            cnt = 0
            for j in align_forw, align_back:
                part_score = __np_score__(j)*match + __np_score__(j, zeros=False)*mismatch

                
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
            part_score = __np_score__(j)*match + __np_score__(j, zeros=False)*mismatch

            if part_score >= score:
                score = part_score
                diff = max_lenght_seq -i -1


    return (score, diff, switch)


def eval_allign_np(reads:list, par:list = [3, -2]) -> np.ndarray:
    """Funtion that evaulate the alignment

    reads: list of DNA sequences, each of the read is a list of integers that resemble the real sequence

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
        overlapping number -> 24.1, 6.1, 56.1, ...
            To avoid problem later with 0 a 1 digit is added for then remove it. So 0.30 become 0.301 but the corret indices are 12 and 30.

        These two numbers are link by the position in the matrix which are the trasposition
        Score 14.0 in position (1,5) --> 34.1 in position (5,1). Only the score position is referred
        to the direction of the edge.
        1 ---> 5 with allignment score 14 and read_1 is overlapped with read_5 in positions 34 (both included)

    Example of a matrix with three reads:

        | 1    | 2    | 3    
     1  | 0    |3.0   | 23.1 
     2  | 60.1 |  0   | 23.0
     3  | 18.0 | 70.1 |  0
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
                alignment = __np_align_func__(reads[i], reads[j], match = par[0], mismatch = par[1])

                if alignment[0] > 0:

                    if alignment[2]:
                        # Swithch happend so reads[j] is longer then reads[i]
                        if alignment[1] > 0:
                            # cond = first sequence is upstream
                            weigth_matrix[j, i] = alignment[0]
                            weigth_matrix[i, j] = float(f"{abs(alignment[1])}.1")
                        
                        else:
                            # cond = first sequence is downstream
                            weigth_matrix[i, j] = alignment[0]
                            weigth_matrix[j, i] = float(f"{abs(alignment[1])}.1")

                    else:
                        if alignment[1] > 0:
                            # cond = first sequence is upstream
                            weigth_matrix[i, j] = alignment[0]
                            weigth_matrix[j, i] = float(f"{abs(alignment[1])}.1")
                        
                        else:
                            # cond = first sequence is downstream
                            weigth_matrix[j, i] = alignment[0]
                            weigth_matrix[i, j] = float(f"{abs(alignment[1])}.1")

                else:
                    continue

                    
        visited.popleft()
    return weigth_matrix


def eval_nonzeros(graph:np.ndarray)-> int:

    cnt = 0
    for i in range(len(graph)):
        cnt += np.count_nonzero(graph[i])

    return cnt


def final_consensus(path:list, reads:list, positions:list, length:int, max_coverage: int = 16, verbose:bool = False) ->str:
    """This function create a matrix and write down the numbers resembling the path found by the ants algorithm
    """
    #Diff is included

    cons_matrix = np.zeros((max_coverage, length))
    leng = len(cons_matrix[0])
    cum_dif = 0
    adding = np.zeros((max_coverage, int(length/100)))

    for i,j in path:
        # Here i,j represent the edge of the graph, to retrive not the score but the alignment
        # the function needs the opposite position where there are these informations matrix[j][i]

        num = str(positions[j][i]).split(".")
        dif = int(num[0])

        if cons_matrix[0,0] == 0:
            # This first part is for starting the writing of the matrix
            
            for pos in range(0, len(reads[i])):
                if cons_matrix[0, pos] != 0:
                    cons_matrix = np.append(cons_matrix, adding, 1)
                cons_matrix[0, pos] = reads[i][pos]
            cum_dif += dif
            temp = 0
            for p in range(cum_dif + 1, cum_dif + len(reads[j]) + 1): # added +1 in last mod
                if cons_matrix[1,pos] != 0:
                    cons_matrix = np.append(cons_matrix, adding, 1)
                cons_matrix[1, p] = reads[j][temp]
                temp += 1

        else:
            # There is a check if the initialized matrix is big enough to contain all tha bases, columns wise
            if cons_matrix.shape[1] < cum_dif + len(reads[j])*2:
                cons_matrix = np.append(cons_matrix, adding, 1)
            else:
                cum_dif += dif
                temp = 0
                for pos in range(cum_dif + 1, cum_dif + len(reads[j]) + 1): # added +1 in last mod
                    row = 0
                    while cons_matrix[row, pos] >= 1:
                        row += 1
                    # There is a check if the initialized matrix is big enough to contain all tha bases, row wise
                        if row == cons_matrix.shape[0]:
                            cons_matrix = np.append(cons_matrix, np.zeros((2, cons_matrix.shape[1])) ,0)
                    cons_matrix[row, pos] = reads[j][temp]
                    temp +=1

    return cons_matrix


def __re_build__(cons_matrix:np.ndarray)->str:
    
    dictionary = "ATCG"
    cons_seq = ""
    for i in range(0, len(cons_matrix)):
        base = [x for x in cons_matrix[:,i] if x > 0]
        if base == []:
            return cons_seq
        ind = []
        for num in [ord(c) for c in dictionary]:
            ind.append(base.count(num))
        more_frequent = ind.index(max(ind))
        # TODO stats
        cons_seq += dictionary[more_frequent]

    return cons_seq


def join_consensus_sequence(consensus_matrix:np.ndarray, cpus:int)-> str:
    "This functio is just to implement the use of multiples core for recostructing the final sequence."

    step = int(len(consensus_matrix)/cpus)
    cnt = 0
    partials = []
    for i in range(step, len(consensus_matrix) + step, step):
        partials.append((cnt,i))
        cnt += i
    sub_parts = [consensus_matrix[:,i:j] for i,j in partials]
    
    results = Parallel(n_jobs=cpus)(delayed(__re_build__)(i) for i in sub_parts)

    return "".join(results)


def out_files(path_out:str ,reads:list, candidate:list, matrix:list):
    """Files: fasta with the assembly sequence, variants tsf and stats file.
    Txt file for training
    """
    
    # Final results and final consensus sequence
    c = [(i.element[0], i.element[1]) for i in candidate]
    d = final_consensus(c, reads, length=5000, positions = matrix)

    al = function()

    # Writing the results:
    ll = []
    ll.append("Thr first line is the reconstructed seq, while the second is the real sequence:\n")
    cnt=0
    for i in range(50,len(al[0]),50):
        ll.append(str(al[0][cnt:i]))
        ll.append("\n")
        ll.append(str(al[1][cnt:i]))
        ll.append("\n\n")
        cnt += 50

    ll.append("\n")
    ll.append("Score of the allignment after the reconstruction:\n")
    ll.append(str(al[2]))
    ll.append("\nThe percentage of macht in the allignment is:")
    ll.append("\n")

    cnt = 0
    for i in range(len(al[0])):
        if al[0][i] == al[1][i]:
            cnt += 1
    ll.append(str(cnt/len(seq))) 

    if not os.path.exists(path_out):
        os.makedirs

    new_file = open(path_out, "w")
    new_file.writelines(ll)
    new_file.close()

    return None

