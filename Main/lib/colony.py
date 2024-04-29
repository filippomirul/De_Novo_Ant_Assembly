from Bio import SeqIO
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from random import Random
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


def de_code(read:np.ndarray)->str:
    return "".join([chr(c) for c in read])


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
        plt.savefig("C:\\Users\\filoa\\OneDrive\\Desktop\\Programming_trials\\Assembler")

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


@jit(nopython = True)
def __prepare_simpl_intup__(matrix:np.ndarray)->list:
    """Is needed to set up the parallelization, divide the matrix in columns
    """
    # output = (array, len_array, column)

    len_arrray = matrix.shape[0]
    my_list = []

    for i in range(len(matrix)):
        my_list.append((list(matrix[:,i]), len_arrray, i))

    return my_list


def __matrix_selection__(input_tuple:tuple, cut_off = 0.2)->list:
    """Value the distibution of the columns, in this way select the ones
    above the third quantile
    """

    # Init
    array = input_tuple[0]
    array_len = input_tuple[1]
    column = input_tuple[2]

    # DO DO
    links = [x for x in array if (x > 0) and (str(x).split(".")[1] == "0")]
    chosen = sorted(links)[-(int(len(links)*cut_off)):]
    dissmissable_links = []

    for i in links:
        if i not in chosen:
            dissmissable_links.append((array.index(i), column))

    return dissmissable_links


def __matrix_sempl__(matrix:np.ndarray, dissmissable_links:list) -> np.ndarray:
    """Changes the occurrencie valued as unprobable in zeros, eraising in this way the link
    """
    for epoch in dissmissable_links:
        for i,j in epoch:
            matrix[i,j] = 0. 
            matrix[j,i] = 0.

    return matrix


def graph_semplification(graph:np.ndarray, cores:int)->np.ndarray:
    
    list_of_tuple = __prepare_simpl_intup__(graph)

    indeces_to_cut = Parallel(n_jobs = cores)(delayed(__matrix_selection__)(i) for i in list_of_tuple)

    matrix = __matrix_sempl__(graph, dissmissable_links=indeces_to_cut)

    return matrix


def eval_nonzeros(graph:np.ndarray)-> int:

    cnt = 0
    for i in range(len(graph)):
        cnt += np.count_nonzero(graph[i])

    return cnt

