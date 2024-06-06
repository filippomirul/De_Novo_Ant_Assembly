import datetime
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
from numba import jit, prange
from Bio import pairwise2


def save_list(data:list, where:str)-> None:
    """Function for saving list files"""

    with open(where, "wb") as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    return None

def load_list(where:str)-> list:
    
    with open(where, 'rb') as f:
        data = pickle.load(f)
    return data

def __de_code__(read:np.ndarray)->str:
    return "".join([chr(c) for c in read])


def __uni_code__(read:str)->np.ndarray:
    return np.array([ord(c) for c in read], dtype=np.float32)


def parallel_coding(reads:list, number_cpus = 1, uni_coding=True):
    if uni_coding:
        reads = Parallel(n_jobs=number_cpus)(delayed(__uni_code__)(i)for i in reads)
        return reads
    else:
        reads = Parallel(n_jobs=number_cpus)(delayed(__de_code__)(i)for i in reads)
        return reads


def custom_reads(seq: str, res_path:str, length_reads:int = 160, coverage:int = 5, verbose = False, gap =False) -> list:
    """The function splits the sequence in input into reads.
    The splitting is done using random numbers, and the number of reads is given by: (len(seq)/length_read)*coverage.
    """
    len_sequence = (len(seq))
    number_of_reads = int(len_sequence/length_reads) * coverage
    starting_pos = random.sample(range(0, len_sequence - int(length_reads/2) + 1), number_of_reads)
    reads = []
    # print(f"Len starting position: {len(starting_pos)}")
    if gap:
        num_of_gap = random.randint(1, 4)
        # print(f"Number of gap: {num_of_gap}")
        new_starting_pos = []
        not_to_keep = []

        for j in range(num_of_gap):
            rand = random.randint(0, len_sequence - length_reads)
            # print(f"gap: {rand}")
            for i in starting_pos:
                if abs(i - rand) < length_reads:
                    not_to_keep.append(i)
        for i in starting_pos:
            if i not in not_to_keep:
                new_starting_pos.append(i)

        # print(f"Len new starting position: {len(new_starting_pos)}")
        starting_pos = new_starting_pos

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

    print(f"[{datetime.datetime.now()}]: There are {y.count(0)} bases that have 0 coverage.")

    if verbose:
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        save_path = res_path + "/coverage.png"

        plt.plot(y)
        plt.xlabel("Position")
        plt.ylabel("Coverage")
        plt.title("Coverage Plot")
        plt.savefig(save_path)

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
                        # The index diff is included
                        diff = max_lenght_seq -i -1
                    else:
                        # If the diff value is negative the second sequence if the one upstream
                        # The index diff is included
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


@jit(nopython=True)
def __split_align__(tuple:tuple):
    """This function crates a tuple for each edge -> (from node, to node, score, diff, distance)"""
    
    reads = tuple[0]
    epoch = tuple[1] # row
    distance_vector = np.zeros(len(reads))

    # molt = MEAN_LENGTH *100
    molt = 1
    comparison = reads[epoch]
    out = []
    for i in range(len(reads)):
        if i == epoch:
            continue
        else:
            alignment = __np_align_func__(reads[i], comparison)

            if alignment[0] > 0:

                if alignment[2]:
                    if alignment[1] > 0:
                        out.append(( epoch, i, alignment[0], alignment[1], molt/alignment[0]))
                        distance_vector[i] = molt/alignment[0]
                        # distance_vector[i] = len(reads[i]) - abs(alignment[2] * 2) +1 
                    else:
                        distance_vector[i] = molt/alignment[0]
                        # distance_vector[i] = len(reads[i]) - abs(alignment[2] * 2) +1

                else:
                    if alignment[1] > 0:
                        distance_vector[i] = molt/alignment[0]
                        # distance_vector[i] = len(reads[i]) - abs(alignment[2] * 2) +1
  
                    else:
                        out.append(( epoch, i, alignment[0], alignment[1], molt/alignment[0]))
                        distance_vector[i] = molt/alignment[0]
                        # distance_vector[i] = len(reads[i]) - abs(alignment[2] * 2) +1



    return (out, distance_vector)


def links_formation(links:list, cpu=2)->list:

    return Parallel(n_jobs=cpu)(delayed(__split_align__)(i)for i in [(links,j) for j in range(len(links))])

################
"""
Same as for the ants for retriving the score, keep edges with a list of list to reduce the score retriving. Since here
is not needed the score but the difference for the final alignment wich is the same thing as the score but instead at position 2 is at 3.

"""
################


def final_consensus(path:list, reads:list, positions:list, length:int = 50000, max_coverage: int = 16) -> np.ndarray:
    """This function create a matrix and write down the numbers resembling the path found by the ants algorithm
    """
    #Diff is included

    cons_matrix = np.zeros((max_coverage, length), dtype=np.float32)
    cum_dif = 0
    # adding = np.zeros((max_coverage, int(length/100)), dtype=np.float32)

    for i,j in path:
        # Here i,j represent the edge of the graph, to retrive not the score but the alignment
        # the function needs the opposite position where there are these informations matrix[j][i]
        
        num = str(positions[j][i]).split(".")
        dif = int(num[0])
        

        if cons_matrix[0,0] == 0:
            # This first part is for starting the writing of the matrix
            # print(len(reads[i]))
            for pos in range(0, len(reads[i])):
                cons_matrix[0, pos] = reads[i][pos]
                
            cum_dif += dif
            temp = 0

            for p in range(cum_dif , cum_dif + len(reads[j])): 
                # if cons_matrix[1,pos] > 0:
                #     cons_matrix = np.append(cons_matrix, adding, 1)
                cons_matrix[1, p] = reads[j][temp]
                temp += 1

        else:
            
            # There is a check if the initialized matrix is big enough to contain all tha bases, columns wise
            if cons_matrix.shape[1] < (cum_dif + len(reads[j])*2):
                # print(f"mat_shape:{cons_matrix.shape[1]}, tot: {cum_dif + len(reads[j])*2}, cum_dif:{cum_dif}, len_reads: {len(reads[j])}")
                cons_matrix = np.append(cons_matrix, np.zeros((cons_matrix.shape[0], int(length/100)), dtype=np.float32), 1)
                # print("Here")
            else:
                cum_dif += dif
                temp = 0
                for pos in range(cum_dif, cum_dif + len(reads[j])): 
                    row = 0
                    while cons_matrix[row, pos] > 0:
                        row += 1
                    # There is a check if the initialized matrix is big enough to contain all tha bases, row wise
                        if row == cons_matrix.shape[0]:
                            cons_matrix = np.append(cons_matrix, np.zeros((2, cons_matrix.shape[1]), dtype=np.float32) ,0)
                            # print("Here")
                    cons_matrix[row, pos] = reads[j][temp]
                    # print(f"Position: {(row, pos)} is {reads[j][temp]}")
                    # print(f"So {cons_matrix[row, pos]}")
                    temp +=1
    # print(f"cum_dif: {cum_dif}")
    return cons_matrix


def __re_build__(cons_matrix:np.ndarray)->str:
    
    dictionary = "ATCG"
    cons_seq = ""
    for i in range(0, cons_matrix.shape[1]):
        base = [x for x in cons_matrix[:,i] if x > 0]
        if base == []:
            return cons_seq
        ind = []
        tot_bases = 0
        for num in [ord(c) for c in dictionary]:
            occur = base.count(num)
            ind.append(occur)
            tot_bases += occur
        more_frequent = ind.index(max(ind))
        # TODO stats
        cons_seq += dictionary[more_frequent]

    return cons_seq


def join_consensus_sequence(consensus_matrix:np.ndarray, cpus:int)-> str:
    "This function is just to implement the use of multiples core for recostructing the final sequence."

    step = int(consensus_matrix.shape[1]/cpus)
    cnt = 0
    partials = []
    for i in range(step, consensus_matrix.shape[1] + step, step):

        partials.append((cnt, i))
        cnt += step
        if cnt == step:
            cnt += 1

    sub_parts = [consensus_matrix[:,i:j] for i,j in partials]

    res = Parallel(n_jobs=cpus)(delayed(__re_build__)(i) for i in sub_parts)
    print(res)
    return "".join(res)


def __printing_alignment__(seq_1:str, seq_2:str)->str:

    al = __np_align_func__(seq_1, seq_2)

    if al[2]:
        if al[1] > 0:
            dif = "-" * al[1]
            seq_1 = dif + seq_1
            add = "-" * abs(len(seq_1) - len(seq_2))
            seq_2 = seq_2 + add
        else:
            dif = "-" * abs(al[1])
            seq_2 = dif + seq_2
            add = "-" * abs(len(seq_1) - len(seq_2))
            seq_1 = seq_1 + add


    else:
        if al[1] > 0:
            dif = "-" * al[1]
            seq_2 = dif + seq_2
            add = "-" * abs(len(seq_1) - len(seq_2))
            seq_1 = seq_1 + add

        else:
            dif = "-" * abs(al[1])
            seq_1 = dif + seq_1
            add = "-" * abs(len(seq_1) - len(seq_2))
            seq_2 = seq_2 + add
    
    return seq_1, seq_2

# TODO check
def efficiency(reference:str, recostructed_sequence:str, cpus = 2)-> int:
    """Evaluate the efficiency of the recostruction in testing (when --test is parsed)
    Transform the sequences in np array to do the alignment and then return the percentage of corrected based aligned.
    """
    ord_list = parallel_coding([reference, recostructed_sequence], number_cpus=cpus)

    al = __np_align_func__(seq_one=ord_list[0], seq_two=ord_list[1])

    num_epoch = min(len(reference), len(recostructed_sequence))

    cnt = 0
    if al[2]:
        if al[1] > 0:
            for i in range(0, num_epoch-al[1]):
                if recostructed_sequence[i+ al[1]] == reference[i]:
                    cnt += 1
            # recostructed_sequence[al[1]:] upstream
        else:
            dif = abs(al[1])
            for i in range(0, num_epoch- dif):
                if reference[i + dif] == recostructed_sequence[i]:
                    cnt += 1
    else:
        if al[1] > 0:
            for i in range(0, num_epoch- al[1]):
                if reference[i + al[1]] == recostructed_sequence[i]:
                    cnt += 1
        else:
            dif = abs(al[1])
            for i in range(0, num_epoch - dif):
                if recostructed_sequence[i + dif] == reference[i]:
                    cnt += 1
    
    return cnt/len(reference)


def out_files(ref: str, reconstructed_seq: str, out_path:str):
    """Files: fasta with the assembly sequence, variants tsf and stats file.
    Txt file for training
    """

    final_alignment = pairwise2.align.localms(reconstructed_seq, ref, 3,-1,-30,-30)[0]

    out_path = out_path + "/assembly_results.txt"

    with open(out_path, "w") as results_file:
        results_to_write = ["Reference sequence\n", ref, "\n", "\n","Sequence reconstructed by ACO assembler:\n", reconstructed_seq, "\n", "\n",
                        "Length of the reconstructed sequence:\n", str(len(reconstructed_seq)), "\n", "\n", "Score of the alignment:\n", str(final_alignment[2]), "\n", "\n"]
        results_file.writelines(results_to_write)
        local_ref = final_alignment[1]                                                                                          # the ref sequence, as in the local alignment
        local_reconstructed = final_alignment[0]                                                                                # the reconstructed sequence, as in the local alignment
        splitted_local_ref = [local_ref[i:i + 100] for i in range(0, len(local_ref), 100)]                                      # the first, split every 100pb
        splitted_local_reconstructed = [local_reconstructed[i:i + 100] for i in range(0, len(local_reconstructed), 100)]        # the second, split every 100bp

        mism_counter = 0
        for i in range(len(local_reconstructed)):                                                                               # small script to count number of matches
            if local_reconstructed[i] == local_ref[i]:
                mism_counter = mism_counter + 1
        perc_of_matches = (mism_counter / len(local_ref))*100
        results_file.writelines(["Percentage of matches:\n", str(perc_of_matches), "\n", "\n"])

        results_file.writelines (["Local alignment:\n", "\n"])                                                                  # small script to print decently the local alignment
        for i in range(len(splitted_local_ref)):
            results_file.write("reference")
            results_file.write("\n")
            results_file.write(splitted_local_ref[i])
            results_file.write("\n")
            results_file.write(splitted_local_reconstructed[i])
            results_file.write("\n")
            results_file.write("reconstructed")
            results_file.write("\n")
            results_file.write("\n")

