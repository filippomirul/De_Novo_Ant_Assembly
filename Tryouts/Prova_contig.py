from node2vec import Node2Vec 
import networkx as nx
from scipy.io import loadmat
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import datetime
from numba import jit
from Bio import SeqIO
import random
from scipy.cluster.hierarchy import dendrogram
from seaborn import violinplot

print(f"[{datetime.datetime.now()}]")


def extracting_sequence_from_data(input_path:str, limit = 5000)->str:

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


def __uni_code__(read:str)->np.ndarray:
    return np.array([ord(c) for c in read], dtype=np.float32)


def parallel_coding(reads:list, number_cpus = 1, uni_coding=True)->list:
    if uni_coding:
        reads = Parallel(n_jobs=number_cpus)(delayed(__uni_code__)(i)for i in reads)
        return reads
    else:
        raise NameError

def custom_reads(seq: str, res_path:str, length_reads:int = 160, coverage:int = 5, verbose = False, gap =False, num_gap = None) -> list:
    """The function splits the sequence in input into reads.
    The splitting is done using random numbers, and the number of reads is given by: (len(seq)/length_read)*coverage.
    """
    len_sequence = (len(seq))
    number_of_reads = int(len_sequence/length_reads) * coverage
    starting_pos = random.sample(range(0, len_sequence - int(length_reads/2) + 1), number_of_reads)
    reads = []
    # print(f"Len starting position: {len(starting_pos)}")
    if gap:
        if num_gap != None:
            num_of_gap = num_gap
        else:
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
        if os.path.exists(res_path):
            save_path = res_path + "/coverage.png"
        else:
            save_path = os.getcwd() + "/coverage.png"

        plt.plot(y)
        plt.xlabel("Position")
        plt.ylabel("Coverage")
        plt.title("Coverage Plot")
        plt.show()
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

import os

#####
"""
The following can be parallelized parsing rows of the matrix. However multiples links has the possibility to be formed.
Good thing, since can be further linked and lowered the complexity. Therefore to not incurr in errors or bugs
the presence of cross links between them has to be checked.
"""
####

def select_high_score_overlap(edges:list, threshold= 0.70)->list:
    """This function search for two consecutive reads with most of them overlapping and a third,
    which overlap with the fisrt with o relative low score.
    Tha aim is to link the fisrt and the third with a high score, in order to lower the complexity of the graph.
    """
    return None


def replace_high_overlap():
    """ Here the selected nodes are replace with others high score links, informations has to be kept, so variants will be
    stored in a file for later use when consensus sequence will be reconstruct.
    """
    return None


jit(nopython=True)
def __list_selection_(row_edges:list, cut_off = 0.1)->list:
    """Keep only those edges with a score above a certain percentile, default value keep only above 90%
    Input: [[(0, 3, 45, -33, 11), (23, 45, 79, -123, 0.23), ...]
    """
    
    weights = [x[2] for x in row_edges]
    limit = sorted(weights)[-(int(len(row_edges) * cut_off))]
    keep_links = []

    for i in range(len(weights)):
        if weights[i] > limit:
            keep_links.append(row_edges[i])


    return keep_links


def edge_selection(edges:list, cpu=2)->list:
    """Performe edge selection only on the weight between two nodes
    Input: [[(0, 3, 45, -33, 11), (23, 45, 79, -123, 0.23), ...], [(), (), ..], ... ]
    """

    res = Parallel(n_jobs=cpu)(delayed(__list_selection_)(edges[i][0]) for i in range(len(edges)))

    # print(len(res))
    # print(len(edges))

    dist_matrix = np.vstack((edges[0][1], edges[1][1]))

    select_edges = []

    for i in range(len(res)):
        for j in res[i]:
            select_edges.append(j)
        if i == 0 or i == 1:
            continue
        else:
            dist_matrix = np.vstack((dist_matrix, edges[i][1]))

    return select_edges, dist_matrix


seq = extracting_sequence_from_data("C:\\Users\\filoa\\Desktop\\Programming_trials\\Assembler\\Main\\Data\\GCA_014117465.1_ASM1411746v1_genomic.fna", limit=5000)

reads = custom_reads(seq, res_path="", length_reads=150, gap=True, verbose=True, coverage=20, num_gap=2)

reads = parallel_coding(reads)

len_reads = [len(i) for i in reads]

MEAN_LENGTH = np.mean(len_reads)

links = links_formation(reads, cpu=6)

# fig, axis = plt.subplot(1,4)
# cnt = 3
# x = []
# t = 0
# for i in range(len(links)):
#     x.append([])
#     for j in links[i][0]:
#         x[i].append(j[2])
#     axis[0,i].plt.title(f"Row number {i}")
#     axis[0,i].plt.violinplot(x[i])
#     if i == cnt:
#         plt.show()
#         break


# plt.close()
# plt.title("all links")
# plt.violinplot(x)
# plt.show()

y = []
for i in range(len(links)):
    t = []
    for j in links[i][0]:
        t.append(j[2])
    y.append(np.median(np.array(t)))
lim = np.quantile(y, 0.05)

print(f"{datetime.datetime.now()}############################")
print(len(y))
print(len([i for i in y if i < lim]))

plt.close()
plt.title("Median of rows")
plt.violinplot(y, quantiles=[0.05], showextrema=False)

plt.show()

# # Complitly statistical 
# import statistics

edges, dist_matrix = edge_selection(links, cpu=6)

# # print(edges[0])


w = []
for i in range(len(edges)):
    w.append(edges[i][2])

plt.close()
plt.title("all edges after selection")
plt.violinplot(w, quantiles=[0.5, 0.9])
plt.show()

# v = []
# for i in range(len(edges)):
#     h = []
#     for j in edges[i]:
#         h.append(j[2])
#     v.append(np.median(np.array(h)))

# plt.close()
# plt.title("Median after selection")
# plt.violinplot(v, quantiles=[0.3, 0.5], showextrema=False)
# plt.show()

def has_quality(edges_out_node:list, threshold:int):
    
    scores = [i[2] for i in edges_out_node]
    cnt = 0
    for i in scores:
        if i > threshold:
            cnt +=1

    if cnt/len(scores) <= 0.1:
        return True
    else:
        return False
    
import statistics

def contig_search(edges:list, threshold=0.6, quantile=0.3)->list:
    """This function semplify further the number of edges, try to find those nodes with very few and low in score link.
    Those will be used to divide the genome to forme contigs in order to further tackle the complexity problem.
    Morover doing so the ant algorithm will performe better and can introduce gaps.
    Gaps will be define downstream with repetition of N (50 times).

    Input:
        edges: edge list [[(), (), ... ], [], [] ... ]  # input is row wise, first row is all the edges exiting from node 0 etc
        threshold:
        quantile:
    Output:

    """

    mean_vector = []

    for i in edges:
        if len(i) < 1:
            continue
        else:
            mean_vector.append(statistics.mean([j[2] for j in i]))

    for i in range(len(edges)):
        if mean_vector[i] < np.quantile(mean_vector, threshold):
            return None






    mean_vector = np.array(mean_vector)
    quantile_limit = np.quantile(mean_vector, quantile)
    single_limit = np.quantile(mean_vector, 0.6)

    for i in range(len(edges)):
        if mean_vector[i] < quantile_limit:
            r = __list_selection_(edges[i])
            links_to_keep = []
            for j in range(len(r)):
                if r[j][2] > threshold:
                    links_to_keep.append(r[j])

            edges[i] = links_to_keep

    possible_gaps = []




    # for i in range(len(edges)):
    #     if len(edges[i]) < 1:
    #         possible_gaps.append(True)
    #     # elif np.median(np.array([i[2] for i in edges[i]])) < single_limit:
    #     elif has_quality(edges[i], single_limit):
    #         possible_gaps.append(True)
    #     else:
    #         continue

    # possible_gaps = [i for i in range(len(possible_gaps)) if possible_gaps[i] == True]

    return possible_gaps, edges


# plt.hist(means, bins=int(len(means)/10))

# problem when dealing with empty list

# for i in range(len(hard_links)):
#     keep = False
#     for j in hard_links[i]:
#         if j[2] > threshold:
#             keep = True
#     b.append(keep)
# print(b.count(False))
# c = [i for i in range(len(b)) if b[i] == False]
# print(c)

# pos , edge = contig_search(edges)

# sss = __uni_code__(seq)

# for i in pos:
#     print(__np_align_func__(reads[i], sss)[1])

from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS

def cluster(distance_matrix: np.ndarray, linkage="average", dist_threshold= 0):
    "This part performe the hierarchical clustering to the matrix distance between each point/node"

    cluster = AgglomerativeClustering(compute_distances=True, distance_threshold = dist_threshold,  n_clusters=None,
                                        metric="precomputed", linkage=linkage).fit(distance_matrix)
    
    return cluster


def plot_dendrogram(model, plot_save_path="dendrogram_cluster", **kargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0

        for child_idx in merge:

            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    plt.title("Hierarchical Clustering Dendrogram")
    dendrogram(linkage_matrix, **kargs)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

    return None

model = cluster(dist_matrix)

plt.close()
plt.hist(model.labels_)
plt.show()

plot_dendrogram(model, truncate_mode = "level", p=2)

model2 = MDS(n_components=64, dissimilarity="precomputed", n_jobs=-1, max_iter=len(reads), metric = False)
mm = model2.fit_transform(dist_matrix)

print(dist_matrix)


dd = PCA(n_components=2)
ff = dd.fit_transform(mm)

# print(mm[0])


x = [i[0] for i in ff]
y = [i[1] for i in ff]

plt.scatter(x,y)
plt.show()