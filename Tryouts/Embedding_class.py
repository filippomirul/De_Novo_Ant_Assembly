# python

import os
from node2vec import Node2Vec
import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


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

def custom_reads(seq: str, res_path:str, length_reads:int = 160, coverage:int = 5, verbose = False, gap =False, num_gap = None, pos = False) -> list:
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

    if pos:
        return reads, starting_pos

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


jit(nopython=True)
def __list_selection_(row_edges:list, cut_off = 0.05)->list:
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

    len_edges = [len(i) for i in res]

    plt.title("Number of links")
    plt.scatter(POS, len_edges)
    plt.show()

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


seq = extracting_sequence_from_data("C:\\Users\\filoa\\Desktop\\Programming_trials\\Assembler\\Main\\Data\\GCA_014117465.1_ASM1411746v1_genomic.fna",
                                    limit=8000)

reads, POS = custom_reads(seq, res_path="", length_reads=80, gap=True, verbose=True, coverage=20, num_gap=4, pos = True)


print(len(reads))

reads = parallel_coding(reads,number_cpus=3)

print(f"[{datetime.datetime.now()}]: Done reads encoding")

len_reads = [len(i) for i in reads]

MEAN_LENGTH = np.mean(len_reads)
print(f"[{datetime.datetime.now()}]: Done stuff")


links = links_formation(reads, cpu=3)
print(f"[{datetime.datetime.now()}]: Done links")

v = []
for i in range(len(links)):
    h = []
    for j in links[i][0]:
        h.append(j[2])
    v.append(np.median(np.array(h)))

plt.title("Median all points")
plt.scatter(POS, v)
plt.show()


edges, dist_matrix = edge_selection(links, cpu=3)
print(f"[{datetime.datetime.now()}]: Done reads selection")

class Graph_embed():
    """This is an embedding class
    """
    
    def __init__(self, edges, distance_matrix):
        
        self.edges = [(x[0], x[1], x[2]) for x in edges] # change score with distance 2-4
        self.graph = nx.DiGraph()
        self.dist_matrix = distance_matrix

        self.cluster = None
        self.embedding = None
        self.x_cord = None
        self.y_cord = None


    def embed(self,  dimensions=30, walk_length=300, num_walks=4000, workers=1, window=10, min_count=1, batch_words=4):
        """Build the graph through networkx with the edges evaluated before
        initial setting: dimensions=30, walk_length=50, num_walks=200, workers=1, window=10, min_count=1, batch_words=4
        """

        self.graph.add_weighted_edges_from([i for i in self.edges])

        node2vec = Node2Vec(self.graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)

        self.embedding = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)

        indexes = [int(i) for i in self.embedding.wv.index_to_key]            

        vector_list = [self.embedding.wv.get_vector(node) for node in self.embedding.wv.index_to_key]

        pca = PCA(n_components=2)
        comp = pca.fit(vector_list).transform(vector_list)

        self.x_cord = [i[0] for i in comp]
        self.y_cord = [i[1] for i in comp]

        self.x_cord = [self.x_cord[i] for i in indexes]
        self.y_cord = [self.y_cord[i] for i in indexes]

        return None
        

    def cluster_building(self, distance=True, linkage="average", dist_threshold= 0.125):
        "This part performe the hierarchical clustering to the matrix distance between each point/node"

        self.cluster = AgglomerativeClustering(compute_distances=True, distance_threshold = dist_threshold,  n_clusters=None,
                                        metric="precomputed", linkage=linkage)

        if distance:
            self.cluster.fit(self.dist_matrix)
        else:
            self.cluster.fit(np.column_stack((self.x_cord, self.y_cord)))

    
    def __associate_cluster__(self):
        raise NotImplemented


    def plot_dendrogram(self, plot_save_path="/dendrogram_cluster", **kargs):
        """To further options and informations check function dendrogram from scipy.cluster.hierarchy library.
        """
        # Create linkage matrix and then plot the dendrogram
        # create the counts of samples under each node

        if self.cluster == None:
            raise Exception("Hierarchical clustering object has not been created, run cluster_building")

        if plot_save_path == "/dendrogram_cluster":
            cur_dir = os.getcwd() + plot_save_path 


        counts = np.zeros(self.cluster.children_.shape[0])
        n_samples = len(self.cluster.labels_)
        for i, merge in enumerate(self.cluster.children_):
            current_count = 0

            for child_idx in merge:

                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [self.cluster.children_, self.cluster.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        plt.title("Hierarchical Clustering Dendrogram")
        dendrogram(linkage_matrix, **kargs)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        # plt.savefig(cur_dir)
        plt.show()

        return None


    def plot_embed(self, plot_save_path="embed_plot", test = False ,  **kargs):

        if self.x_cord == None:
            raise Exception("X cordinates are None tupe, nodes are probably not embedded, try running embed function ")

        if plot_save_path == "embed_plot":
            cur_dir = os.getcwd() + plot_save_path 

        plt.title("Node embeddings")
        if self.cluster != None:
            colors = self.cluster.labels_
            plt.scatter(self.x_cord, self.y_cord, c = colors)
            plt.xlabel("Colors indicated the clustering performed with Hierarchical clustering")
        else:
            plt.scatter(self.x_cord, self.y_cord, **kargs)
        plt.show()
        plt.savefig(cur_dir)

        if test:
            plt.title("Node embeddings")
            plt.scatter(self.x_cord, self.y_cord, **kargs)

        return None


g = Graph_embed(edges=edges, distance_matrix=dist_matrix)
# print("Graph")

v = [[]]
cnt = 0
prev = 0
for i in edges:
    if prev != i[0]:
        v.append([])
        cnt += 1
    v[cnt].append( i[2] )
    prev = i[0]

d = []
for i in v:
    d.append(np.median(np.array(i)))

plt.title("Median")
plt.scatter(POS, d)
plt.show()

n = []
for i in range(dist_matrix.shape[0]):
    n.append(np.mean(dist_matrix[i]))

plt.title("Mean dist matrix")
plt.scatter([i for i in range(dist_matrix.shape[0])], n)
plt.show()

tt = np.quantile(n, 0.4)

# g.embed()
g.cluster_building(dist_threshold=tt)
g.plot_dendrogram(truncate_mode = "level", p=12)
# print(g.cluster.labels_)
# print(g.cluster.n_clusters_)


d = []
for i in v:
    d.append(np.mean(np.array(i)))

mm = np.quantile(np.array([x[2] for x in edges]), 0.2)
print(f"mm : {mm}")

plt.title("Mean")
plt.scatter(POS, d, c = g.cluster.labels_)
plt.show()


cnt = 0
c = 0
ma = []

for i in v:
    if np.mean(np.array(i)) < mm:
        cnt += 1
        ma.append(c)
    c += 1

print(ma)
seen = []
cnt = 0

for i in ma:
    if i in seen:
        continue 
    else:
        links = [x[1] for x in edges if x[0] == i and x[1] in ma]
        seen.extend(links)
        cnt += 1
        
print(seen)
print(cnt)

dist_matrix= np.delete(dist_matrix, np.array(ma), 0)
dist_matrix= np.delete(dist_matrix, np.array(ma), 1)

tt = AgglomerativeClustering(compute_distances=True,  n_clusters=cnt, metric="precomputed", linkage="complete")
tt.fit(dist_matrix)

POS = np.delete(np.array(POS), np.array(ma))
d = np.delete(np.array(d), np.array(ma))

plt.title("Mean fixed cluster")
plt.scatter(POS, d, c = tt.labels_)
plt.show()

g.plot_embed(c= tt.labels_)

# Bisogna contare le righe con media sotto una certa, vedere i collegamenti vicini a quei punti e fare un mini cluster. Usare quel numero di cluster per fare i gruppi.
# Oppure
# Dividere in tanti cluster e unirli in un secondo momento,