from joblib import Parallel, delayed
from numba import jit
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from node2vec import Node2Vec 
import networkx as nx
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import datetime
from numba import jit
from scipy.cluster.hierarchy import dendrogram
import os

def eval_nonzeros(graph:np.ndarray)-> int:

    cnt = 0
    for i in range(len(graph)):
        cnt += np.count_nonzero(graph[i])

    return cnt

jit(nopython=True)
def __matrix_selection__(input_tuple:tuple, cut_off = 0.1)->list:
    """Value the distibution of the columns
    """

    # Init
    array = input_tuple[0]
    row = input_tuple[1]

    # 
    links = [x for x in array if (x > 0) and (str(x).split(".")[1] != "1")]
    # to_not_touch = [x for x in array if (str(x).split(".")[1] == "1")]
    chosen = sorted(links)[-(int(len(links) * cut_off)):]
    dissmissable_links = []

    cnt=0
    for i in array:
        if i not in chosen and i in links:
            dissmissable_links.append((row, cnt))
            cnt += 1
        else:
            cnt += 1
            continue

    # print(dissmissable_links)
    return dissmissable_links

jit(nopython=True)
def __matrix_sempl__(matrix:np.ndarray, dissmissable_links:list) -> np.ndarray:
    """Changes the occurrencie valued as unprobable in zeros, eraising in this way the link
    """
    for epoch in dissmissable_links:
        for i,j in epoch:
            matrix[i,j] = 0
            matrix[j,i] = 0

    return matrix


def graph_semplification(graph:np.ndarray, cores:int)->np.ndarray:

    indeces_to_cut = Parallel(n_jobs = cores)(delayed(__matrix_selection__)(i) for i in [(graph[i,:], i) for i in range(graph.shape[0])])

    # print(graph.shape[0]**2)
    # print(indeces_to_cut[0])

    matrix = __matrix_sempl__(graph, dissmissable_links=indeces_to_cut)

    return matrix
        

def cluster(distance_matrix: np.ndarray, distance=True, linkage="single", dist_threshold= 0.003):
    "This part performe the hierarchical clustering to the matrix distance between each point/node"

    cluster = AgglomerativeClustering(compute_distances=True, distance_threshold = dist_threshold,  n_clusters=None,
                                        metric="precomputed", linkage=linkage).fit(distance_matrix)
    
    return cluster


def plot_dendrogram(model, plot_save_path="dendrogram_cluster"):
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
    dendrogram(linkage_matrix)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.savefig(plot_save_path)

    return None


