from joblib import Parallel, delayed
from numba import jit
import numpy as np
from sklearn.cluster import AgglomerativeClustering

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

    print(graph.shape[0]**2)
    print(indeces_to_cut[0])

    matrix = __matrix_sempl__(graph, dissmissable_links=indeces_to_cut)

    return matrix

def __graph_embedding__():
    """Trasform the nodes in vectors in a pre-defined space
    """
    raise NotImplemented


def __cluster_definition__():
    """Group the points (nodes/reads) in cluster to create contigs
    """
    raise NotImplemented


def cluster_link_mantainance():
    """The links between cluster represent the order of the contigs or possible high order alignments
    """    
    raise NotImplemented

