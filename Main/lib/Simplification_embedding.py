import datetime
from joblib import Parallel, delayed
import grape
from numba import jit
import numpy as np

@jit(nopython = True)
def __prepare_simpl_intup__(matrix:np.ndarray)->list:
    """Is needed to set up the parallelization, divide the matrix in columns
    """
    # output = (array, column)

    my_list = []

    for i in range(len(matrix)):
        my_list.append((list(matrix[:,i]), i))

    return my_list


def __matrix_selection__(input_tuple:tuple, cut_off = 0.1)->list:
    """Value the distibution of the columns
    """

    # Init
    array = input_tuple[0]
    column = input_tuple[1]

    # 
    links = [x for x in array if (x > 0) and (str(x).split(".")[1] == "0")]
    to_not_touch = [x for x in array if (str(x).split(".")[1] == "1")]
    chosen = sorted(links)[-(int(len(links)*cut_off)):]
    dissmissable_links = []

    cnt=0
    for i in array:
        if (i not in chosen) and (i not in to_not_touch):
            dissmissable_links.append((cnt,column))
        cnt += 1

    # for i in links:
    #     if i not in chosen:
    #         dissmissable_links.append((array.index(i), column))

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