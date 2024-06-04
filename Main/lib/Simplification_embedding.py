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
import pickle

def save_list(data:list, where:str)-> None:
    """Function for saving list files"""

    with open(where, "wb") as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    return None

def load_list(where:str)-> list:
    
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    return data

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
