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

    # print(graph.shape[0]**2)
    # print(indeces_to_cut[0])

    matrix = __matrix_sempl__(graph, dissmissable_links=indeces_to_cut)

    return matrix

class graph_embed():
    
    def __init__(self, edges, dist_threshold= 0.5, dimensions=30, walk_length=50, num_walks=200, workers=1,
                window=10, min_count=1, batch_words=4,  linkage="single"):
        self.edges = edges
        self.dist_matrix = loadmat()
        self.graph = nx.DiGraph()
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.window = window
        self.min_count = min_count
        self.batch_words = batch_words
        self.linkage = linkage
        self.dist_threshold = dist_threshold
        self.model=None
        self.embed=None
        self.x_cord=None
        self.y_cord=None
        

    def embed(self):
        "Build the graph through networkx with the edges evaluated before"
        self.graph.add_weighted_edges_from([(self.edges[0], self.edges[1], self.edges[4]) for i in self.edges])

        node2vec = Node2Vec(self.graph, dimensions=self.dimensions, walk_length=self.walk_length, num_walks=self.num_walks, workers=self.workers)

        self.embed = node2vec.fit(window=self.window, min_count=self.min_count, batch_words=self.batch_words)

        vector_list = [self.embed.wv.get_vector(node) for node in self.embed.wv.index_to_key]

        pca = PCA(n_components=2)
        comp = pca.fit(vector_list).transform(vector_list)

        self.x_cord = [i[0] for i in comp]
        self.y_cord = [i[1] for i in comp]
        

    def cluster(self):
        "This part performe the hierarchical clustering to the matrix distance between each point/node"

        self.model = AgglomerativeClustering(compute_distances=True, distance_threshold = self.dist_threshold,  n_clusters=None,
                                            metric="precomputed", linkage=self.linkage).fit(self.dist_mat)

    
    def __associate_cluster__(self):
        raise NotImplemented

    def plot_dendrogram(self, plot_save_path="dendrogram_cluster"):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(self.model.children_.shape[0])
        n_samples = len(self.model.labels_)
        for i, merge in enumerate(self.model.children_):
            current_count = 0

            for child_idx in merge:

                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [self.model.children_, self.model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        plt.title("Hierarchical Clustering Dendrogram")
        dendrogram(linkage_matrix)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.savefig(plot_save_path)


    def plot_embed(self, plot_save_path="embed_plot"):

        plt.scatter(self.x_cord, self.y_cord)
        plt.savefig(plot_save_path)


