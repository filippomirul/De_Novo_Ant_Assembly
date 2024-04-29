from itertools import permutations
import os
from joblib import Parallel, delayed
from math import modf
from inspyred import swarm
from inspyred import ec
from inspyred.ec import selectors
import numpy as np

# import collections
# collections.Iterable = collections.abc.Iterable
# collections.Sequence = collections.abc.Sequence

def matrix_print(matrix:list) -> None:
    traslator = d = {1:"A", 2:"T", 3:"C", 4:"G", 0:"-"}
    line = []
    for i in range(len(matrix)):
        line.append("")
        for j in range(len(matrix[0])):
            line[i] += traslator[matrix[i][j]]
        print(line[i])

    return 


def __consensus_sequence_partial__(path:list, positions:list , reads_len:int) -> int:
    """
    This is called in to evaluate the length of the sequence, so there is no need to build the actual sequence.
    Therefore is used only the shifting paramiter "dif" to calculate the length.

    path: list of nodes
    positions: matrix with informations
    """
    tot_seq = 0
    cnt = 0

    for i,j in path:

        num = str(positions[j][i]).split(".")
        dif = int(num[0])
    
        if cnt == len(path) - 1:
            tot_seq += dif + reads_len
        else:
            tot_seq += dif
        cnt += 1
    
    return tot_seq 


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
            
            for pos in range(0, len(reads[i]) + 1):
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


def __re_build__(cons_matrix:list):
    
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


def join_consensus_sequence(consensus_matrix:np.ndarray, cpus:int=1)-> str:
    "This functio is just to implement the use of multiples core for recostructing the final sequence."

    step = len(consensus_matrix)/cpus
    cnt = 0
    partials = []
    for i in range(step, len(consensus_matrix) + step, step):
        partials.append((cnt,i))
        cnt += i
    sub_parts = [consensus_matrix[:,i:j] for i,j in partials]
    
    results = Parallel(n_jobs=cpus)(delayed(__re_build__)(i) for i in sub_parts)

    return "".join(results)


def out_files(path_out:str ,reads:list, candidate:list, matrix:list):

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


class Assembly_problem():
    """
    Defines the de novo genome assembly problem.
    
    This class based on the Traveling Salesman problem defines the problem
    of assembling a new genome for which no reference is available (de novo assembly):
    given a set of genomic reads and their pairwise overlap score, find the
    path generating the longest consensus sequence. This problem assumes that 
    the ``weights`` parameter is an *n*-by-*n* matrix of pairwise 
    overlap among *n* reads. This problem is treated as a 
    maximization problem, socfitness values are determined to be the 
    proportional to the sum of the overlaps between each couple of reads
    (the weight of the edge) and the length of the final assembled sequence.
    
    Public Attributes:c
    
    - *weights* -- the two-dimensional list of pairwise overlap 
    - *components* -- the set of ``TrailComponent`` objects constructed
      from the ``weights`` attribute, where the element is the ((source,
      destination), weight)
    - *bias* -- the bias in selecting the component of maximum desirability
      when constructing a candidate solution for ant colony optimization 
      (default 0.5)
    """
    
    def __init__(self, matrix:list, approximate_length:int, reads_len:int):
        self.weights = matrix
        self.reads_len = reads_len
        self.components = [swarm.TrailComponent((i, j), value=(self.weights[i][j])) for i, j in permutations(range(len(self.weights)), 2) if (modf(self.weights[i,j])[0] == 0) and (self.weights[i,j] != 0)]
        self.bias = 0.65
        self.bounder = ec.DiscreteBounder([i for i in range(len(self.weights))])
        self.best_path = None
        self.maximize = True
        self.length = approximate_length
    
    def constructor(self, random, args):
        """Return a candidate solution for an ant colony optimization."""
        candidate = []
        feasible_components = [1]   #Fake initialization to allow while loop to start
        
        # We need to visit all the nodes that CAN be visited, the graph is directed and not complete, meaning we can have no more nodes to visit without visiting all the
        # nodes in the graph, thus, our termination condition is not visitin all the nodes but not having anymore feasible components
        while len(feasible_components) > 0:
            # At the start of the visit, all the components are feasible
            if len(candidate) == 0:
                feasible_components = self.components
            elif len(candidate) == len(self.weights) - 1: # All the nodes have been visited
                return candidate
            else:
                # Update feasible components and set of already visited nodes considering the node visited in the last iteration
                last = candidate[-1]
                already_visited = [c.element[0] for c in candidate]
                already_visited.extend([c.element[1] for c in candidate])
                already_visited = set(already_visited)
                feasible_components = [c for c in self.components if (c.element[0] == last.element[1]) and (c.element[1] not in already_visited)]
            if len(feasible_components) == 0:
                return candidate
            # Choose a feasible component
            if random.random() <= self.bias:
                next_component = max(feasible_components)
            else:
                next_component = selectors.fitness_proportionate_selection(random, feasible_components, {'num_selected': 1})[0]
            candidate.append(next_component)
        return candidate
    
    # TODO Implement
    def cross_over(path:list, matrix:list):
        """This function recombine the solution, is a sort of crossing-over. Takes the path and the score associated to each edge
        iterate over the path and switch two edge.
        """
        imaginary_string = range(len(path))

        min_1 = path.index(min([c.value for c in path]))
        min_2 = path.index(min([c.value for c in path if (c.element[0] == min_1[0]) and (c.element[1] == min_1[1])]))
        if min_2 == None:
            return None
        else:
            # make cross over between those two
            return None

    # TODO rebuild
    def evaluator(self, candidates:list, args):
        """Return the fitness values for the given candidates."""
        fitness = []
        for candidate in candidates:
            total = 0
            for c in candidate:
                total += self.weights[c.element[0]][c.element[1]]
            last = (candidate[-1].element[1], candidate[0].element[0])
            current_path=[(c.element[0], c.element[1]) for c in candidate] 
            total += self.weights[last[0]][last[1]]
            current_sequence = __consensus_sequence_partial__(current_path, positions=self.weights, reads_len = self.reads_len)
            length_score = abs((self.length-current_sequence)/self.length)
            s = [5, 3, 1, 0.5, 0.2]
            perc=[0, 0.01, 0.05, 0.08, 0.1, 0.2]
            l_score = 0.1
            for i in range(len(perc)-1):
                if length_score >= perc[i] and length_score < perc[i+1]:
                    # print(perc.index(i))
                    l_score = s[perc.index(perc[i])]

            if self.best_path == None or len(current_path) > len(self.best_path):
                self.best_path = current_path
            
            score = total*l_score
            fitness.append(score)

        return fitness

