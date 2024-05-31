from itertools import permutations
from math import modf
from inspyred import swarm
from inspyred import ec
from inspyred.ec import selectors
import numpy as np

# import collections
# collections.Iterable = collections.abc.Iterable
# collections.Sequence = collections.abc.Sequence


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
    
    def __init__(self, matrix:np.ndarray, approximate_length:int, reads_len:int):
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
        print(candidate)
        return candidate
    
    # TODO Implement
    def cross_over(self, path:list, matrix:list):
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
            # print(candidate)
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
    
    # def evaluator(self, candidates, args):
    #         """Return the fitness values for the given candidates."""
    #         #P rappresenta la lunghezza stimata del nostro dna
    #         #Pm rappresenta P-0.04P
    #         #PM rappresenta P+0.04P
    #         #sigma = peso della penalità, che aumenta all'aumentare della distanza dal valore di lunghezza dal limite superiore o inferiore.
    #         fitness = []
    #         Pm = self.length - 0.04*self.length
    #         PM = self.length + 0.04*self.length
    #         for candidate in candidates:
    #                 total = 0
    #                 lencandidate = __consensus_sequence_partial__(path = candidate, positions=self.weights, reads_len=self.reads_len)
    #                 for c in candidate:
    #                     total += self.weights[c.element[0]][c.element[1]]
    #                 if lencandidate >= Pm:
    #                     if lencandidate <= PM :
    #                             fitness.append(total) 
    #                     else:
    #                             total=total-self.sigma*(lencandidate-PM)
    #                             fitness.append(total)
    #                 else:
    #                         total=total-self.sigma*(Pm-lencandidate)
    #                         fitness.append(total)
    #         return fitness

