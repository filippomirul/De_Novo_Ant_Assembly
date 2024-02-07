from Bio import pairwise2
from Bio import SeqIO
import yaml
from math import modf
from Bio.Seq import Seq
from inspyred import swarm
from inspyred import benchmarks
from inspyred import ec
from inspyred.ec import selectors
from collections import deque
import numpy as np
import itertools
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# TODO CUDASW++

def comstum_reads(seq: str, length_reads = 160, coverage = 5, verbose = False) -> list:
    
    """The function split the sequence in input in reads.
    The splitting is done using random numbers, the amount of reds is given by: (len(seq)/length_read)*coverage.
    """

    number_of_reads = int(len(seq)/length_reads) * coverage
    starting_pos = random.sample(range(0, len(seq)-length_reads+1), number_of_reads)
    reads = []

    for num in starting_pos:
        reads.append(seq[num:num+length_reads])

    if verbose == True:
        y = [0 for i in range(0,len(seq)+1)]
        for i in starting_pos:
            for j in range(i, i+length_reads+1):
                y[j] += 1 
        plot = sns.set_theme(style="darkgrid")
        plot = sns.lineplot(y)
        # plot.sevefig("")
        print(f"There are {y.count(0)} bases that have 0 coverage.")

    return reads

def eval_allign(reads:list, par = [3, -2, -40, -40]):
    """Funtion that evaulate the alignment

    reads: list of DNA sequences, each of the read is a string

    par: list of parameters to performe the alignment
    es (the examples represent the defoult parameters):
    match_score = 2,
    mismatch_penalty = -5,
    gap_open_penalty = -22,
    gap_extension_penalty = -5

    output:
    Matrix with the weigts (distances) between the reads (nodes)
    In this matrix there are both the scores of the alignment, recognizable for the tipical integer score and
    a flot number which is needed after to recompose the sequence, it indicates the overlapping bases.
    Ex:
        allignment score -> 2.0, 13.0, ...
        overlapping number -> 12.24, 1.6, 19.56, ...
            the number before the . is the starting base, while the one after is the ending base. To avoid problem later
            with 0 a 1 digit is added for then remove it. So 12.30 become 12.301 but the corret indices are 12 and 30.

        These two numbers are link by the position in the matrix which are the trasposition
        Score 14.0 in position (1,5) --> 12.34 in position (5,1). Only the score position is referred
        to the direction of the edge.
        1 ---> 5 with allignment score 14 and read_1 is overlapped with read_5 in positions from 12 to 34 (both included)

    Example of a matrix with three reads:

        | 1    | 2    | 3    
     1  | 0    |3.0   |12.231 
     2  |34.601|  0   | 23.0
     3  | 18.0 |45.701|  0
    """
    length = len(reads)
    # initialization of the matrices
    weigth_matrix = np.zeros((length, length))

    # The score of the allingment of read[1] to read[2] is the same of the opposite (read[2] to read[1])
    # So when the function found the diretionality of the allignment put the score in rigth spot and a 0 in the wrong one.
    visited = deque([j for j in range(length)])
    
    for i in range(length):

        for j in visited:

            if i == j:
                # the diagonal of the matrix has 0 score because we are allinging the same sequence
                continue
            else:
                # pairwise must return a positive score, if there is no one it return None
                allignment = pairwise2.align.localms(Seq(reads[i]), Seq(reads[j]), par[0], par[1], par[2], par[3])
                if len(allignment) == 0:
                    continue
                else:
                    allignment= allignment[0]
                      
                start = allignment[3]
                over = allignment[4] - start
                # return object [seqA, seqB, score, start(inc), end(ex)]

                if allignment[0][0] == "-":
                    # This means that the first reads in input has a gap at the beginning of the allignment.
                    # Therefore the first reads in input (i) is downstream,
                    # so I add the score in the matrix but in the position (j,i) instead of (i,j) where there is a 0
                    diff = allignment[0].count("-")
                    weigth_matrix[j][i] = allignment[2]*over
                    weigth_matrix[i][j] = float(f"{diff}.{start}1")
                    # to avoid to loosing a 0 is been introduced a 1 digit which will be removed afterwords

                else:
                    # In the opposite case, where the i read is upstream (i,j) has the score, while (j,i) has a 0   
                    diff = allignment[1].count("-")                 #
                    weigth_matrix[i][j] = allignment[2]*over
                    weigth_matrix[j][i] = float(f"{diff}.{start}1")

                    
        visited.popleft()
    print(f"Done matrix {len(weigth_matrix)}x{len(weigth_matrix)}")
    return weigth_matrix

def matrix_print(matrix:list) -> None:
    traslator = d = {1:"A", 2:"T", 3:"C", 4:"G", 0:"-"}
    line = []
    for i in range(len(matrix)):
        line.append("")
        for j in range(len(matrix[0])):
            line[i] += traslator[matrix[i][j]]
        print(line[i])

    return    

def consensus_sequence(path:list, reads:list, positions:list, length:int, last = False, max_coverage= 12, verbose = False) -> str:
    """Rebluild from the list of reds, the path and the matrix with the scores the allignment.

    path:list of tuple with edges --> [(1,3), (3,6), ...]
    reads: list of the reads ---> ["ATCGA", "AGGCTG", ...] 
    positions: is the weigth matrix, but will be considered only the number linked with the base overlapping

    output: a string with the sequece reconstructed    

    Ex
        path = [(6,5), (5,9), (9,11), (11,7), (7,4), (4,1), (1,3)]
        reads = ['RAGIL', 'LISTI', 'LIFRA', 'STICH', 'GILIS', 'ERCAL', 'SUPER', 'FRAGI', 'ILIST', 'RCALI', 'PERCA', 'ALIFR']
        positions: for space reason the matrix is not presented, but is similar to the one in the eval_allign help.
    """

    if last:

        D = {"A":1, "T":2, "C":3, "G":4}
        d = {1:"A", 2:"T", 3:"C", 4:"G"}
        rec = np.zeros((max_coverage, length))
        leng = len(rec[0])
        cum_dif = 0
        adding = np.zeros((max_coverage, int(length/100)))

        for i,j in path:
            # Here i,j represent the edge of the graph, to retrive not the score but the alignment
            # the function needs the opposite position where there are these informations matrix[j][i]
            # something like 12.22, 12 is the strating base 22 is the ending base of the overlapping, both included.

            num = str(positions[j][i]).split(".")
            # start = int(num[1][:-1])  # included
            dif = int(num[0])

            if rec[0,0] == 0:
                
                for pos in range(0, len(reads[i])):
                    if rec[0,pos]!=0:
                        rec = np.append(rec,adding, 1)
                    rec[0,pos] = D[reads[i][pos]]
                cum_dif += dif
                temp = 0
                for p in range(cum_dif, cum_dif + len(reads[j])):
                    if rec[1,pos]!=0:
                        rec = np.append(rec, adding, 1)
                    rec[1, p] = D[reads[j][temp]]
                    temp +=1

            else:
                cum_dif += dif
                temp = 0
                for pos in range(cum_dif, cum_dif+len(reads[j])):
                    if rec[0,pos]!=0:
                        rec = np.append(rec, adding, 1)
                    row = 0
                    while rec[row, pos] >= 1:
                        row += 1
                    rec[row, pos] = D[reads[j][temp]]
                    temp +=1

        if verbose:
            # TODO here we wants stats
            matrix_print(rec)
            
        cons_seq = ""
        for i in range(0, leng):
            base = [int(x) for x in rec[:,i] if x > 0]
            if base == []:
                return cons_seq
            ind = []
            for num in [1,2,3,4]:
                ind.append(base.count(num))
            more_frequent = ind.index(max(ind)) + 1
            # TODO stats
            cons_seq += d[more_frequent]

        return cons_seq

    else:
        # TODO implement because maybe is not necessary the build the seq, maybe only adding numbers
        tot_seq = []

        for i,j in path:

            num = str(positions[j][i]).split(".")
            dif = int(num[0])  # included
            # start = int(num[1][:-1])
        
            if len(tot_seq) == len(path)-1:
                tot_seq.append(reads[i][:dif] + reads[j])
            else:
                tot_seq.append(reads[i][:dif])

                # The function goes backword so before reconstructing the sequence there is the need to reverse the list
                #  with the part of the reds to assemble
        real_seq = ""

        for seq in tot_seq:
            real_seq += seq
        
        return len(real_seq)   


# def tuning_parameters(yaml_file = "training.yaml"):
#     """This function is just to tune the parameter of the function
#     """

#     with open(yaml_file, "r") as file:
#         file = yaml.safe_load(file)

#     sequences = file["sequence_to_test"].values()
#     parameters = file["allignment_parameters"].values()

#     with open(yaml_file, "a") as y_file:

#         for name_sequence in sequences:
#             seq = ""
#             y_file["scores"] = {}
#             for param in parameters:
#                 print(name_sequence[7:-4])
#                 print(param)
#                 for seq_record in SeqIO.parse(name_sequence, format="fasta"):
#                     seq += seq_record.seq.upper()
#                 print(seq[:80])
#                 y_file["scores"]
#                 run algorithm
#                 file["scores"]

class Assembly_problem():
    """Defines the de novo genome assembly problem.
    
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
    
    def __init__(self, reads, approximate_length):
        self.weights = eval_allign(reads)
        self.reads = reads
        self.components = [swarm.TrailComponent((i, j), value=(self.weights[i][j])) for i, j in itertools.permutations(range(len(self.weights)), 2) if modf(self.weights[i,j])[0] == 0]
        self.bias = 0.5
        self.bounder = ec.DiscreteBounder([i for i in range(len(self.weights))])
        self.best_path = None
        self.maximize = True
        self.length = approximate_length
    
    def constructor(self, random, args):
        """Return a candidate solution for an ant colony optimization."""
        self._use_ants = True
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
                feasible_components = [c for c in self.components if c.element[0] == last.element[1] and c.element[1] not in already_visited]
            if len(feasible_components) == 0:
                return candidate
            # Choose a feasible component
            if random.random() <= self.bias:
                next_component = max(feasible_components)
            else:
                next_component = selectors.fitness_proportionate_selection(random, feasible_components, {'num_selected': 1})[0]
            candidate.append(next_component)
        return candidate
    
    def cross_over(path:list, matrix:list)->list:
        """This function recombine the solution. Takes the path and the score associated to each edge
        iterate over the path and switch two edge.
        """
        imaginary_string = range(len(path))

        min_1 = path.index(min([c.value for c in path]))
        min_2 = path.index(min([c.value for c in path if (c.element[0] == min_1[0]) and (c.element[1] == min_1[1])]))
        return None

    
    def evaluator(self, candidates, args):
        """Return the fitness values for the given candidates."""
        fitness = []
        for candidate in candidates:
            total = 0
            for c in candidate:
                total += self.weights[c.element[0]][c.element[1]]
            last = (candidate[-1].element[1], candidate[0].element[0])
            current_path=[(i.element[0],c.element[1]) for i in candidate]
            total += self.weights[last[0]][last[1]]
            current_sequence = consensus_sequence(current_path, reads=self.reads, positions=self.weights, length=self.length)
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

# from utils.utils_07.exercise_1 import *
# from utils.utils_07.plot_utils import *
from random import Random
import inspyred
import collections
collections.Iterable = collections.abc.Iterable
collections.Sequence = collections.abc.Sequence

seq = ""
for seq_record in SeqIO.parse("C:\\Users\\filoa\\OneDrive\\Desktop\\Bio AI\\Project_Bio_AI\\Data\\Aspergillus_flavus.fna", format="fasta"):
    seq += seq_record.seq.upper()

seq = seq[:10000]

reads = comstum_reads(seq, length_reads=3000, coverage=6)

# common parameters
pop_size = 80
max_generations = 8
seed = 12
prng = Random(seed)
display = True
# ACS specific parameters
evaporation_rate = 0.7
learning_rate = 0.1

args = {}
args["fig_title"] = "ACS"

# run ACS
problem = Assembly_problem(reads, len(seq))
ac = inspyred.swarm.ACS(prng, problem.components)
# ac.observer = [plot_observer]
ac.terminator = inspyred.ec.terminators.generation_termination

final_pop = ac.evolve(generator=problem.constructor, 
                      evaluator=problem.evaluator, 
                      bounder=problem.bounder,
                      maximize=problem.maximize, 
                      pop_size=pop_size,
                      max_generations=max_generations,
                      evaporation_rate=evaporation_rate,
                      learning_rate=learning_rate,**args)
best_ACS = max(ac.archive)


c = [(i.element[0], i.element[1]) for i in best_ACS.candidate]
d = consensus_sequence(c, reads, length=5000, positions=problem.weights, last=True)
# print(len(d))
al = pairwise2.align.localms(d, seq, 3,-1,-5,-5)[0]
# print(al[0])
# print(al[1])
# print(al[2])

cont = 0
for i in range(len(seq)):
    if al[0][i] == al[1][i]:
        cont += 1
perc = cont/len(seq)

ll = []
ll.append("Thr first line is the reconstructed seq, while the second is the real sequence:\n")
cnt=0
for i in range(50,2*len(seq),50):
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
ll.append(str(perc))

import os

path = "C:\\Users\\filoa\\OneDrive\\Desktop\\Programming_trials\\Ant_colony_assembly\\Aspergillus_2_info.txt"
if not os.path.exists(path):
    os.makedirs

new_file = open(path, "w")
results_to_write = ll
new_file.writelines(results_to_write)
new_file.close()