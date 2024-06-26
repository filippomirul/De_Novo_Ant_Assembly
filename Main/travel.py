#!/usr/bin/env python
# -*- coding: utf-8 -*-


from lib.ants import *
from lib.colony import *
from inspyred import swarm, ec
from random import Random
import os
import argparse
import datetime
from scipy.io import loadmat, savemat


def main():

    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-i", "--input", type = str, help = "The input must be a fasta or fastq file", required = True)
    parser.add_argument("-o", "--output_directory", type = str, help = "Directory of output", default = "./standard_output")
    parser.add_argument("-p", "--population_size", type = int, default = 80,
                        help = "")
    parser.add_argument("-e", "--evaporation_rate", type = float, default = 0.2,
                         help="Internal parameter of the ant colony system")
    parser.add_argument("-r", "--learning_rate", type = float, default = 0.4,
                        help = "Internal parameter for the ant colony system")
    parser.add_argument("-v", "--verbose", type = bool, default = False,
                        help = "Prints and return more information on how the process is developing")
    parser.add_argument("-cpus", "--cpus_cores", type = int,
                        help = "Number of cpu to use", default = 2)
    parser.add_argument("--reads_lenght", default=200, type = int)
    parser.add_argument("-g", "--max_generation", default = 10, help = "Number of iterations/generatios of the ant colony algorithm")
    parser.add_argument("-s", "--simulation", default = True, help="This is development only!")
    parser.add_argument("-L", "--ipothetical_length", default=10000, type = int,
                         help = "For a better reconstruction of the genome an ipotetical lenght of the sequence to rebuild is fondamental for retriving good results")

    args = parser.parse_args()

    current_path = os.getcwd()
    data_out_path = current_path + "/Data"
    final_array_path = data_out_path + "/final_array.pkl"
    selected_edge_path = data_out_path +"/selected_edges.pkl"



    seed = random.randint(1, 100)
    prng = Random(seed)

    problem = Assembly_problem(edges=load_list(where=args.input), approximate_length = args.ipothetical_length, reads_len=args.reads_lenght)

    print(f"[{datetime.datetime.now()}]: Assembly problem has been asserted!")

    ac = swarm.ACS(prng, problem.components)
    
    ac.terminator = ec.terminators.generation_termination

    if args.verbose:
        display = True
        ac.observer = ec.observers.stats_observer
    else:
        display = False

    print(f"[{datetime.datetime.now()}]: Proceeding with ants ...")

    final_pop = ac.evolve(generator = problem.constructor,
                        evaluator = ec.evaluators.parallel_evaluation_mp, 
                        mp_evaluator = problem.evaluator, 
                        bounder = problem.bounder,
                        maximize = problem.maximize,
                        # mp_nprocs = args.cpus_cores,
                        pop_size = args.population_size,
                        max_generations = args.max_generation,
                        evaporation_rate = args.evaporation_rate,
                        learning_rate = args.learning_rate) 
    
    # max(ac.archive).candidate
    final = np.array([ i.element for i in max(ac.archive).candidate])

    # print(final)
    print(f"[{datetime.datetime.now()}]: Ants have been travelling for so long, but they finally did it!!")

    save_list(data=final, where=final_array_path)

#######################################################

if __name__ == "__main__":
    main()