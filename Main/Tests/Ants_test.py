# python

# This python scripts is used for tenting travelling_ants

import datetime
from lib.ants import *
from lib.colony import *
from inspyred import swarm, ec
from random import Random
import os
import argparse

print(f"[{datetime.datetime.now()}]:")

def main():

    input = ""
    ipothetical_length = 2000
    reads_length = 200
    pop_size = 100
    num_gen = 12
    evaporation_rate = 0.4
    learnong_rate =0.2
    verbose = True


    seed = random.randint(1, 100)
    prng = Random(seed)

    problem = Assembly_problem(matrix = input, approximate_length = ipothetical_length, reads_len=reads_length)

    print(f"[{datetime.datetime.now()}]: Assembly problem has been asserted!")

    ac = swarm.ACS(prng, problem.components)
    
    ac.terminator = ec.terminators.generation_termination

    if verbose:
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
                        pop_size = pop_size,
                        max_generations = num_gen,
                        evaporation_rate = evaporation_rate,
                        learning_rate = learnong_rate)  #, **args)
    
    # max(ac.archive).candidate
    final = np.array([ i.element for i in max(ac.archive).candidate])

    # print(final)
    print(f"[{datetime.datetime.now()}]: Ants have been travelling for so long, but they finally did it!!")



#######################################################

if __name__ == "__main__":
    main()