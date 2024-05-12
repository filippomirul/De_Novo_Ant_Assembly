# DNA-A (De Novo Ant Assembly)

## Table of content

- [Requirements](#requirement)
- [Description](#descrioption)
- [Usage](#usage)
- [Credits](#credits)
- [License](#licence)

## Requirement

- Python >= 3.8
- Scipy
- Numpy
- Numba
- Matplotlib
- Inspyred
- Sklearn
- Biopython


## Descrioption

This algorithm, instead of the usual De Bujin graph used in the majority of similar methods to performe de novo assembly, builds an Overlap-Layout Consensus (OLC) graph starting from the reads given in input. At this data the only input format available is fastq, however the file passed could be, not only deriving from illumina sequencing platform, but also from both nanopore (ONT) and hifiam (Pacbio) platforms.

After this first step, the graph will be simplified eraising those branches and links which retrive poor overlap between the reads. All this part is done in a completly statistical analysis, not taking in consideration the sequences of the reads or the phred score of them.
To assert the presence of contigs (gaps in the coverage), the algorithm has been implemented with a hierarchical clustering of the remaining weighted edges(embedding of the graph will be implemented to retrive further information and statistical analysis). The clustering is performed in order to divide the problem in smaller one, a consequence of this partition is the increasing efficiency of the ANT colony system downstream.

Aftermath each cluster is given as input to the ant colony system, which resolve the optimum problem of finding the best path (best assembly) and return a candidate solution. The path has to be now converted into sequence, in doing so the algorithm builds a consensus matrix. The final matrix evaluate and return the most probaboble base, taken in consideration the multiple overlap created by the algorithm and eacg base phred score. Possibles SNPs or variants will be stored in a separate file.

## Usage

Up to date the algorithm function only with option `--test`

``` bash
usage: ants_assembly.py [-h] [-i INPUT [INPUT ...]] [-o OUTPUT_DIRECTORY] [--test [TEST]] [-p POPULATION_SIZE]
                        [-e EVAPORATION_RATE] [-r LEARNING_RATE] [-v VERBOSE] [-cpus CPU_CORES] [-g MAX_GENERATION]       
                        [-L IPOTHETICAL_LENGTH] [--ester_egg ESTER_EGG]

options:
  -h, --help            show this help message and exit
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        The input must be a fasta or fastq file
  -o OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Directory of output
  --test [TEST]         This is for testing
  -p POPULATION_SIZE, --population_size POPULATION_SIZE
  -e EVAPORATION_RATE, --evaporation_rate EVAPORATION_RATE
                        Internal parameter of the ant colony system
  -r LEARNING_RATE, --learning_rate LEARNING_RATE
                        Internal parameter for the ant colony system
  -v VERBOSE, --verbose VERBOSE
                        Prints and return more information on how the process is developing
  -cpus CPU_CORES, --cpu_cores CPU_CORES
                        Number of cpu to use; default = 2
  -g MAX_GENERATION, --max_generation MAX_GENERATION
                        Number of iterations/generatios of the ant colony algorithm
  -L IPOTHETICAL_LENGTH, --ipothetical_length IPOTHETICAL_LENGTH
                        For a better reconstruction of the genome an ipotetical lenght of the sequence to rebuild is      
                        fondamental for retriving good results
```


## Credits



## Licence