from Bio import SeqIO
import matplotlib.pyplot as plt
import datetime

def extracting_sequence_from_data(input_path:str, limit = 5000)->str:

    #Extracting the sequence from the fasta and selecting the lenght:
    seq = ""
    len_seq = 0
    for seq_record in SeqIO.parse(input_path, format="fasta"):
        seq += seq_record.seq.upper()
        len_seq += len(seq_record)
        if len_seq > limit:
            continue
    seq = seq[:limit]

    return str(seq)

def extract_reads(input_path:str)-> list:

    reads = []
    for seq in SeqIO.parse(input_path, format="fastq"):
        reads.append(seq.seq.upper())

    phred_score = []
    cnt = 0

    for seq_record in SeqIO.parse(input_path, "qual"):
        phred_score.append((cnt, seq_record.letter_annotations["phred_quality"]))
        cnt +=1

    return reads, phred_score

def extract_ont(file:str) ->list:
    raise NotImplemented

def extract_hifiam(file:str) ->list:
    raise NotImplemented