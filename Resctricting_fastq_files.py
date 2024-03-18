import pysam
from Bio import SeqIO
import textwrap
import argparse


parser = argparse.ArgumentParser(
    formatter_class = argparse.RawDescriptionHelpFormatter,
    description = textwrap.dedent("""
    This scripts extract from a bam file all the name of the reads and uses these names to filter a fastq format file (parsed)
    to reduce the number of reads in it. The input fastq is not overwrite, instead a new one is created.
    """))

parser.add_argument("-b", "--bam_file", type = str, help = "Bam file from which extract the names of the reads")
parser.add_argument("-i", "--input", type = str, help = "Fastq file input")
parser.add_argument("-o", "--output", type = str, help = "Fastq file output")
args = parser.parse_args()

bam_file = args.bam_file
fastq_file = args.input
output_file = args.output

ids_to_keep = []
record_for_fastq = []
filtered_records = []

# Open the BAM file for reading
with pysam.AlignmentFile(bam_file, "rb") as bam:
    # Iterate over aligned reads and save their names
    for read in bam:
        ids_to_keep.append(read.query_name)

# Open fastq file and read the id of the sequences if it is in the list above is saved
for record in SeqIO.parse(fastq_file, "fastq"):
    if record.id in ids_to_keep:
        filtered_records.append(record)

with open(output_file, "w") as handle:
    SeqIO.write(filtered_records, handle, "fastq")

# C:\Users\filoa\Downloads\SRR28365120.fastq.gz