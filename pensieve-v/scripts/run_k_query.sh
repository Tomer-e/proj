#!/bin/bash
#SBATCH --job-name=pensieve_q
#SBATCH --cpus-per-task=8
#SBATCH --mem=10g
#SBATCH --output=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/sed_res
#SBATCH --partition=long
#SBATCH --time=1-0
#SBATCH --signal=B:SIGUSR1@300



#hd query
for fp in $(seq 0.4 .05 0.5)
do
for j in {1..2}
do
  python3 ../queries/marabou_K_query_hd.py ../model/output_graph.pb $j $fp
done
done