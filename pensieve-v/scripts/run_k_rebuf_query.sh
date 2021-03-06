#!/bin/bash
#SBATCH --job-name=rebuf_pensieve.job
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --output=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/results/res_0202/slurm_rebuf.out
#SBATCH --time=5-0

RES_DIR=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/results/res_0202

QUERY=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/queries/marabou_K_rebuf_query.py
MODEL=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/model/output_graph.pb

cd $RES_DIR
for br in {5..2}
do
    for j in {8..2}
    do
        for fp in $(seq 0.4 .02 1.6)
        do
              echo model = $MODEL
              echo query = $QUERY
              echo k = $j download time = $fp
              python3 $QUERY $MODEL $j $fp $br
        done
    done
done
exit 0

