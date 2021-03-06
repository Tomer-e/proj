#!/bin/bash
#SBATCH --job-name=sd_pensieve.job
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --output=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/results/res_0202/slurm_sd.out
#SBATCH --time=5-0

RES_DIR=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/results/res_0202

QUERY=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/queries/marabou_K_query_sd.py
MODEL=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/model/output_graph.pb

cd $RES_DIR
for j in {8..2}
do
    for fp in $(seq 0.05 .01 0.4)
    do
          echo model = $MODEL
          echo query = $QUERY
          echo k = $j download time = $fp
          python3 $QUERY $MODEL $j $fp
    done
done

exit 0
