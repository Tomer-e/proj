#!/bin/bash
#SBATCH --job-name=sd8_pensieve.job
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --output=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/sed_res_k8/slurm_sd.out
#SBATCH --time=5-0

RES_DIR=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/sed_res_k8/

QUERY=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/queries/marabou_K_query_sd.py
MODEL=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/model/output_graph.pb

cd $RES_DIR
j=8
for fp in $(seq 0.05 .01 0.4)
do
      echo model = $MODEL
      echo query = $QUERY
      echo k = $j download time = $fp
      python3 $QUERY $MODEL $j $fp
done

exit 0
