#!/bin/bash
#SBATCH --job-name=pensieve.job
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --output=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/sed_res_0102/slurm.out
#SBATCH --time=5-0

QUERY=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/queries/marabou_K_query_hd.py
MODEL=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/model/output_graph.pb

RES_DIR=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/sed_res_0102


cd $RES_DIR
for fp in $(seq 0.4 .05 1)
do
    for j in {1..8}
    do
          echo model = $MODEL
          echo query = $QUERY
          echo k = $j download time = $fp
          python3 $QUERY $MODEL $j $fp
    done
done

QUERY=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/queries/marabou_K_query_sd.py
MODEL=/cs/labs/guykatz/tomerel/vrl/proj/pensieve-v/model/output_graph.pb



for fp in $(seq 0 .05 0.4)
do
    for j in {1..8}
    do
          echo model = $MODEL
          echo query = $QUERY
          echo k = $j download time = $fp
          python3 $QUERY $MODEL $j $fp
    done
done


exit 0