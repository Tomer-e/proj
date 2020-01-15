#!/bin/bash
#SBATCH --job-name=dnc_pcc_0
#SBATCH --cpus-per-task=8
#SBATCH --mem=1g
#SBATCH --output=/cs/labs/guykatz/tomerel/vrl/proj/PCC-v/dnc_outputs/solver.out
#SBATCH --partition=long
#SBATCH --time=2-0
#SBATCH --signal=B:SIGUSR1@300


#function extract_result {
#	python3 resources/utils/result_extractor.py
#}
#
#pgid=$(($(ps -o pgid= -p $$)))
#trap 'kill -TERM $pid & wait $pid ; extract_result ; exit' SIGUSR1
#
#
#function extract_result {
#	python3 resources/utils/result_extractor.py
#}
#
#pgid=$(($(ps -o pgid= -p $$)))
#trap 'kill -TERM -$pgid & extract_result' SIGTERM
#
#pwd; hostname; date
#
#cwd=$(pwd)
#
#resources/solvers/dnc/run.sh resources/nnet/ACASXU_run2a_3_6_batch_2000.nnet resources/property3.txt resources/results/solver_result/solver_1_nnet_25.sum '--dnc --num-workers=16' & pid=$!
#wait $pid
#
#cd $cwd
#
#extract_result
#
#date


#set j = 1
for j in {1..3}
do
  python3 correct_queries/marabou_K_query3.py model/output_graph.pb $j
done
