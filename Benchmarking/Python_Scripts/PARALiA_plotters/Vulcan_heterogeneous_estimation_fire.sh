#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

module load gcc/11.2.0
module load system/cuda/11.6.0
module load python/3.10
source ~/.bashrc

alpha=1.2345
beta=1.1154
CoCopelia_run_hetero=./vulcan-install/Benchmarking/testing-bin/dgemm_runner_hetero

$CoCopelia_run_hetero 1 1 -1 1000000000 N N $alpha $beta 3500 3500 3500 0 0 0 0 &
$CoCopelia_run_hetero 1 10 -1 1000000000 N N $alpha $beta 3500 3500 3500 1 1 1 1 &
$CoCopelia_run_hetero 1 100 -1 1000000000 N N $alpha $beta 1250 1250 1250 2 2 2 2 &
$CoCopelia_run_hetero 1 1000 -1 1000000000 N N $alpha $beta 1250 1250 1250 3 3 3 3 &
$CoCopelia_run_hetero 1 10000 -1 1000000000 N N $alpha $beta 3500 3500 3500 4 4 4 4 &
$CoCopelia_run_hetero 1 1000000 -1 1000000000 N N $alpha $beta 3500 3500 3500 6 6 6 6 &
