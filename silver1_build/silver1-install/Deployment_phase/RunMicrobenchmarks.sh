#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

DEPLOY_DIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/Deployment_phase
BACKEND=CuCuBLAS

machine=silver1
device=$1

BIN_DIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/silver1_build/silver1-install/Deployment_phase/bin
RES_DIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/silver1_build/silver1-install/Deployment_phase/Database/Benchmark-Results

if [[ "$machine" == "dungani" ]]; then
	export CUDA_DEVICE_ORDER=PCI_BUS_ID
fi

mkdir -p "${RES_DIR}/microbench_logs"

transfer_log="${RES_DIR}/microbench_logs/transfer_microbench_gpu.log"
dgemm_log="${RES_DIR}/microbench_logs/dgemm_microbench_gpu.log"
sgemm_log="${RES_DIR}/microbench_logs/sgemm_microbench_gpu.log"

micro_transfer_exec="${BIN_DIR}/RunCuCuBlasTransferBench"
micro_dgemm_exec="${BIN_DIR}/RunCuCuBlasDgemmBench"
micro_sgemm_exec="${BIN_DIR}/RunCuCuBlasSgemmBench"


echo "Performing microbenchmarks for transfers..."
rm $transfer_log
echo "$micro_transfer_exec $device -1 &>> $transfer_log"
$micro_transfer_exec $device -1 &>> $transfer_log
echo "$micro_transfer_exec -1 $device &>> $transfer_log"
$micro_transfer_exec -1 $device &>> $transfer_log
echo "Done"

# dgemm micro-benchmark Tile size
rm $dgemm_log
echo "Performing microbenchmarks for dgemm..."
for TransA in N T;
do
for TransB in N T;
do
	echo "$micro_dgemm_exec $device $TransA $TransB &>> $dgemm_log"
	$micro_dgemm_exec $device $TransA $TransB &>> $dgemm_log
	echo "Done"
done 
done

# sgemm micro-benchmark Tile size
rm $sgemm_log
echo "Performing microbenchmarks for sgemm..."
for TransA in N T;
do
for TransB in N T;
do
	echo "$micro_sgemm_exec $device $TransA $TransB &>> $sgemm_log"
	$micro_sgemm_exec $device $TransA $TransB &>> $sgemm_log
	echo "Done"
done 
done

