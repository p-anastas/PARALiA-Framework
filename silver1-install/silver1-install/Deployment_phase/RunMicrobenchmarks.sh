#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

DEPLOY_DIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/Deployment_phase
BACKEND=CuCuBLAS

machine=silver1
device=$1

devices=4

BIN_DIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/silver1-install/silver1-install/Deployment_phase/bin
RES_DIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/silver1-install/silver1-install/Deployment_phase/Database/Benchmark-Results

if [[ "$machine" == "dungani" ]]; then
	export CUDA_DEVICE_ORDER=PCI_BUS_ID
fi

mkdir -p "${RES_DIR}/microbench_logs"

transfer_log="${RES_DIR}/microbench_logs/transfer_microbench.log"
dgemm_log="${RES_DIR}/microbench_logs/dgemm_microbench.log"
daxpy_log="${RES_DIR}/microbench_logs/daxpy_microbench.log"
sgemm_log="${RES_DIR}/microbench_logs/sgemm_microbench.log"

micro_transfer_exec="${BIN_DIR}/RunCuCuBlasLinkBench"
micro_dgemm_exec="${BIN_DIR}/RunCuCuBlasDgemmBench"
micro_sgemm_exec="${BIN_DIR}/RunCuCuBlasSgemmBench"
micro_daxpy_exec="${BIN_DIR}/RunCuCuBlasDaxpyBench"

echo "Performing microbenchmarks for transfers..."
rm $transfer_log
for (( end_device = -1; end_device < $devices -1; end_device++ ))
do
	echo "$micro_transfer_exec $device $end_device &>> $transfer_log"
	$micro_transfer_exec $device $end_device &>> $transfer_log
	echo "$micro_transfer_exec $end_device $device &>> $transfer_log"
	$micro_transfer_exec $end_device $device &>> $transfer_log
done
echo "Done"

#exit 1

rm $daxpy_log
echo "Performing microbenchmarks for daxpy..."
echo "$micro_daxpy_exec $device &>> $daxpy_log"
$micro_daxpy_exec $device &>> $daxpy_log

echo "Done"

#exit 1

# dgemm micro-benchmark Tile size
rm $dgemm_log
echo "Performing microbenchmarks for dgemm..."
for TransA in N T;
do
	for TransB in N T;
	do
		echo "$micro_dgemm_exec $device $TransA $TransB &>> $dgemm_log"
		$micro_dgemm_exec $device $TransA $TransB &>> $dgemm_log
	done
done
echo "Done"

exit 1

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
