#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

PROJECT_INSTALL_DIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/silver1-install/silver1-install

BACKEND=
machine=silver1
NUM_DEVICES=

LIBSC_DIR=$PROJECT_INSTALL_DIR/Benchmarking
LIBSC_TEST_LOG_DIR=$LIBSC_DIR/testLogs

mkdir -p "${LIBSC_TEST_LOG_DIR}/exec_logs"

alpha=1.2345
beta=1.1154


for FUNC in Dgemm Sgemm
do
	perf_log="${LIBSC_TEST_LOG_DIR}/exec_logs/${FUNC}_perf_eval.log"
	rm $perf_log

	CoCopelia_run=$LIBSC_DIR/testing-bin/CoCoPeLia${FUNC}Runner
	CoCopelia_run_best=$LIBSC_DIR/testing-bin/CoCoPeLia${FUNC}RunnerBest
	CoCopelia_run_bestest=$LIBSC_DIR/testing-bin/CoCoPeLia${FUNC}RunnerBestest
	CoCopelia_run_old=$LIBSC_DIR/testing-bin/CoCoPeLia${FUNC}RunnerOld
	CoCopelia_run_best_old=$LIBSC_DIR/testing-bin/CoCoPeLia${FUNC}RunnerBestOld
	cuBLASXt_run=$LIBSC_DIR/testing-bin/cuBLASXt${FUNC}Runner
	BLASX_run=$LIBSC_DIR/testing-bin/BLASx${FUNC}Runner
	BLASXEX_run=$LIBSC_DIR/testing-bin/BLASxEx${FUNC}Runner
	XKBLAS_run=$LIBSC_DIR/testing-bin/XKBLAS${FUNC}Runner
	echo "Performing Benchmarks for ${FUNC} evaluation..."
	## A = Full offload scenario (Initially all data on CPU), B = Partial offload scenario
	TransA=N
	TransB=N
	for scenario in "A" "B"
	do
		C_loc=-1
		if [ "$scenario" = "A" ];
		then
			A_loc=-1
			B_loc=-1
		else
			if [ $NUM_DEVICES -eq 1 ]
			then
				A_loc=0
				B_loc=0
			elif [ $NUM_DEVICES -ge 2 ]
			then
				A_loc=0
				B_loc=1
			fi
		fi


		# I) Mid-sized square problems
		for Sq in {2048..16384..1024}
		do
			echo "$CoCopelia_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			$CoCopelia_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc  &>> $perf_log
			#echo "$CoCopelia_run_bestest -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			#$CoCopelia_run_bestest -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc  &>> $perf_log
			echo "$CoCopelia_run_best -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			$CoCopelia_run_best -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc  &>> $perf_log
			#echo "$CoCopelia_run_old -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			#$CoCopelia_run_old -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc  &>> $perf_log
			#echo "$CoCopelia_run_best_old -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			#$CoCopelia_run_best_old -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc  &>> $perf_log
			#echo "$cuBLASXt_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			#$cuBLASXt_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log
			#echo "$BLASX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			#$BLASX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log
			#echo "$BLASXEX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			#$BLASXEX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log
			#echo "LD_PRELOAD=/home/users/panastas/PhD_stuff/Other-libs/xkblas-silver1/install/lib/libxkblas_blaswrapper.so XKBLAS_GPUSET=3 XKBLAS_CACHE_LIMIT=80  $XKBLAS_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			#LD_PRELOAD=/home/users/panastas/PhD_stuff/Other-libs/xkblas-silver1/install/lib/libxkblas_blaswrapper.so XKBLAS_GPUSET=3 XKBLAS_CACHE_LIMIT=80  $XKBLAS_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log
		done
		exit 1
		# II) Large-sized square problems (usually >> GPU memory)
		for Sq in {16384..40000..4096}
		do
			echo "$CoCopelia_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			$CoCopelia_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc  &>> $perf_log
			#echo "$CoCopelia_run_bestest -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			#$CoCopelia_run_bestest -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc  &>> $perf_log
			#echo "$CoCopelia_run_best -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			#$CoCopelia_run_best -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc  &>> $perf_log
			#echo "$CoCopelia_run_old -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			#$CoCopelia_run_old -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc  &>> $perf_log
			#echo "$CoCopelia_run_best_old -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			#$CoCopelia_run_best_old -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc  &>> $perf_log
			#echo "$cuBLASXt_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			#$cuBLASXt_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log
			#echo "$BLASX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			#$BLASX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log
			#echo "$BLASXEX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			#$BLASXEX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log
			#echo "LD_PRELOAD=/home/users/panastas/PhD_stuff/Other-libs/xkblas-silver1/install/lib/libxkblas_blaswrapper.so XKBLAS_GPUSET=3 XKBLAS_CACHE_LIMIT=80  $XKBLAS_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
			#LD_PRELOAD=/home/users/panastas/PhD_stuff/Other-libs/xkblas-silver1/install/lib/libxkblas_blaswrapper.so XKBLAS_GPUSET=3 XKBLAS_CACHE_LIMIT=80  $XKBLAS_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $perf_log
		done
		# III) Non-square Problems
		for inbalanced in 8192 16384
		do
			# K < M,N
			for ctr in 3 4 5 6 7 8 9 10;
			do
				fat=$(($inbalanced*$ctr/2))
				double_thin=$(($inbalanced*4/$ctr/$ctr))

				echo "$CoCopelia_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $fat $fat $double_thin $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
				$CoCopelia_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $fat $fat $double_thin $A_loc $B_loc $C_loc $C_loc &>> $perf_log
				echo "$cuBLASXt_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $fat $fat $double_thin $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
				$cuBLASXt_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $fat $fat $double_thin $A_loc $B_loc $C_loc $C_loc &>> $perf_log
				echo "$BLASX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $fat $fat $double_thin $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
				$BLASX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $fat $fat $double_thin $A_loc $B_loc $C_loc $C_loc &>> $perf_log
				echo "$BLASXEX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $fat $fat $double_thin $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
				$BLASXEX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $fat $fat $double_thin $A_loc $B_loc $C_loc $C_loc &>> $perf_log
				echo "LD_PRELOAD=/home/users/panastas/PhD_stuff/Other-libs/xkblas-silver1/install/lib/libxkblas_blaswrapper.so XKBLAS_GPUSET=3 XKBLAS_CACHE_LIMIT=80  $XKBLAS_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $fat $fat $double_thin $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
				LD_PRELOAD=/home/users/panastas/PhD_stuff/Other-libs/xkblas-silver1/install/lib/libxkblas_blaswrapper.so XKBLAS_GPUSET=3 XKBLAS_CACHE_LIMIT=80  $XKBLAS_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $fat $fat $double_thin $A_loc $B_loc $C_loc $C_loc &>> $perf_log
			done
			# K > M,N
			for ctr in 3 4 5 6 7 8 9 10;
			do
				double_fat=$(($inbalanced*$ctr*$ctr/4))
				thin=$(($inbalanced*2/$ctr))

				echo "$CoCopelia_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $thin $thin $double_fat $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
				$CoCopelia_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $thin $thin $double_fat $A_loc $B_loc $C_loc $C_loc &>> $perf_log
				echo "$cuBLASXt_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $thin $thin $double_fat $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
				$cuBLASXt_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $thin $thin $double_fat $A_loc $B_loc $C_loc $C_loc &>> $perf_log
				echo "$BLASX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $thin $thin $double_fat $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
				$BLASX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $thin $thin $double_fat $A_loc $B_loc $C_loc $C_loc &>> $perf_log
				echo "$BLASXEX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $thin $thin $double_fat $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
				$BLASXEX_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $thin $thin $double_fat $A_loc $B_loc $C_loc $C_loc &>> $perf_log
				echo "LD_PRELOAD=/home/users/panastas/PhD_stuff/Other-libs/xkblas-silver1/install/lib/libxkblas_blaswrapper.so XKBLAS_GPUSET=3 XKBLAS_CACHE_LIMIT=80  $XKBLAS_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $thin $thin $double_fat $A_loc $B_loc $C_loc $C_loc &>> $perf_log"
				LD_PRELOAD=/home/users/panastas/PhD_stuff/Other-libs/xkblas-silver1/install/lib/libxkblas_blaswrapper.so XKBLAS_GPUSET=3 XKBLAS_CACHE_LIMIT=80  $XKBLAS_run -1 -1 -1 -1 $TransA $TransB $alpha $beta $thin $thin $double_fat $A_loc $B_loc $C_loc $C_loc &>> $perf_log
			done
		done
	done
	echo "Done"
done
