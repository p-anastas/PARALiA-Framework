#!/bin/bash

PROJECT_INSTALL_DIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/silver1-install/silver1-install

BACKEND=
machine=silver1

device=$1

AUTOTUNE_DIR=$PROJECT_INSTALL_DIR/Autotuning_Runtime
AUTOTUNE_TEST_LOG_DIR=$AUTOTUNE_DIR/testLogs

mkdir "${AUTOTUNE_TEST_LOG_DIR}/exec_logs"
mkdir "${AUTOTUNE_TEST_LOG_DIR}/predictions"
mkdir "${AUTOTUNE_TEST_LOG_DIR}/validation_runs"

CoCopelia_pred="$AUTOTUNE_DIR/bin/CoCoPeLiaPredictTileTester"

for FUNC in Dgemm Sgemm
do
	echo "Generating predictions for CoCopelia ${FUNC} validation..."
	rm ${AUTOTUNE_TEST_LOG_DIR}/predictions/CoCoPeLiaLogPrediction-${FUNC}_dev-${device}.log-backup
	mv ${AUTOTUNE_TEST_LOG_DIR}/predictions/CoCoPeLiaLogPrediction-${FUNC}_dev-${device}.log ${AUTOTUNE_TEST_LOG_DIR}/predictions/CoCoPeLiaLogPrediction-${FUNC}_dev-${device}.log-backup
	test_log=${AUTOTUNE_TEST_LOG_DIR}/exec_logs/${FUNC}_val_pred.log
	rm $test_log
	for TransA in N T;
	do
		for TransB in N T;
		do
			for A_loc in 1 0;
			do
				for B_loc in 1 0;
				do
					for C_loc in 1 0;
					do
						for Sq in {4096..16384..4096}
						do 
							echo "$CoCopelia_pred $device $FUNC X $TransA $TransB $Sq $Sq $Sq $A_loc $B_loc $C_loc $A_loc $B_loc $C_loc $Sq $Sq $Sq &>> $test_log"
							$CoCopelia_pred $device $FUNC X $TransA $TransB $Sq $Sq $Sq $A_loc $B_loc $C_loc $A_loc $B_loc $C_loc $Sq $Sq $Sq &>> $test_log
						done
					done 
				done
			done
		done
	done
	A_loc=1
	B_loc=1
	C_loc=1
	for TransA in N T;
	do
		for TransB in N T;
		do
			for inbalanced in {4096..16384..4096}
			do 
				for ctr in 3 4 5 #2 3 4 5 6 7 8; # testbedI for 12000 can't do 7,8
				do 
					fat=$(($inbalanced*$ctr/2))
					double_thin=$(($inbalanced*4/$ctr/$ctr))

					echo "$CoCopelia_pred $device $FUNC X $TransA $TransB $fat $fat $double_thin $A_loc $B_loc $C_loc $A_loc $B_loc $C_loc $fat $fat $fat&>> $test_log"
					$CoCopelia_pred $device $FUNC X $TransA $TransB $fat $fat $double_thin $A_loc $B_loc $C_loc $A_loc $B_loc $C_loc $fat $fat $fat&>> $test_log
				done
			
				for ctr in 3 4 5; #2 3 4 5 6 7 8 9 10 11 12 13 14 15 16;
				do
					double_fat=$(($inbalanced*$ctr*$ctr/4))
					thin=$(($inbalanced*2/$ctr))

					echo "$CoCopelia_pred $device $FUNC X $TransA $TransB $thin $thin $double_fat $A_loc $B_loc $C_loc $A_loc $B_loc $C_loc $double_fat $double_fat $double_fat &>> $test_log"
					$CoCopelia_pred $device $FUNC X $TransA $TransB $thin $thin $double_fat $A_loc $B_loc $C_loc $A_loc $B_loc $C_loc $double_fat $double_fat $double_fat &>> $test_log
				done
			done
		done
	done
	echo "${FUNC} predictions Done"
done 

#exit 1

LIBSC_DIR=$PROJECT_INSTALL_DIR/Benchmarking
device_token=$((10**$device))
cpu_ratio=0
alpha=1.2345
beta=1.1154

for FUNC in Dgemm #Sgemm 
do
	CoCopelia_run=$LIBSC_DIR/testing-bin/CoCoPeLia${FUNC}Runner
	#cuBLASXt_run=$LIBSC_DIR/testing-bin/cuBLASXt${FUNC}Runner
	test_log=${AUTOTUNE_TEST_LOG_DIR}/validation_runs/${FUNC}_val_set.log
	mv $test_log $test_log-backup
	echo "Generating set for CoCopelia ${FUNC} validation..."
	for TransA in N T;
	do
		for TransB in N T;
		do
			for A_loc in -1 $device;
			do
				for B_loc in -1 $device;
				do
					for C_loc in -1 $device;
					do
						for T in {512..16384..512}
						do 
							for Sq in {4096..16384..4096}
							do 
								echo "$CoCopelia_run 1 $device_token $T $cpu_ratio $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $test_log"
								$CoCopelia_run 1 $device_token $T $cpu_ratio $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $test_log
								#echo "$cuBLASXt_run 1 $device_token $T $cpu_ratio $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $test_log"
								#$cuBLASXt_run 1 $device_token $T $cpu_ratio  $TransA $TransB $alpha $beta $Sq $Sq $Sq $A_loc $B_loc $C_loc $C_loc &>> $test_log
							done

						done

					done 
				done
			done
		done
	done
	A_loc=-1
	B_loc=-1
	C_loc=-1
	for TransA in N T;
	do
		for TransB in N T;
		do
			for T in {512..16384..512}
			do 
				for inbalanced in {4096..16384..4096}
				do 
					for ctr in 3 4 5 #2 3 4 5 6 7 8; # testbedI for 12000 can't do 7,8
					do 
						fat=$(($inbalanced*$ctr/2))
						double_thin=$(($inbalanced*4/$ctr/$ctr))

						echo "$CoCopelia_run 1 $device_token $T $cpu_ratio $TransA $TransB $alpha $beta $fat $fat $double_thin $A_loc $B_loc $C_loc $C_loc &>> $test_log"
						$CoCopelia_run 1 $device_token $T $cpu_ratio $TransA $TransB $alpha $beta $fat $fat $double_thin $A_loc $B_loc $C_loc $C_loc $T &>> $test_log
						#echo "$cuBLASXt_run $device $TransA $TransB $alpha $beta $fat $fat $double_thin $A_loc $B_loc $C_loc $C_loc &>> $test_log"
						#$cuBLASXt_run $device $TransA $TransB $alpha $beta $fat $fat $double_thin $A_loc $B_loc $C_loc $C_loc &>> $test_log
					done
			
					for ctr in 3 4 5; #2 3 4 5 6 7 8 9 10 11 12 13 14 15 16;
					do
						double_fat=$(($inbalanced*$ctr*$ctr/4))
						thin=$(($inbalanced*2/$ctr))

						echo "$CoCopelia_run 1 $device_token $T $cpu_ratio $TransA $TransB $alpha $beta $thin $thin $double_fat $A_loc $B_loc $C_loc $C_loc &>> $test_log"
						$CoCopelia_run 1 $device_token $T $cpu_ratio $TransA $TransB $alpha $beta $thin $thin $double_fat $A_loc $B_loc $C_loc $C_loc &>> $test_log
						#echo "$cuBLASXt_run 1 $device_token $T $cpu_ratio $TransA $TransB $alpha $beta $thin $thin $double_fat $A_loc $B_loc $C_loc $C_loc &>> $test_log"
						#$cuBLASXt_run 1 $device_token $T $cpu_ratio $TransA $TransB $alpha $beta $thin $thin $double_fat $A_loc $B_loc $C_loc $C_loc &>> $test_log
					done
				done
			done
		done
	done
done
echo "Done"
