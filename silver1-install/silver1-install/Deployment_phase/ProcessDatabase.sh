#!/bin/bash

DEPLOY_DIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/Deployment_phase
PYTHON_DIR=${DEPLOY_DIR}/Python-DataManage

BACKEND=CuCuBLAS

machine=silver1
device=$1

SCRIPT_DIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/silver1-install/silver1-install/Deployment_phase

RES_DIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/silver1-install/silver1-install/Deployment_phase/Database/Benchmark-Results
OUTDIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/silver1-install/silver1-install/Deployment_phase/Database/Processed

mkdir -p $OUTDIR

if [[ "$BACKEND" == *"CuCuBLAS"* ]];then
	library_id=0
fi

echo "python3 $SCRIPT_DIR/BenchToLookupTables.py -i $RES_DIR -o $OUTDIR -f Dgemm -d $device -l $library_id"
python3 $SCRIPT_DIR/BenchToLookupTables.py -i $RES_DIR -o $OUTDIR -f Dgemm -d $device -l $library_id
#python3 $SCRIPT_DIR/BenchToLookupTables.py -i $RES_DIR -o $OUTDIR -f Sgemm -d $device -l $library_id

echo "python3 $SCRIPT_DIR/BenchToLookupTables.py -i $RES_DIR -o $OUTDIR -f Daxpy -d $device -l $library_id"
python3 $SCRIPT_DIR/BenchToLookupTables.py -i $RES_DIR -o $OUTDIR -f Daxpy -d $device -l $library_id

echo "python3 $SCRIPT_DIR/LinkPreprocessing.py -i $RES_DIR -o $OUTDIR -d $device -l $library_id"
python3 $SCRIPT_DIR/LinkPreprocessing.py -i $RES_DIR -o $OUTDIR -d $device -l $library_id
