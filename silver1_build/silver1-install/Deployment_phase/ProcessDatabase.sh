#!/bin/bash

DEPLOY_DIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/Deployment_phase
PYTHON_DIR=${DEPLOY_DIR}/Python-DataManage
R_DIR=${DEPLOY_DIR}/R-statistics

BACKEND=CuCuBLAS

machine=silver1
device=$1


RES_DIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/silver1_build/silver1-install/Deployment_phase/Database/Benchmark-Results
OUTDIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/silver1_build/silver1-install/Deployment_phase/Database/Processed

if [[ "$BACKEND" == *"CuCuBLAS"* ]];then
	library_id=0
fi

python3 $PYTHON_DIR/BenchToLookupTables.py -i $RES_DIR -o $OUTDIR -f Dgemm -d $device -l $library_id
python3 $PYTHON_DIR/BenchToLookupTables.py -i $RES_DIR -o $OUTDIR -f Sgemm -d $device -l $library_id

Rscript --vanilla $R_DIR/transfer_models.r $RES_DIR $OUTDIR $device $library_id


