#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

DEPLOY_DIR="/home/petyros/mount/PhD_stuff/CoCoPeLia-Framework/Deployment_phase"
#/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/Deployment_phase
BACKEND=CuCuBLAS

machine=$1
device=$2

PYTHON_DIR=${DEPLOY_DIR}/Python-DataManage
R_DIR=${DEPLOY_DIR}/R-statistics
RES_DIR=${DEPLOY_DIR}/${BACKEND}/${machine}-install/Benchmark-Results
DB_DR=${DEPLOY_DIR}/${BACKEND}/${machine}-install/Database

if [[ "$BACKEND" == *"CuCuBLAS"* ]];then
	library_id=0
fi

python3 $PYTHON_DIR/BenchToLookupTables.py -i $RES_DIR -f Dgemm -d $device -l $library_id
python3 $PYTHON_DIR/BenchToLookupTables.py -i $RES_DIR -f Sgemm -d $device -l $library_id

Rscript --vanilla $R_DIR/transfer_models.r $RES_DIR $device $library_id


