#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

PROJECT_INSTALL_DIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/silver1-install/silver1-install

BACKEND=
machine=silver1

LIBSC_DIR=$PROJECT_INSTALL_DIR/Benchmarking

# Set to 0 for basic correctness testing or 1 for also testing huge sizes ( > GPU mem). Setting to 1 might lead to very long testing times depending on the system.
RUN_LARGE_FLAG=0

for FUNC in Dgemm #Sgemm 
do	
	CoCopelia_test=$LIBSC_DIR/testing-bin/CoCoPeLia${FUNC}Tester
	$CoCopelia_test $RUN_LARGE_FLAG
done





