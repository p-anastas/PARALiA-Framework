#!/bin/bash

DEPLOY_DIR=/tmp/tmp.YXAEzsRmOZ/cmake-build-debug/silver1-install/Deployment_phase

devices=2

for (( i = 0; i < $devices; i++ ))
do 
	$DEPLOY_DIR/RunMicrobenchmarks.sh $i
	$DEPLOY_DIR/ProcessDatabase.sh $i
done


