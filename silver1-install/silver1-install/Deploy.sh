#!/bin/bash

DEPLOY_DIR=/home/users/panastas/PhD_stuff/CoCoPeLia-Framework/silver1-install/silver1-install/Deployment_phase

devices=4

for (( i = -1; i < $devices -1; i++ ))
do
	$DEPLOY_DIR/RunMicrobenchmarks.sh $i
	$DEPLOY_DIR/ProcessDatabase.sh $i
done
