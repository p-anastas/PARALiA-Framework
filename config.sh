#!/bin/bash

## This is the configuration template file for PARALiA compilation and deployment in a new system.
## The intended use is to create one such config for each different deplyed system (e.g. config_sys1.sh, config_sys2.sh etc) in order to allow easy deployment.

#--------------------------------------------------------------Basic----------------------------------------------------------------#

# CHECKME: A desired name for the testbed to be used for your build-dirs and logfiles.
system='system_name'
export PARALIA_SYSTEM=${system}

# CHECKME: Define cuda architecture 80") # (Tesla K40 = 35, GTX 1060/70 = 61,) P100 = 60, V100 = 70, A100 = 80
export PARALIA_CUDA_ARCH=42

# CHECKME: Define cuda toolkit path. If left 'default' PARALIA will try to use the default compiler calls without prefix(es)
export PARALIA_CUDA_TOOLKIT_PREFIX='default'

# CHECKME: Define gcc compiler path. If left 'default' PARALIA will try to use the default compiler calls without prefix(es)
export PARALIA_CXX_PREFIX='default'

# CHECKME: Define path for prebuild openblas. NOTE: OpenBLAS built using the same gcc is adviced.
export PARALIA_OPENBLAS_PREFIX='fillme'

# CHECKME: Define path for prebuild openblas. NOTE: OpenBLAS built using the same gcc is adviced.
export PARALIA_BLASX_PREFIX="fillme"

# CHECKME: Define path for prebuild openblas. NOTE: OpenBLAS built using the same gcc is adviced.
export PARALIA_XKBLAS_PREFIX="fillme"

# CHECKME: Define the directory PARALIA will be installed in. 'default' = buildir/${system}-install
export PARALIA_INSTALL_PREFIX='default'

#---------------------------------------------------------------Extra--------------------------------------------------------------#

# CHECKME: Define the Watt for your CPU - no CPU power tool used currently.
export PARALIA_W_CPU_PREDEF=42
