#!/bin/bash

## This is the configuration template file for PARALiA compilation and deployment in a new system.
## The intended use is to create one such config for each different deplyed system (e.g. config_sys1.sh, config_sys2.sh etc) in order to allow easy deployment.

#--------------------------------------------------------------Basic----------------------------------------------------------------#

# CHECKME: A desired name for the testbed to be used for your build-dirs and logfiles.
system="karol_8A100"
export PARALIA_SYSTEM=${system}

# CHECKME: A folder to change to before firing benchmarks, to avoid disk errors. 
export CD_FOLDER=/mnt/proj1/dd-23-129

# CHECKME: The number of devices available to PARALiA for utilization. NOTE: If CPU computation is enabled (CMakefile parameter, default = ON) this number should be equal to system_GPUs + 1 !!!
export PARALIA_SYSTEM_DEVNUM=9

# CHECKME: Define cuda architecture # (Tesla K40 = 35, GTX 1060/70 = 61,) P100 = 60, V100 = 70, A100 = 80
export PARALIA_CUDA_ARCH=80

# CHECKME: Define cuda toolkit path. If left 'default' PARALIA will try to use the default compiler calls without prefix(es)
export PARALIA_CUDA_TOOLKIT_PREFIX="/usr/local/cuda-12.2"

# CHECKME: Define cuda load command (e.g. should always be '-lcuda', but its not! If you don't know what to do, leave it)
export PARALIA_CUDA_LOAD_COMMAND='-lcuda'

# CHECKME: Define gcc compiler path. If left 'default' PARALIA will try to use the default compiler calls without prefix(es)
export PARALIA_CXX_PREFIX="default"

# CHECKME: Define path for prebuild openblas. NOTE: OpenBLAS built using the same gcc is adviced.
export PARALIA_OPENBLAS_PREFIX="/apps/modules/numlib/OpenBLAS/"

# CHECKME: Define path for prebuild boost. NOTE: boost built using the same gcc is adviced.
export PARALIA_BOOST_PREFIX="/apps/all/Boost/1.81.0-GCC-12.2.0" # "/home/users/panastas/Lib_install/boost-1.69"

# CHECKME: Also build BLASX benchmarks. NOTE: Requires pre-build BLASX with same compiler(s)
export PARALIA_BLASX_BENCH=0
# CHECKME: Define path for prebuild blasx.
export PARALIA_BLASX_PREFIX="/home/users/panastas/PhD_stuff/Other-libs/BLASX-pinned"

# CHECKME: Also build BLASX benchmarks. NOTE: Requires pre-build XKBLAS with same compiler(s)
export PARALIA_XKBLAS_BENCH=0
# CHECKME: Define path for prebuild xkblas.
export PARALIA_XKBLAS_PREFIX="/home/it4i-panasta/XKBLAS-install"

# CHECKME: Define the directory PARALIA will be installed in. 'default' = buildir/${system}-install
export PARALIA_INSTALL_PREFIX="default"

# CHECKME: Define the Watt for your CPU - no CPU power tool used currently.
export PARALIA_W_CPU_PREDEF=280

#-----------------------------------------------------------------------------------------------------------------------------------#
