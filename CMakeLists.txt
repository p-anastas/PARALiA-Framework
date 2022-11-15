cmake_minimum_required (VERSION 3.10)

## Define a name for the backend library wrappers/implementations used in this build, Currently implemented only with CuCuBLAS(==CUDA+CUBLAS)
set(BACKEND "CuCuBLAS")
message("CoCopeLia-Framework - BACKEND: " ${BACKEND})
add_definitions(-DBACKEND="${BACKEND}")
## Define the project version (used in logfiles for result testing)

set(PROJECT_VERSION "1.4")
message("CoCopeLia-Framework - PROJECT_VERSION: " ${PROJECT_VERSION})
add_definitions(-DVERSION="${PROJECT_VERSION}")

## Enable Asynchronous execution for Subkernels - CoCo overlap
add_definitions(-DASYNC_ENABLE)

## Enable GPU backend resourse reuse (streams, handles)
add_definitions(-DBACKEND_RES_REUSE_ENABLE)

## Enable GPU and CPU allocated buffer reuse between different calls
add_definitions(-DBUFFER_REUSE_ENABLE)

## Enable lock-free versions of event & Queue management in Universal_helpers
add_definitions(-DUNIHELPER_LOCKFREE_ENABLE)

## Enable Locking with mutexes instead of busy-waiting
#add_definitions(-DENABLE_MUTEX_LOCKING)

## Enable Running multiple backend operations in parallel for each queue
add_definitions(-DENABLE_PARALLEL_BACKEND)
## The maximum allowed parallel backend operations per queue
add_definitions(-DMAX_BACKEND_L=4)

## Enable overlapping Send & recv Communication
add_definitions(-DENABLE_SEND_RECV_OVERLAP)

## Enable using the CPU as an extra device for part of the total workload, if applicable
add_definitions(-DENABLE_CPU_WORKLOAD)

## Enable using a different parallel worker for scheduling each tile request
#add_definitions(-DENABLE_PTHREAD_TILE_REQUEST)

## Enable pre-fetching (cheap) tiles for future subkernels
#add_definitions(-DENABLE_TILE_PREFETCH)

## Enable using transfer hops for faster transfers between multiple links
add_definitions(-DENABLE_TRANSFER_HOPS)
## Enable using transfer hops for W tiles as well (no data reuse, but potentially better BW)
#add_definitions(-DENABLE_TRANSFER_W_HOPS)
## Enable using heuristic-based W-hops (and Send-Recv Over) activation
#add_definitions(-DENABLE_PREDICT_HOP_MODE)

## Enable sharing of bandwidth between links that utilize the same underlying hardware communication resources
add_definitions(-DENABLE_LINK_BW_SHARING) # by putting them on different queues

## TODO: EXPERMENTAL: Enable power measuring for benchmarks, and energy-related prediction modes for the Autotuner
add_definitions(-DENABLE_POWA)
add_definitions(-DCPU_W_PREDEF=300) # TODO: Vulcan CAUTION!!! Define this for your CPU - no CPU power tool installed.
#add_definitions(-DCPU_W_PREDEF=170) # TODO: silver1 CAUTION!!! Define this for your CPU - no CPU power tool installed.
## Choose the Unit combination that maximizes total performance (minimum time).
#add_definitions(-DPREDICT_OPTIMIZE_TARGET="PERF")
## Choose the Unit combination that minimizes consumed energy (J).
## Rarely chooses multi-unit execution, since its almost always less energy-efficient than single-unit.
#add_definitions(-DPREDICT_OPTIMIZE_TARGET="ENERGY")
#add_definitions(-DPREDICT_OPTIMIZE_TARGET="POWER-DELAY")
add_definitions(-DPREDICT_OPTIMIZE_TARGET="ENERGY-DELAY")
#add_definitions(-DPREDICT_OPTIMIZE_TARGET="OTHER_TBD")

## Define at which percentage to normalize float values for similar data (e.g. costs, splits etc) to avoid empirical errors leaking into decisions
## For example see Autotuning_runtime->CoCoPeLiaModel->CoCoPeLiaNormalizeSplit
add_definitions(-DNORMALIZE_NEAR_SPLIT_LIMIT=0.05)

## Define the minimum allowed percentile contribution in total time from each potential unit
## In case of energy, the maximum allowed percentile energy increase by each new unit.
add_definitions(-DMINIMUM_UNIT_CONTRIBUTION=0.05)

set(SCHEDULE_TYPE 0) # 0 = static, 1 = runtime selection

if (${SCHEDULE_TYPE} EQUAL 0)
  ## Define which subkernel distribution to use
  #add_definitions(-DDISTRIBUTION="ROUND-ROBIN")
  #add_definitions(-DDISTRIBUTION="SPLIT-NAIVE")
  #add_definitions(-DDISTRIBUTION="SPLIT-CHUNKS-ROBIN")
  #add_definitions(-DDISTRIBUTION="SPLIT-CHUNKS-ROBIN-REVERSE")
  add_definitions(-DDISTRIBUTION="2D-BLOCK-CYCLIC")
elseif(${SCHEDULE_TYPE} EQUAL 1)
  add_definitions(-DDISTRIBUTION="ROUND-ROBIN")
  add_definitions(-DRUNTIME_SCHEDULER_VERSION)
else()
endif()

#Run subkernels in provided order
add_definitions(-DSERIAL_SUBKERNEL_SELECTION)

## Define subkernel selection heuristic values. 0 = No penalty
add_definitions(-DMULTIFETCH_PENALTY=2)
add_definitions(-DFETCH_UNAVAILABLE_PENALTY=0.15)
add_definitions(-DWTILE_TRANSFER_PENALTY=1)
add_definitions(-DPARALLELBBOTLENECK_PENALTY=0.5)
add_definitions(-DEXSTEAL_PENALTY=10)

#Run in testing mode : details about models, decisions and time spend in operations.
#add_definitions(-DTEST)

#Run in Subkernel testing mode : details about time spend in subkernel operations.
#add_definitions(-DSTEST)

#Run in Deep Subkernel testing mode : many details about time spend in subkernel operations.
#add_definitions(-DDSTEST)

#Run in Link Hop testing mode; transfers using link hops are logged and timed.
#add_definitions(-DTTEST)

#Run in debug mode, which includes detailed function calls and info for error-checking
#add_definitions(-DDEBUG)

#Run in deep debug mode, for command-to-command debugging
#add_definitions(-DDDEBUG)

#Run Prediction debug mode (autotuning runtime stuff)
#add_definitions(-DPDEBUG)

#Run Prediction deep debug mode (all autotuning runtime stuff)
#add_definitions(-DDPDEBUG)

#Run in caching debug mode, for cache-related debugging
#add_definitions(-DCDEBUG)

#Run in unihelper debug mode, for unihelper-related debugging
#add_definitions(-DUDEBUG)

#Run in unihelper deep debug mode, for unihelper-related debugging
#add_definitions(-DUDDEBUG)

## Define a name for the testbed, to be used for the whole framework setup
#set(TESTBED_NAME "silver1") # TODO: silver1
set(TESTBED_NAME "vulcan")  # TODO: Vulcan
message("CoCopeLia-Framework - TESTBED_NAME: " ${TESTBED_NAME})
add_definitions(-DTESTBED="${TESTBED_NAME}")

## Define the (max) number of devices for utilization (0-num_devices)
#set(NUM_LOCATIONS 4) # TODO: silver1
set(NUM_LOCATIONS 9) # TODO: Vulcan
message("CoCopeLia-Framework - NUM_LOCATIONS: " ${NUM_LOCATIONS})
add_definitions(-DLOC_NUM=${NUM_LOCATIONS})

## Define the max GPU memory percentage (%) a problem can use ( not accounting for streams and other control structs; e.g. only the problem data ).
set(PROBLEM_GPU_PERCENTAGE 70)
message("CoCopeLia-Framework - PROBLEM_GPU_PERCENTAGE: " ${PROBLEM_GPU_PERCENTAGE})
add_definitions(-DPROBLEM_GPU_PERCENTAGE=${PROBLEM_GPU_PERCENTAGE})

###TODO:Define specific compiler, flags, includes, libs for the target system (IF needed).

## 1) C++

#	a) auto

#	b) compile flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -lm -Wuninitialized")

message("CoCopeLia-Framework - CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

#	c) auto

#	d) auto

## 2) CUCUBLAS TODO: Other backends?
#set (CUDA_PREFIX "/usr/local/cuda-11.6") # TODO: Silver1
add_definitions(-DCUDA_VER=1100)
#set (CUDA_PREFIX "/usr/local/cuda-9.2")
#add_definitions(-DCUDA_VER=920)
set (CUDA_PREFIX "/opt/system/cuda/11.7.0") # TODO: Vulcan
set (CXX_PREFIX "/opt/hlrs/non-spack/2022-02/compiler/gcc/11.2.0") # TODO: Vulcan


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70) #35 for Tesla K40, 70 for V100, 61 for GTX 1060/70
endif()

# 	a) Compiler
set (CMAKE_CUDA_COMPILER ${CUDA_PREFIX}/bin/nvcc)
set (CMAKE_CXX_COMPILER ${CXX_PREFIX}/bin/c++) # TODO: Vulcan
set (CMAKE_C_COMPILER ${CXX_PREFIX}/bin/gcc) # TODO: Vulcan

# 	b) Compile flags
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w -O3")
message("CoCopeLia-Framework - CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})
#set(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS})


# 	c) Includes
set (CUDA_INCLUDE_DIRS ${CUDA_PREFIX}/include ${CUDA_PREFIX}/samples/common/inc)
include_directories(${CUDA_INCLUDE_DIRS})
message("CoCopeLia-Framework - CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})

#set(OPENBLAS_PREFIX "/home/users/panastas/Lib_install/OpenBLAS-gcc93") # TODO: silver1
set(OPENBLAS_PREFIX "/zhome/academic/HLRS/xex/xexpanas/OpenBLAS_install_vulcan-11.2") # TODO: Vulcan
set (OPENBLAS_INCLUDE_DIRS ${OPENBLAS_PREFIX}/include)
include_directories(${OPENBLAS_INCLUDE_DIRS})
message("CoCopeLia-Framework - OPENBLAS_INCLUDE_DIRS: " ${OPENBLAS_INCLUDE_DIRS})

# 	d) Link Libraries
set (CUDA_LD ${CUDA_PREFIX}/lib64) # or /lib depending on system
#set (CUDA_LINK "-L${CUDA_LD} -lcudart -lcublas -lcurand -lcuda") # TODO: silver1
set (CUDA_LINK "-L${CUDA_LD} -lcudart -lcublas -lcurand ") # TODO: Vulcan /opt/system/nvidia/ALL.ALL.510.73.08/libcuda.so.510.73.08 ?
message("CoCopeLia-Framework - CUDA_LINK: " ${CUDA_LINK} )
set (INTERNAL_LINK_LIB "${CUDA_LINK} -lrt -ldl")
message("CoCopeLia-Framework - INTERNAL_LINK_LIB: " ${INTERNAL_LINK_LIB})
set (OPENBLAS_LD ${OPENBLAS_PREFIX}/lib)
set (OPENBLAS_LINK  "-L${OPENBLAS_LD} -lopenblas" )
message("CoCopeLia-Framework - OPENBLAS_LINK: " ${OPENBLAS_LINK} )

project (CoCopeLia-Framework VERSION ${PROJECT_VERSION} DESCRIPTION "The complete CoCoPeLia Framework")

if ("${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr/local")
message("CoCopeLia-Framework - CMAKE_INSTALL_PREFIX not provided (with --install-prefix <directory>) - project will be installed at default build dir.")
 set (CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${TESTBED_NAME}-install)
endif()
set (CMAKE_INSTALL_PROJECT ${CMAKE_INSTALL_PREFIX})
message("CoCopeLia-Framework - CMAKE_INSTALL_PROJECT: " ${CMAKE_INSTALL_PROJECT})

configure_file(${PROJECT_SOURCE_DIR}/Deploy.in ${CMAKE_INSTALL_PROJECT}/Deploy.sh @ONLY)

set(PROJECT_BENCH_DIR ${PROJECT_SOURCE_DIR}/Benchmarking)
configure_file(${PROJECT_BENCH_DIR}/Test_correctness.in ${CMAKE_INSTALL_PROJECT}/Test_correctness.sh @ONLY)
configure_file(${PROJECT_BENCH_DIR}/Targeted_performance_evaluation.in ${CMAKE_INSTALL_PROJECT}/Targeted_performance_evaluation.sh @ONLY)
configure_file(${PROJECT_BENCH_DIR}/Full_model_validation.in ${CMAKE_INSTALL_PROJECT}/Full_model_validation.sh @ONLY)
configure_file(${PROJECT_BENCH_DIR}/Full_performance_evaluation.in ${CMAKE_INSTALL_PROJECT}/Full_performance_evaluation.sh @ONLY)

set(VALUETYPEID 1)
if (${VALUETYPEID} EQUAL 0)
  add_definitions(-DVALUE_TYPE=float)
elseif(${VALUETYPEID} EQUAL 1)
  add_definitions(-DVALUE_TYPE=double)
else()
endif()

ADD_SUBDIRECTORY (Universal_helpers)

ADD_SUBDIRECTORY (Deployment_phase)
add_definitions(-DDEPLOYDB="${DEPLOYMENT_DIR_DB}")

ADD_SUBDIRECTORY (Autotuning_runtime)
ADD_SUBDIRECTORY (Library_scheduler)


ADD_SUBDIRECTORY (Benchmarking)
