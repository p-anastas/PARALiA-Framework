cmake_minimum_required (VERSION 3.10)

## Define a name for the backend library wrappers/implementations used in this build, Currently implemented only with CuCuBLAS(==CUDA+CUBLAS)
set(BACKEND "CuCuBLAS")
message("CoCopeLia-Framework - BACKEND: " ${BACKEND})
add_definitions(-DBACKEND="${BACKEND}")
## Define the project version (used in logfiles for result testing)

set(PROJECT_VERSION "1.3")
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
#add_definitions(-DENABLE_CPU_WORKLOAD)

## Enable using the CPU as an extra device for part of the total workload, if applicable
#add_definitions(-DENABLE_PTHREAD_TILE_REQUEST)

set(SCHEDULE_TYPE 1) # 0 = static, 1 = runtime selection

if (${SCHEDULE_TYPE} EQUAL 0)
  ## Define which subkernel distribution to use
  #add_definitions(-DDISTRIBUTION="ROUND-ROBIN")
  #add_definitions(-DDISTRIBUTION="SPLIT-NAIVE")
  add_definitions(-DDISTRIBUTION="SPLIT-CHUNKS-ROBIN")
  #add_definitions(-DDISTRIBUTION="SPLIT-CHUNKS-ROBIN-REVERSE")
  #add_definitions(-DDISTRIBUTION="2D-BLOCK-CYCLIC")
elseif(${SCHEDULE_TYPE} EQUAL 1)
  add_definitions(-DDISTRIBUTION="ROUND-ROBIN")
  add_definitions(-DRUNTIME_SCHEDULER_VERSION)
else()
endif()

## Define which subkernel selection heuristic to use
#FIXME: NO-WRITE-SHARE do not work with set number of subkernels, only for correctness checking.
#add_definitions(-DSELECT_HEURISTIC="NAIVE-NO-WRITE-SHARE")
#add_definitions(-DSELECT_HEURISTIC="MINIMIZE-FETCH-NO-WRITE-SHARE-MULTIFETCH-PENALTY-MULTIDEV-FAIR")

#add_definitions(-DSELECT_HEURISTIC="NAIVE")
#add_definitions(-DSELECT_HEURISTIC="MINIMIZE-FETCH")
#add_definitions(-DSELECT_HEURISTIC="MINIMIZE-FETCH-WRITE-PENALTY")
add_definitions(-DSELECT_HEURISTIC="MINIMIZE-FETCH-WRITE-PENALTY-MULTIFETCH-PENALTY")
## This last mode is pointless for static schedule type
#add_definitions(-DSELECT_HEURISTIC="MINIMIZE-FETCH-WRITE-PENALTY-MULTIFETCH-PENALTY-MULTIDEV-FAIR")

# Extra distribution-related values
add_definitions(-DPARALLEL_BACKEND_SET_PENALTY=0.25)
add_definitions(-DWTILE_TRANSFER_PENALTY=2)
add_definitions(-DMULTIFETCH_PENALTY=1)

## Define at which percentage to normalize float values for similar data (e.g. costs, splits etc) to avoid empirical errors leaking into decisions
## For example see Autotuning_runtime->CoCoPeLiaModel->CoCoPeLiaNormalizeSplit
add_definitions(-DNORMALIZE_NEAR_SPLIT_LIMIT=0.05)

#Run in testing mode : details about models, decisions and time spend in operations.
add_definitions(-DTEST)

#Run in Subkernel testing mode : details about time spend in subkernel operations.
add_definitions(-DSTEST)

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
#set(TESTBED_NAME "dungani")
set(TESTBED_NAME "silver1")
#set(TESTBED_NAME "vulcanGPUs")
message("CoCopeLia-Framework - TESTBED_NAME: " ${TESTBED_NAME})
add_definitions(-DTESTBED="${TESTBED_NAME}")


## Define the (max) number of devices for utilization (0-num_devices)
set(NUM_DEVICES 3)
message("CoCopeLia-Framework - NUM_DEVICES: " ${NUM_DEVICES})
add_definitions(-DDEV_NUM=${NUM_DEVICES})

## Define the max GPU memory percentage (%) a problem can use ( not accounting for streams and other control structs; e.g. only the problem data ).
set(PROBLEM_GPU_PERCENTAGE 80)
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
set (CUDA_PREFIX "/usr/local/cuda-11.6")
add_definitions(-DCUDA_VER=1100)
#set (CUDA_PREFIX "/usr/local/cuda-9.2")
#add_definitions(-DCUDA_VER=920)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70) #35 for Tesla K40, 70 for V100, 61 for GTX 1060/70
endif()

# 	a) Compiler
set (CMAKE_CUDA_COMPILER ${CUDA_PREFIX}/bin/nvcc)
# 	b) Compile flags
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w -O3")
message("CoCopeLia-Framework - CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})
#set(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS})


# 	c) Includes
set (CUDA_INCLUDE_DIRS ${CUDA_PREFIX}/include ${CUDA_PREFIX}/samples/common/inc)
include_directories(${CUDA_INCLUDE_DIRS})
message("CoCopeLia-Framework - CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})

set(OPENBLAS_PREFIX "/home/users/panastas/Lib_install/OpenBLAS-gcc93")
set (OPENBLAS_INCLUDE_DIRS ${OPENBLAS_PREFIX}/include)
include_directories(${OPENBLAS_INCLUDE_DIRS})
message("CoCopeLia-Framework - OPENBLAS_INCLUDE_DIRS: " ${OPENBLAS_INCLUDE_DIRS})

# 	d) Link Libraries
set (CUDA_LD ${CUDA_PREFIX}/lib64) # or /lib depending on system
set (CUDA_LINK "-L${CUDA_LD} -lcudart -lcublas -lcurand -lcuda") #/opt/system/nvidia/tesla.v100.440.95.01/libcuda.so.440.95.01
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
