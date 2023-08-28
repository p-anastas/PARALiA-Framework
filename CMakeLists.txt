cmake_minimum_required (VERSION 3.10)

## Define a name for the backend library wrappers/implementations used in this build, Currently implemented only with CuCuBLAS(==CUDA+CUBLAS)
set(BACKEND "CuCuBLAS")
message("PARALiA-Framework - BACKEND: " ${BACKEND})
add_definitions(-DBACKEND="${BACKEND}")
## Define the project version (used in logfiles for result testing)

execute_process ( COMMAND bash -c "source ${CMAKE_CURRENT_BINARY_DIR}/../config_silver1.sh")

set(PROJECT_VERSION "2.0")
message("PARALiA-Framework - PROJECT_VERSION: " ${PROJECT_VERSION})
add_definitions(-DVERSION="${PROJECT_VERSION}")

## Enable Asynchronous execution for Subkernels
add_definitions(-DASYNC_ENABLE)

## Enable GPU backend resourse reuse (streams, handles)
add_definitions(-DBACKEND_RES_REUSE_ENABLE)

## Enable GPU and CPU allocated buffer reuse between different calls
add_definitions(-DBUFFER_REUSE_ENABLE)

## Enable lock-free versions of event & Queue management in Universal_helpers
add_definitions(-DUNIHELPER_LOCKFREE_ENABLE)

## Choose the output algorithm for obtaining the actual results of each routine. Options:
## - ALGO_WR: The output tiles are processed normally. This entails being fetched at the beggining, 
##            locked while being written on (TODO) and written back after all computations are complete.
## - ALGO_WR_LAZY: The output tiles are fetched from the source lazily. Computations will use
##             local buffers. After all computation on each tile is fired, its source values will be
##             be fetched, reduced locally in the last computation location and written back.
## - ALGO_WREDUCE: The output tiles are not fetched from the source. Instead, computations will use
##             local buffers. After they are complete, tiles are reduced to the output (=source) location.
#add_definitions(-DOUTPUT_ALGO_MODE="ALGO_WR")
add_definitions(-DOUTPUT_ALGO_MODE="ALGO_WR_LAZY")
#add_definitions(-DOUTPUT_ALGO_MODE="ALGO_WREDUCE")

## The maximum allowed parallel backend operations per queue
add_definitions(-DMAX_BACKEND_L=8)

## The workspace size that should be assigned to each cublas handle
## If set to -1, the default ws will be used instead 
## NOTE-FIXME!!! : Currently WS effects the PCIe BW when a cublas kernel is launched. 
## NOTE-FIXME!!! This has a serious overhead if SKs are launched during selection
## NOTE-FIXME!!! Use of SUBKERNELS_FIRE_WHEN_READY is advised for mitigating this
add_definitions(-DWS_SZ=0)

## Enable overlapping Send & recv Communication FIXME: Not implemented
add_definitions(-DENABLE_SEND_RECV_OVERLAP)

## Enable using the CPU as an extra device for part of the total workload, if applicable
add_definitions(-DENABLE_CPU_WORKLOAD)

## Define if Subkernel operations are fired together with data fetches 
## OR when their data become available
add_definitions(-DSUBKERNELS_FIRE_WHEN_READY)

## Define which subkernel distribution to use
#add_definitions(-DDISTRIBUTION="ROUND-ROBIN")
#add_definitions(-DDISTRIBUTION="SPLIT-NAIVE")
#add_definitions(-DDISTRIBUTION="SPLIT-CHUNKS-ROBIN")
#add_definitions(-DDISTRIBUTION="SPLIT-CHUNKS-ROBIN-REVERSE")
add_definitions(-DDISTRIBUTION="2D-BLOCK-CYCLIC")

## Enable using transfer hops for faster transfers between multiple links
#add_definitions(-DENABLE_TRANSFER_HOPS)
## Enable using transfer hops for W tiles as well (no data reuse, but potentially better BW)
#add_definitions(-DENABLE_TRANSFER_W_HOPS)

## The number of sub-buffers FasTCoCoMemcpy2DAsync will use to increase effective BW
## In general effective_trans_time = max_trans_time(route[]) + 1/SBO * sub_trans_time(route_minus_max[])
## Rule of thumb: large SBO -> increased transfer latency, small SBO -> decreased bandwidth
add_definitions(-DSTREAMING_BUFFER_OVERLAP=8)
## TODO: Route chains could be implemented later (e.g. instead of 1->2->3 to use 1->2 parallel to 1->3)

## Enable using estimated performance aggregation for initializing the link_hop_route
#add_definitions(-DENABLE_ESPA)
## Define the strategy for agreggating the transfer time of a problem if ESPA is used. 0 = max, 1 = sum.
#add_definitions(-DESPA_COMMUNICATION_AGGREGATOR=0)

## Enable sharing of bandwidth between links that utilize the same underlying hardware communication resources
#add_definitions(-DENABLE_LINK_BW_SHARING) # by putting them on different queues

## TODO: Enable power measuring for benchmarks, and energy-related prediction modes for the Autotuner
add_definitions(-DENABLE_POWA)
add_definitions(-DCPU_W_PREDEF=$ENV{PARALIA_W_CPU_PREDEF})

## Choose the Unit combination that maximizes total performance (minimum time).
add_definitions(-DPREDICT_OPTIMIZE_TARGET="PERF")
## Choose the Unit combination that minimizes consumed energy (J).
## Rarely chooses multi-unit execution, since its almost always less energy-efficient than single-unit.
#add_definitions(-DPREDICT_OPTIMIZE_TARGET="ENERGY")

## Choose the Unit combination that maximizes the power-delay product (similar results with energy).
#add_definitions(-DPREDICT_OPTIMIZE_TARGET="POWER-DELAY")

## Choose the Unit combination that maximizes the energy-delay product - closer to "PERF" but also slightly considering energy.
#add_definitions(-DPREDICT_OPTIMIZE_TARGET="ENERGY-DELAY")

## Choose the Unit combination based on a configurable minimum acceptable percentile performance-improvement-per-J-increase.
## Example for choosing U1(tpred = X, En = J1) vs U2(tpred = Y, En = J2) units with PERPER_LIMIT: if ( X/Y >= PERPER_LIMIT*J2/J1) U2 else U1
#add_definitions(-DPREDICT_OPTIMIZE_TARGET="PERF-PER-J")
## PERPER_LIMIT higher -> "ENERGY", lower -> "PERF"
add_definitions(-DPERPER_LIMIT=0.0)

#add_definitions(-DPREDICT_OPTIMIZE_TARGET="OTHER_TBD")

## Define at which percentage to normalize float values for similar data (e.g. costs, splits etc) to avoid empirical errors leaking into decisions
add_definitions(-DNORMALIZE_NEAR_SPLIT_LIMIT=0.05)

## Define the minimum allowed percentile contribution in total time from each potential unit
## In case of energy, the maximum allowed percentile energy increase by each new unit.
add_definitions(-DMINIMUM_UNIT_CONTRIBUTION=0.05)

## Allow fetching from "Recorded" (i.e. transfer scheduled, not complete) blocks locations if closer to the destination unit.
add_definitions(-DALLOW_FETCH_RECORDED)

#Run subkernels in provided order
add_definitions(-DSERIAL_SUBKERNEL_SELECTION)

## Define subkernel selection heuristic values. 0 = No penalty
## 
add_definitions(-DSUBKERNEL_SELECT_MAX_FETCHED)
#add_definitions(-DSUBKERNEL_SELECT_MIN_RONLY_ETA)

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

#Run Simple Prediction debug mode (well-presented simple autotuner stuff)
#add_definitions(-DSDEBUG)

#Run Prediction debug mode (autotuner stuff)
#add_definitions(-DPDEBUG)

#Run Prediction deep debug mode (all autotuner stuff)
#add_definitions(-DDPDEBUG)

#Run in caching debug mode, for cache-related debugging
#add_definitions(-DCDEBUG)

#Run in extended_linkmap debug mode
#add_definitions(-DUDEBUG)

#Run in extended_linkmap deep debug mode, for linkmap-related debugging
#add_definitions(-DUDDEBUG)

## Define a name for the testbed, to be used for the whole framework setup
set(TESTBED_NAME $ENV{PARALIA_SYSTEM})
message("PARALiA-Framework - TESTBED_NAME: " ${TESTBED_NAME})
add_definitions(-DTESTBED="${TESTBED_NAME}")

## Define the (max) number of devices for utilization (0-num_devices)
set(NUM_LOCATIONS $ENV{PARALIA_SYSTEM_DEVNUM})
message("PARALiA-Framework - NUM_LOCATIONS: " ${NUM_LOCATIONS})
add_definitions(-DLOC_NUM=${NUM_LOCATIONS})

## Define the max GPU memory percentage (%) a problem can use ( not accounting for streams and other control structs; e.g. only the problem data ).
set(PROBLEM_GPU_PERCENTAGE 70)
message("PARALiA-Framework - PROBLEM_GPU_PERCENTAGE: " ${PROBLEM_GPU_PERCENTAGE})
add_definitions(-DPROBLEM_GPU_PERCENTAGE=${PROBLEM_GPU_PERCENTAGE})

###TODO:Define specific compiler, flags, includes, libs for the target system (IF needed).

#-------------------------------------------------CUDA----------------------------------------------------------#

if(NOT $ENV{PARALIA_CUDA_TOOLKIT_PREFIX} STREQUAL "default")
  set (CUDA_PREFIX $ENV{PARALIA_CUDA_TOOLKIT_PREFIX})
  set (CMAKE_CUDA_COMPILER ${CUDA_PREFIX}/bin/nvcc)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES $ENV{PARALIA_CUDA_ARCH})
endif()

set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w -O3")
message("PARALiA-Framework - CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})

set (CUDA_INCLUDE_DIRS ${CUDA_PREFIX}/include ${CUDA_PREFIX}/samples/common/inc)
include_directories(${CUDA_INCLUDE_DIRS})
message("PARALiA-Framework - CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})

set (CUDA_LD ${CUDA_PREFIX}/lib64) # or /lib depending on system
set (CUDA_LINK "-L${CUDA_LD} -lcudart -lcublas -lcurand $ENV{PARALIA_CUDA_LOAD_COMMAND}")
message("PARALiA-Framework - CUDA_LINK: " ${CUDA_LINK} )
set (INTERNAL_LINK_LIB "${CUDA_LINK} -lrt -ldl")
message("PARALiA-Framework - INTERNAL_LINK_LIB: " ${INTERNAL_LINK_LIB})

#-------------------------------------------------C/C++---------------------------------------------------------#

if(NOT $ENV{PARALIA_CXX_PREFIX} STREQUAL "default")
  set (CXX_PREFIX $ENV{PARALIA_CXX_PREFIX})
  set (CMAKE_CXX_COMPILER ${CXX_PREFIX}/bin/c++)
  set (CMAKE_C_COMPILER ${CXX_PREFIX}/bin/gcc)
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -lm -Wuninitialized")
message("PARALiA-Framework - CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

#set(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS})

#-----------------------------------------------OpenBLAS--------------------------------------------------------#

set(OPENBLAS_PREFIX $ENV{PARALIA_OPENBLAS_PREFIX})

set (OPENBLAS_INCLUDE_DIRS ${OPENBLAS_PREFIX}/include)
include_directories(${OPENBLAS_INCLUDE_DIRS})
message("PARALiA-Framework - OPENBLAS_INCLUDE_DIRS: " ${OPENBLAS_INCLUDE_DIRS})

set (OPENBLAS_LD ${OPENBLAS_PREFIX}/lib)
set (OPENBLAS_LINK  "-L${OPENBLAS_LD} -lopenblas" )
message("PARALiA-Framework - OPENBLAS_LINK: " ${OPENBLAS_LINK})

#---------------------------------------------Start Project-----------------------------------------------------#

project (PARALiA-Framework VERSION ${PROJECT_VERSION} DESCRIPTION "The complete PARALiA Framework" LANGUAGES CUDA CXX)

if($ENV{PARALIA_INSTALL_PREFIX} STREQUAL "default")
  message("PARALiA-Framework - PARALIA_INSTALL_PREFIX not provided, project will be installed under build dir.")
  set (CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${TESTBED_NAME}-install)
else()
  set (CMAKE_INSTALL_PREFIX $ENV{PARALIA_INSTALL_PREFIX})
endif()
set (CMAKE_INSTALL_PROJECT ${CMAKE_INSTALL_PREFIX})
message("PARALiA-Framework - CMAKE_INSTALL_PREFIX, CMAKE_INSTALL_PROJECT: " ${CMAKE_INSTALL_PROJECT})

configure_file(${PROJECT_SOURCE_DIR}/Deploy.in ${CMAKE_INSTALL_PROJECT}/Deploy.sh @ONLY)

set(PROJECT_BENCH_DIR ${PROJECT_SOURCE_DIR}/Benchmarking)
configure_file(${PROJECT_BENCH_DIR}/Test_correctness.in ${CMAKE_INSTALL_PROJECT}/Test_correctness.sh @ONLY)
configure_file(${PROJECT_BENCH_DIR}/Targeted_performance_evaluation.in ${CMAKE_INSTALL_PROJECT}/Targeted_performance_evaluation.sh @ONLY)
configure_file(${PROJECT_BENCH_DIR}/Full_model_validation.in ${CMAKE_INSTALL_PROJECT}/Full_model_validation.sh @ONLY)
configure_file(${PROJECT_BENCH_DIR}/Full_performance_evaluation.in ${CMAKE_INSTALL_PROJECT}/Full_performance_evaluation.sh @ONLY)
configure_file(${PROJECT_BENCH_DIR}/Heterogeneous_performance_estimation.in ${CMAKE_INSTALL_PROJECT}/Heterogeneous_performance_estimation.sh @ONLY)

#set(VALUETYPEID 1)
#if (${VALUETYPEID} EQUAL 0)
#  add_definitions(-DVALUE_TYPE=float)
#elseif(${VALUETYPEID} EQUAL 1)
#  add_definitions(-DVALUE_TYPE=double)
#else()
#endif()

#Used for defining Universal_helpers subdir setup mode. Always one for PARALiA
set (BUILD_BLAS_WRAPPERS_FLAG 1)
add_definitions(-DBUILD_BLAS_WRAPPERS_FLAG)

ADD_SUBDIRECTORY (Extended_LinkMap)

ADD_SUBDIRECTORY (Database_builder)
add_definitions(-DDEPLOYDB="${DEPLOYMENT_DIR_DB}")

ADD_SUBDIRECTORY (Autotuner)

ADD_SUBDIRECTORY (Library_scheduler)

ADD_SUBDIRECTORY (Benchmarking)
