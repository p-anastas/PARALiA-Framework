###########################################################################################
#---------------------------------Project-wide Options------------------------------------#
###########################################################################################

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

###########################################################################################
#---------------------------Internal resource reuse options-------------------------------#
###########################################################################################

## Enable Asynchronous execution for Subkernels
add_definitions(-DASYNC_ENABLE)

## Enable lock-free versions of event & Queue management in Universal_helpers
add_definitions(-DUNIHELPER_LOCKFREE_ENABLE)

## Enable GPU backend resourse reuse (streams, handles)
add_definitions(-DQUEUE_REUSE_ENABLE)

## Enable GPU and CPU allocated buffer reuse between different calls
add_definitions(-DBUFFER_REUSE_ENABLE)

## The level of reuse for the autotuning metadata of every routine call by SIMILAR subsequent calls
## NONE -> Rerun full autotuning every time
## MODELS -> Keep the models but rerun the distribution and routing functions
## PROBLEMS -> Keep a cache of size PROBLEM_MD_CACHE with the autotuning MD of DIFFERENT routine calls
##             This enables the full autotuning metadata reuse of up to PROBLEM_MD_CACHE problems. 
#add_definitions(-DMETADATA_REUSE_NONE)
#add_definitions(-DMETADATA_REUSE_MODELS)
add_definitions(-DMETADATA_REUSE_PROBLEMS) # FIXME: Currently bugged
add_definitions(-DPROBLEM_MD_CACHE=100) # Only relevant for "PROBLEMS" reuse level

###########################################################################################
#---------------------------Algorithmic/Software Optimizations----------------------------#
###########################################################################################

## The number of parallel workers that can be utilized at each device for comp-comp overlap. 
add_definitions(-DMAX_BACKEND_L=8)

## Enable using the CPU as an extra device for part of the total workload, if applicable
add_definitions(-DENABLE_CPU_WORKLOAD)

## Define if Subkernel operations are fired together with data fetches 
## OR when their data become available
add_definitions(-DSUBKERNELS_FIRE_WHEN_READY)

## The output algorithm for obtaining the actual results of each routine. Options:
## - ALGO_WR: The output tiles are processed normally. This entails being fetched at the beggining, 
##            locked while being written on (TODO) and written back after all computations are complete.
## - ALGO_WR_LAZY: The output tiles are fetched from the source lazily. Computations will use
##             local buffers. After all computation on each tile is fired, its source values will be
##             be fetched, reduced locally in the last computation location and written back.
## - ALGO_WREDUCE: The output tiles are not fetched from the source. Instead, computations will use
##             local buffers. After they are complete, tiles are reduced to the output (=source) location.
add_definitions(-DOUTPUT_ALGO_MODE="ALGO_WR")
#add_definitions(-DOUTPUT_ALGO_MODE="ALGO_WR_LAZY")
#add_definitions(-DOUTPUT_ALGO_MODE="ALGO_WREDUCE")

###########################################################################################
#---------------------------------Routing optimizations-----------------------------------#
###########################################################################################

## Enable overlapping Send & recv Communication FIXME: Not implemented
add_definitions(-DENABLE_SEND_RECV_OVERLAP)

## Enable sharing of bandwidth between links that utilize the same underlying hardware communication resources
#add_definitions(-DENABLE_LINK_BW_SHARING) # by putting them on different queues

## The number of sub-buffers FasTCoCoMemcpy2DAsync will use to increase effective BW
## In general effective_trans_time = max_trans_time(route[]) + 1/SBO * sub_trans_time(route_minus_max[])
## Rule of thumb: large SBO -> increased transfer latency, small SBO -> decreased bandwidth
add_definitions(-DSTREAMING_BUFFER_OVERLAP=8)

###########################################################################################
#-----------------------------------Routing selection-------------------------------------#
###########################################################################################

#add_definitions(-DP2P_FETCH_FROM_INIT)
#add_definitions(-DP2P_FETCH_FROM_GPU_SERIAL)
#add_definitions(-DP2P_FETCH_FROM_GPU_DISTANCE)
#add_definitions(-DCHAIN_FETCH_SERIAL)
#add_definitions(-DCHAIN_FETCH_RANDOM)
#add_definitions(-DCHAIN_FETCH_TIME)
add_definitions(-DCHAIN_FETCH_QUEUE_WORKLOAD)

## An extra heuristic to avoid only trusting QUEUE_ETAs for route selection
## Since some routes migh result in extremely low botlenecks due to serialization
## The multiplier defines the max bandwidth multiplier that is allowed to be used to replace a route
## For example, 1 means that only allow replacement with links of >= bandwidth
## The higher this value is, the more ETA-based the selection criteria
add_definitions(-DBANDWIDTH_DIFFERENCE_CUTTOF_RATIO=5)

## Enable the use of transfer hops through memory locations that do not need a tile
## Improves bandwidth, but adds (extra) network load to fast lanes. Currently not integrated in already-chained transfers.
add_definitions(-DENABLE_TRANSFER_HOPS)

#add_definitions(-DHOP_FETCH_BANDWIDTH)
#add_definitions(-DHOP_FETCH_QUEUE_WORKLOAD)
add_definitions(-DHOP_FETCH_BW_PLUS_ETA)

###########################################################################################
#----------------------------------Autotuning options-------------------------------------#
###########################################################################################

## The number of itterations that will performed to tune the per-device ratio workload split.
## Inconsequential for homogeneous cases, higher = better for heterogeneous ones.
add_definitions(-DRATIO_TUNE_ITTER=10)

#Enable device reordering optimization during split optimizing based on maximizing d2d bandwidths.
# Seems buggy for now...
#add_definitions(-DREORDER_DEVICES)

###########################################################################################
#---------------------------------Tiling size selection-----------------------------------#
###########################################################################################

## Decision regarding perplexing tile selection and workload selection
## by applying estimated overlap sl from tile to device split prediction
add_definitions(-DAPPLY_TILE_SL_TO_WORKLOAD_SPLIT)
## TODO: For future work could autotune conditions.
## Conditions (with decreasing priority): 
## 1: NO-Imbalance: Spliting creates subkernels that can be distributed equally to devices without WR-sharing
## 2: NO-remainder: split should not create remainder Tiles if possible
## 3: T-min: Tile should be >= than a preset value
add_definitions(-DTILE_MIN=2048)
add_definitions(-DTILE_MIN_SLOWDOWN=0.2)
## 4: SK-num: subkernels per device must be >= MIN_DESIRED_SK_DEV
add_definitions(-DMIN_DESIRED_SK_DEV=64)
add_definitions(-DMAX_DESIRED_SK_DEV=512)
add_definitions(-DMAX_DESIRED_SK_DEV_SLOWDOWN=0.1)
## 5: T-max: Tile should be <= than a preset value
add_definitions(-DTILE_MAX=4096)
add_definitions(-DTILE_MΑΧ_SLOWDOWN=0.1)


###########################################################################################
#------------------------------Subkernel selection options--------------------------------#
###########################################################################################

#Run subkernels in provided order
#add_definitions(-DSERIAL_SUBKERNEL_SELECTION)

## Define subkernel selection heuristic values.
#add_definitions(-DSUBKERNEL_SELECT_MIN_RONLY_ETA)
add_definitions(-DSUBKERNEL_SELECT_FETCH_ETA_PLUS_MIN_PENDING)

###########################################################################################
#--------------------------------Subkernel distributions----------------------------------#
###########################################################################################

add_definitions(-DDISTRIBUTION="2D-BLOCK-CYCLIC")

#--------------------------------------Experimental----------------------------------------#

#add_definitions(-DDISTRIBUTION="ROUND-ROBIN")
#add_definitions(-DDISTRIBUTION="SPLIT-NAIVE")
#add_definitions(-DDISTRIBUTION="SPLIT-CHUNKS-ROBIN")
#add_definitions(-DDISTRIBUTION="SPLIT-CHUNKS-ROBIN-REVERSE")

###########################################################################################
#---------------------------------Optimization targets------------------------------------#
###########################################################################################

## Define at which percentage to normalize float values for similar data (e.g. costs, splits etc) to avoid empirical errors leaking into decisions
add_definitions(-DNORMALIZE_NEAR_SPLIT_LIMIT=0.05)
#TODO: FIXME !!!! TRIM execution logs automatically, currently by hand for testing

## Define the minimum allowed percentile contribution in total time from each potential unit
## In case of energy, the maximum allowed percentile energy increase by each new unit.
add_definitions(-DMINIMUM_UNIT_CONTRIBUTION=0.05)

## Enable power measuring for benchmarks, and energy-related prediction modes for the Autotuner
add_definitions(-DENABLE_POWA)
add_definitions(-DCPU_W_PREDEF=$ENV{PARALIA_W_CPU_PREDEF})

## Choose the Unit combination that maximizes total performance (minimum time).
#add_definitions(-DPREDICT_OPTIMIZE_TARGET="PERF")
## Choose the Unit combination that minimizes consumed energy (J).
## Rarely chooses multi-unit execution, since its almost always less energy-efficient than single-unit.
#add_definitions(-DPREDICT_OPTIMIZE_TARGET="ENERGY")

## Choose the Unit combination that maximizes the power-delay product (similar results with energy).
#add_definitions(-DPREDICT_OPTIMIZE_TARGET="POWER-DELAY")

## Choose the Unit combination that maximizes the energy-delay product - closer to "PERF" but also slightly considering energy.
add_definitions(-DPREDICT_OPTIMIZE_TARGET="ENERGY-DELAY")

#--------------------------------------Experimental----------------------------------------#

## Choose the Unit combination based on a configurable minimum acceptable percentile performance-improvement-per-J-increase.
## Example for choosing U1(tpred = X, En = J1) vs U2(tpred = Y, En = J2) units with PERPER_LIMIT: if ( X/Y >= PERPER_LIMIT*J2/J1) U2 else U1
#add_definitions(-DPREDICT_OPTIMIZE_TARGET="PERF-PER-J")
## PERPER_LIMIT higher -> "ENERGY", lower -> "PERF"
add_definitions(-DPERPER_LIMIT=0.0)

#add_definitions(-DPREDICT_OPTIMIZE_TARGET="OTHER_TBD")

###########################################################################################
#----------------------------------DEBUG/TEST Options-------------------------------------#
###########################################################################################

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

#Run in testing mode : details about models, decisions and time spend in operations.
#add_definitions(-DTEST)

#Run in Subkernel testing mode : details about time spend in subkernel operations.
#add_definitions(-DSTEST)

#Run in Deep Subkernel testing mode : many details about time spend in subkernel operations.
#add_definitions(-DDSTEST)

#Run in Link Hop testing mode; transfers using link hops are logged and timed.
#add_definitions(-DTTEST)

###########################################################################################
#------------------------------------Other options----------------------------------------#
###########################################################################################

## The workspace size that should be assigned to each cublas handle
## If set to -1, the default ws will be used instead 
## NOTE-FIXME!!! : Currently WS effects the PCIe BW when a cublas kernel is launched. 
## NOTE-FIXME!!! This has a serious overhead if SKs are launched during selection
## NOTE-FIXME!!! Use of SUBKERNELS_FIRE_WHEN_READY is advised for mitigating this
add_definitions(-DWS_SZ=0)

###########################################################################################
#-----------------------------------CUDA definitions--------------------------------------#
###########################################################################################


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

###########################################################################################
#----------------------------------C/C++ definitions--------------------------------------#
###########################################################################################

if(NOT $ENV{PARALIA_CXX_PREFIX} STREQUAL "default")
  set (CXX_PREFIX $ENV{PARALIA_CXX_PREFIX})
  set (CMAKE_CXX_COMPILER ${CXX_PREFIX}/bin/c++)
  set (CMAKE_C_COMPILER ${CXX_PREFIX}/bin/gcc)
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -lm -Wuninitialized")
message("PARALiA-Framework - CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

#set(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS})

###########################################################################################
#---------------------------------OpenBLAS definitions------------------------------------#
###########################################################################################

set(OPENBLAS_PREFIX $ENV{PARALIA_OPENBLAS_PREFIX})

set (OPENBLAS_INCLUDE_DIRS ${OPENBLAS_PREFIX}/include)
include_directories(${OPENBLAS_INCLUDE_DIRS})
message("PARALiA-Framework - OPENBLAS_INCLUDE_DIRS: " ${OPENBLAS_INCLUDE_DIRS})

set (OPENBLAS_LD ${OPENBLAS_PREFIX}/lib)
set (OPENBLAS_LINK  "-L${OPENBLAS_LD} -lopenblas" )
message("PARALiA-Framework - OPENBLAS_LINK: " ${OPENBLAS_LINK})

###########################################################################################
#-------------------------------------Start Project---------------------------------------#
###########################################################################################

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
