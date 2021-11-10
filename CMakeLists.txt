cmake_minimum_required (VERSION 3.10)

## Define a name for the backend library wrappers/implementations used in this build, Currently implemented only with CuCuBLAS(==CUDA+CUBLAS)
set(BACKEND "CuCuBLAS")
message("CoCopeLia-Framework - BACKEND: " ${BACKEND})
add_definitions(-DBACKEND="${BACKEND}")
## Define the project version (used in logfiles for result testing)
set(PROJECT_VERSION "0.21")
message("CoCopeLia-Framework - PROJECT_VERSION: " ${PROJECT_VERSION})
add_definitions(-DVERSION="${PROJECT_VERSION}")
#Run in testing mode : details about models, decisions and time spend in operations.
#add_definitions(-DTEST)
#Run in debug mode, which includes detailed function calls and info for error-checking
add_definitions(-DDEBUG)

project (CoCopeLia-Framework VERSION ${PROJECT_VERSION} DESCRIPTION "The complete CoCoPeLia Framework")

## Define a name for the testbed, to be used for the whole framework setup
#set(TESTBED_NAME "dungani")
set(TESTBED_NAME "silver1")
message("CoCopeLia-Framework - TESTBED_NAME: " ${TESTBED_NAME})
add_definitions(-DTESTBED="${TESTBED_NAME}")

## Define the (max) number of devices for utilization (0-num_devices)
set(NUM_DEVICES 2)
message("CoCopeLia-Framework - NUM_DEVICES: " ${NUM_DEVICES})
add_definitions(-DDEV_NUM=${NUM_DEVICES})

## Define the number of silmuntaneous allowed CUDA streams for GPU execution.
set(NUM_EXEC_STREAMS 4)
message("CoCopeLia-Framework - NUM_EXEC_STREAMS: " ${NUM_EXEC_STREAMS})
add_definitions(-DNUM_EXEC_STREAMS=${NUM_EXEC_STREAMS})

## Define the max GPU memory percentage (%) a problem can use ( not accounting for streams and other control structs; e.g. only the problem data ).
set(PROBLEM_GPU_PERCENTAGE 90)
message("CoCopeLia-Framework - PROBLEM_GPU_PERCENTAGE: " ${PROBLEM_GPU_PERCENTAGE})
add_definitions(-DPROBLEM_GPU_PERCENTAGE=${PROBLEM_GPU_PERCENTAGE})

###TODO:Define specific compiler, flags, includes, libs for the target system (IF needed).

## 1) C++ 

#	a) auto

#	b) compile flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -lm -Wuninitialized") # -std=gnu89 TODO:needed? 
message("CoCopeLia-Framework - CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

#	c) auto

#	d) auto

## 2) CUCUBLAS TODO: Other backends?
set (CUDA_PREFIX "/usr/local/cuda-11.0")
add_definitions(-DCUDA_VER=1100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70) #35 for Tesla K40, 70 for V100, 61 for GTX 1060/70
endif()

# 	a) Compiler
set (CMAKE_CUDA_COMPILER ${CUDA_PREFIX}/bin/nvcc)

# 	b) Compile flags
set (CMAKE_CUDA_FLAGS "-w -O3")
message("CoCopeLia-Framework - CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})

# 	c) Includes
set (CUDA_INCLUDE_DIRS ${CUDA_PREFIX}/include ${CUDA_PREFIX}/samples/common/inc)
include_directories(${CUDA_INCLUDE_DIRS})
message("CoCopeLia-Framework - CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
set(OPENBLAS_PREFIX "/home/users/panastas/Lib_install/OpenBLAS")
set (OPENBLAS_INCLUDE_DIRS ${OPENBLAS_PREFIX}/include)
include_directories(${OPENBLAS_INCLUDE_DIRS})
message("CoCopeLia-Framework - OPENBLAS_INCLUDE_DIRS: " ${OPENBLAS_INCLUDE_DIRS})

# 	d) Link Libraries
set (CUDA_LD ${CUDA_PREFIX}/lib64) # or /lib depending on system
set (CUDA_LINK "-L${CUDA_LD} -lcudart -lcublas -lcurand") #  
message("CoCopeLia-Framework - CUDA_LINK: " ${CUDA_LINK} )
set (INTERNAL_LINK_LIB "${CUDA_LINK} -lrt -ldl")
message("CoCopeLia-Framework - INTERNAL_LINK_LIB: " ${INTERNAL_LINK_LIB})
set (OPENBLAS_LD ${OPENBLAS_PREFIX}/lib)
set (OPENBLAS_LINK  "-L${OPENBLAS_LD} -lopenblas" )
message("CoCopeLia-Framework - OPENBLAS_LINK: " ${OPENBLAS_LINK} )

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

ADD_SUBDIRECTORY (Universal_helpers)

ADD_SUBDIRECTORY (Deployment_phase)
add_definitions(-DDEPLOYDB="${DEPLOYMENT_DIR_DB}")

ADD_SUBDIRECTORY (Autotuning_runtime)

set(EXPERIMENTAL_VERSION 1) 
if (${EXPERIMENTAL_VERSION} EQUAL 0)
	ADD_SUBDIRECTORY (Library_scheduler)
elseif(${EXPERIMENTAL_VERSION} EQUAL 1)
    ADD_SUBDIRECTORY (Library_scheduler_experimental)
else()
endif()

ADD_SUBDIRECTORY (Benchmarking)
