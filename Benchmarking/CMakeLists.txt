cmake_minimum_required (VERSION 3.10)
### Initialize the project
project (PARALiA-Benchmarks VERSION ${PROJECT_VERSION} DESCRIPTION "Library wrappers for performance comparisson with PARALiA" LANGUAGES CXX CUDA)
message( "------------------------------------------------------------------------------------------------")
message( "Benchmarking - PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR} )
#message( "Library Scheduler - CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR} )
#message( "Library Scheduler - CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR} )

set ( BENCH_DIR ${PROJECT_SOURCE_DIR}/${BACKEND} )
message( "Benchmarking - LIBSC_DIR: " ${BENCH_DIR} )

set (BENCH_DIR_INC ${BENCH_DIR}/include)
include_directories( ${BENCH_DIR_INC})
message( "Benchmarking - Included DIR: " ${BENCH_DIR_INC} )

include_directories(${UNIHELP_INSTALL_INC})
message( "Benchmarking - Included DIR: " ${UNIHELP_INSTALL_INC} )

include_directories(${AUTOTUNE_INSTALL_INC})
message( "Benchmarking - Included DIR: " ${AUTOTUNE_INSTALL_INC} )

include_directories(${LIBSC_INSTALL_INC})
message( "Benchmarking - Included DIR: " ${LIBSC_INSTALL_INC} )

set (BENCH_DIR_SRC ${BENCH_DIR}/src)
message( "Benchmarking - Source DIR: " ${BENCH_DIR_SRC} )

set (BENCH_INSTALL ${CMAKE_INSTALL_PROJECT}/Benchmarking)
file(MAKE_DIRECTORY ${BENCH_INSTALL})

set (BENCH_INSTALL_INC ${BENCH_INSTALL}/include)
file(MAKE_DIRECTORY ${BENCH_INSTALL_INC})

#set (BENCH_INSTALL_LIB ${BENCH_INSTALL}/lib)
#file(MAKE_DIRECTORY ${BENCH_INSTALL_LIB})
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBSC_INSTALL_LIB})

set (BENCH_INSTALL_BIN ${BENCH_INSTALL}/testing-bin)
file(MAKE_DIRECTORY ${BENCH_INSTALL_BIN})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BENCH_INSTALL_BIN})

file(MAKE_DIRECTORY ${BENCH_INSTALL}/testLogs)
add_definitions(-DTESTLIBDIR="${BENCH_INSTALL}/testLogs")

#BLASX libs and includes
if ($ENV{PARALIA_BLASX_BENCH})
    set (BLASX_PREFIX $ENV{PARALIA_BLASX_PREFIX} )
    set (BLASX_INCLUDE_DIRS ${BLASX_PREFIX}/include)
    include_directories(${BLASX_INCLUDE_DIRS} )
    message( "Benchmarking - Included BLASx: " ${BLASX_INCLUDE_DIRS} )
    set (BLASX_LD ${BLASX_PREFIX}/lib)
    set (BLASX_LINK  "-L${BLASX_LD} -lblasx" )
    message( "Benchmarking - Linking BLASx: " ${BLASX_LINK} )
endif()

#XKBLAS libs and includes
if ($ENV{PARALIA_XKBLAS_BENCH})
    set (XKBLAS_PREFIX $ENV{PARALIA_XKBLAS_PREFIX})
    set (XKBLAS_INCLUDE_DIRS ${XKBLAS_PREFIX}/include)
    include_directories(${XKBLAS_INCLUDE_DIRS} )
    message( "Benchmarking - Included XKBLAS: " ${XKBLAS_INCLUDE_DIRS} )
    set (XKBLAS_LD ${XKBLAS_PREFIX}/lib)
    set (XKBLAS_LINK  "-L${XKBLAS_LD}  -lm -lpthread -lkaapi -lxkblas $ENV{PARALIA_CUDA_LOAD_COMMAND}" )
    message( "Benchmarking - Linking XKBLAS: " ${XKBLAS_LINK} )
endif()

add_library(benchparalia SHARED
${BENCH_DIR_SRC}/cuBLASXt/cuBLASWrappers.cpp
${BENCH_DIR_SRC}/cuBLASXt/cuBLASXtWrappers.cpp
${BENCH_DIR_SRC}/testing.cpp
)
target_link_libraries(benchparalia paralia ${INTERNAL_LINK_LIB} ${OPENBLAS_LINK} paralia autotuner unihelpers )
configure_file(${BENCH_DIR_INC}/BackenedLibsWrapped.hpp ${BENCH_INSTALL_INC}/BackenedLibsWrapped.hpp COPYONLY)

if ($ENV{PARALIA_BLASX_BENCH})
    configure_file(${BENCH_DIR_INC}/BLASxWrapped.hpp ${BENCH_INSTALL_INC}/BLASxWrapped.hpp COPYONLY)
endif()

#Enable results validation in "Runner" scripts (adds extra overhead, "Tester" scripts should be used for result validation)
#add_definitions(-DRUNVALIDATION)
#Enable checking the existing logs for each benchmark before executing it, to avoid running the same exact test multiple times.
#add_definitions(-DCHECKLOG)

## Internal testers for PARALia components
add_executable (testLinkOverlap ${BENCH_DIR_SRC}/Component_testers/testLinkOverlap.cpp)
add_executable (testFasTransfer ${BENCH_DIR_SRC}/Component_testers/testFasTransfer.cpp)

target_link_libraries(testLinkOverlap ${INTERNAL_LINK_LIB} benchparalia unihelpers )
target_link_libraries(testFasTransfer ${INTERNAL_LINK_LIB} benchparalia unihelpers )

## Runners for SotA
add_executable (cuBLASXt_dgemm_runner ${BENCH_DIR_SRC}/cuBLASXt/cuBLASXt_dgemm_runner.cpp )
if ($ENV{PARALIA_BLASX_BENCH})
    add_executable (BLASx_dgemm_runner ${BENCH_DIR_SRC}/BLASX/BLASx_dgemm_runner.cpp ${BENCH_DIR_SRC}/BLASX/BLASXWrappers.cpp)
    add_executable (BLASxEx_dgemm_runner ${BENCH_DIR_SRC}/BLASX/BLASxEx_dgemm_runner.cpp ${BENCH_DIR_SRC}/BLASX/BLASXWrappers.cpp)
endif()
if ($ENV{PARALIA_XKBLAS_BENCH})
    add_executable (XKBLAS_dgemm_runner ${BENCH_DIR_SRC}/XKBLAS/XKBLAS_dgemm_runner.cpp)
endif()

target_link_libraries(cuBLASXt_dgemm_runner ${INTERNAL_LINK_LIB} benchparalia unihelpers)
if ($ENV{PARALIA_BLASX_BENCH})
    target_link_libraries(BLASx_dgemm_runner ${INTERNAL_LINK_LIB} benchparalia unihelpers ${BLASX_LINK})
    target_link_libraries(BLASxEx_dgemm_runner ${INTERNAL_LINK_LIB} benchparalia unihelpers ${BLASX_LINK})
endif()
if ($ENV{PARALIA_XKBLAS_BENCH})
    target_link_libraries(XKBLAS_dgemm_runner ${INTERNAL_LINK_LIB} benchparalia unihelpers ${XKBLAS_LINK} )
endif()

## PARALia Runners and Testers
add_executable (dgemm_tester ${BENCH_DIR_SRC}/PARALia/dgemm_tester.cpp )
add_executable (sgemm_tester ${BENCH_DIR_SRC}/PARALia/sgemm_tester.cpp )
add_executable (dgemm_runner ${BENCH_DIR_SRC}/PARALia/dgemm_runner.cpp )
add_executable (dgemm_runner_hetero ${BENCH_DIR_SRC}/PARALia/dgemm_runner_hetero.cu )
add_executable (dgemm_predictor ${BENCH_DIR_SRC}/PARALia/dgemm_predictor.cpp )

add_executable (dgemv_runner ${BENCH_DIR_SRC}/PARALia/dgemv_runner.cpp )

add_executable (daxpy_runner ${BENCH_DIR_SRC}/PARALia/daxpy_runner.cpp )
add_executable (ddot_runner ${BENCH_DIR_SRC}/PARALia/ddot_runner.cpp )

target_link_libraries(dgemm_tester ${INTERNAL_LINK_LIB} benchparalia)
target_link_libraries(sgemm_tester ${INTERNAL_LINK_LIB} benchparalia)
target_link_libraries(dgemm_runner ${INTERNAL_LINK_LIB} benchparalia unihelpers)
target_link_libraries(dgemm_runner_hetero ${INTERNAL_LINK_LIB} benchparalia unihelpers)
target_link_libraries(dgemm_predictor ${INTERNAL_LINK_LIB} benchparalia unihelpers)
target_link_libraries(dgemv_runner ${INTERNAL_LINK_LIB} benchparalia unihelpers)
target_link_libraries(daxpy_runner ${INTERNAL_LINK_LIB} benchparalia unihelpers)
target_link_libraries(ddot_runner ${INTERNAL_LINK_LIB} benchparalia unihelpers)

set (BENCH_INSTALL_INC ${BENCH_INSTALL_INC} PARENT_SCOPE)
#message( "->\nBenchmarking - Shared Library: " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
#message( "Benchmarking - Shared Library headers: " ${BENCH_INSTALL_INC} )
message( "Benchmarking - Library testing bin DIR: " ${BENCH_INSTALL_BIN} )
message( "------------------------------------------------------------------------------------------------")
