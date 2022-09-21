cmake_minimum_required (VERSION 3.10)
### Initialize the project
project (CoCopeLia-Benchmarks VERSION ${PROJECT_VERSION} DESCRIPTION "Library wrappers for performance comparisson with CoCoPeLia" LANGUAGES CXX) #CUDA
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
#set (BLASX_PREFIX "/zhome/academic/HLRS/xex/xexpanas/BLASX-pinned") # TODO: vulcan
set (BLASX_PREFIX "/home/users/panastas/PhD_stuff/Other-libs/BLASX-pinned") # TODO: silver1
set (BLASX_INCLUDE_DIRS ${BLASX_PREFIX}/include)
include_directories(${BLASX_INCLUDE_DIRS} )
message( "Benchmarking - Included BLASx: " ${BLASX_INCLUDE_DIRS} )
set (BLASX_LD ${BLASX_PREFIX}/lib)
set (BLASX_LINK  "-L${BLASX_LD} -lblasx" )
message( "Benchmarking - Linking BLASx: " ${BLASX_LINK} )

#XKBLAS libs and includes
#set (XKBLAS_PREFIX "/zhome/academic/HLRS/xex/xexpanas/xkblas-basic/install_vulcan") # TODO: vulcan cuda 10.1.243
#set (XKBLAS_PREFIX "/zhome/academic/HLRS/xex/xexpanas/xkblas-basic/install_vulcan_11") # TODO: vulcan
set (XKBLAS_PREFIX "/home/users/panastas/PhD_stuff/Other-libs/xkblas-silver1/install") # TODO: silver1
set (XKBLAS_INCLUDE_DIRS ${XKBLAS_PREFIX}/include)
include_directories(${XKBLAS_INCLUDE_DIRS} )
message( "Benchmarking - Included XKBLAS: " ${XKBLAS_INCLUDE_DIRS} )
set (XKBLAS_LD ${XKBLAS_PREFIX}/lib)
set (XKBLAS_LINK  "-L${XKBLAS_LD}  -lm -lpthread -lkaapi -lxkblas -lcuda" )  # TODO: silver1
#set (XKBLAS_LINK  "-L${XKBLAS_LD}  -lm -lpthread -lkaapi -lxkblas /opt/system/nvidia/ALL.ALL.510.73.08/libcuda.so.510.73.08") # TODO: Vulcan   ?
message( "Benchmarking - Linking XKBLAS: " ${XKBLAS_LINK} )

add_library(benchcocopelia SHARED
${BENCH_DIR_SRC}/cuBLASWrappers.cpp
${BENCH_DIR_SRC}/cuBLASXtWrappers.cpp
${BENCH_DIR_SRC}/testing.cpp
)
target_link_libraries(benchcocopelia cocopelia ${INTERNAL_LINK_LIB} ${OPENBLAS_LINK} cocopelia autotuning_runtime unihelpers )
configure_file(${BENCH_DIR_INC}/BackenedLibsWrapped.hpp ${BENCH_INSTALL_INC}/BackenedLibsWrapped.hpp COPYONLY)
configure_file(${BENCH_DIR_INC}/BLASxWrapped.hpp ${BENCH_INSTALL_INC}/BLASxWrapped.hpp COPYONLY)

#Enable results validation in "Runner" scripts (adds extra overhead, "Tester" scripts should be used for result validation)
#add_definitions(-DRUNVALIDATION)
#Enable checking the existing logs for each benchmark before executing it, to avoid running the same exact test multiple times.
#add_definitions(-DCHECKLOG)

#add_executable (cuBLASXtDgemmRunner ${BENCH_DIR_SRC}/cuBLASXtDgemmRunner.cpp )
#add_executable (BLASxDgemmRunner ${BENCH_DIR_SRC}/BLASxDgemmRunner.cpp ${BENCH_DIR_SRC}/BLASXWrappers.cpp)
#add_executable (BLASxExDgemmRunner ${BENCH_DIR_SRC}/BLASxExDgemmRunner.cpp ${BENCH_DIR_SRC}/BLASXWrappers.cpp)
#add_executable (XKBLASDgemmRunner ${BENCH_DIR_SRC}/XKBLASDgemmRunner.cpp)

#add_executable (CoCoPeLiaTestLinkOverlap ${BENCH_DIR_SRC}/CoCoPeLiaTestLinkOverlap.cpp)
#add_executable (CoCoPeLiaTestFasTransfer ${BENCH_DIR_SRC}/CoCoPeLiaTestFasTransfer.cpp)

#target_link_libraries(cuBLASXtDgemmRunner ${INTERNAL_LINK_LIB} benchcocopelia unihelpers)
#target_link_libraries(BLASxDgemmRunner ${INTERNAL_LINK_LIB} benchcocopelia unihelpers ${BLASX_LINK})
#target_link_libraries(BLASxExDgemmRunner ${INTERNAL_LINK_LIB} benchcocopelia unihelpers ${BLASX_LINK})
#target_link_libraries(XKBLASDgemmRunner ${INTERNAL_LINK_LIB} benchcocopelia unihelpers ${XKBLAS_LINK} )

#target_link_libraries(CoCoPeLiaTestLinkOverlap ${INTERNAL_LINK_LIB} benchcocopelia unihelpers )
#target_link_libraries(CoCoPeLiaTestFasTransfer ${INTERNAL_LINK_LIB} benchcocopelia unihelpers )

add_executable (CoCoPeLiaDgemmTester ${BENCH_DIR_SRC}/CoCoPeLiaDgemmTester.cpp )
add_executable (CoCoPeLiaDgemmRunner ${BENCH_DIR_SRC}/CoCoPeLiaDgemmRunner.cpp )
#add_executable (CoCoPeLiaDgemmRunnerBest ${BENCH_DIR_SRC}/CoCoPeLiaDgemmRunnerBest.cpp )
#add_executable (CoCoPeLiaDgemmRunnerBestest ${BENCH_DIR_SRC}/CoCoPeLiaDgemmRunnerBestest.cpp )

#add_executable (CoCoPeLiaDaxpyRunner ${BENCH_DIR_SRC}/CoCoPeLiaDaxpyRunner.cpp )
#add_executable (CoCoPeLiaDaxpyRunnerBest ${BENCH_DIR_SRC}/CoCoPeLiaDaxpyRunnerBest.cpp )

target_link_libraries(CoCoPeLiaDgemmTester ${INTERNAL_LINK_LIB} benchcocopelia)
target_link_libraries(CoCoPeLiaDgemmRunner ${INTERNAL_LINK_LIB} benchcocopelia unihelpers)
#target_link_libraries(CoCoPeLiaDgemmRunnerBest ${INTERNAL_LINK_LIB} benchcocopelia unihelpers)
#target_link_libraries(CoCoPeLiaDgemmRunnerBestest ${INTERNAL_LINK_LIB} benchcocopelia unihelpers)
#target_link_libraries(CoCoPeLiaDaxpyRunner ${INTERNAL_LINK_LIB} benchcocopelia unihelpers)
#target_link_libraries(CoCoPeLiaDaxpyRunnerBest ${INTERNAL_LINK_LIB} benchcocopelia unihelpers)

set (BENCH_INSTALL_INC ${BENCH_INSTALL_INC} PARENT_SCOPE)
#message( "->\nBenchmarking - Shared Library: " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
#message( "Benchmarking - Shared Library headers: " ${BENCH_INSTALL_INC} )
message( "Benchmarking - Library testing bin DIR: " ${BENCH_INSTALL_BIN} )
message( "------------------------------------------------------------------------------------------------")
