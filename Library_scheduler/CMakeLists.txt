cmake_minimum_required (VERSION 3.10)
### Initialize the project 
project (CoCopeLia-Library VERSION 0.0.1 DESCRIPTION "Library wrappers & Tile scheduler for CoCoPeLia" LANGUAGES CXX CUDA)
message( "------------------------------------------------------------------------------------------------")
message( "Library Scheduler - PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR} )
#message( "Library Scheduler - CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR} )
#message( "Library Scheduler - CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR} )

set ( LIBSC_DIR ${PROJECT_SOURCE_DIR}/${BACKENED} )
message( "Library Scheduler - LIBSC_DIR: " ${LIBSC_DIR} )

set (LIBSC_DIR_INC ${LIBSC_DIR}/include)
include_directories( ${LIBSC_DIR_INC})
message( "Library Scheduler - Included DIR: " ${LIBSC_DIR_INC} )

include_directories(${UNIHELP_INSTALL_INC})
message( "Library Scheduler - Included DIR: " ${UNIHELP_INSTALL_INC} )

include_directories(${AUTOTUNE_INSTALL_INC})
message( "Library Scheduler - Included DIR: " ${AUTOTUNE_INSTALL_INC} )

set (LIBSC_DIR_SRC ${LIBSC_DIR}/src)
message( "Library Scheduler - Source DIR: " ${LIBSC_DIR_SRC} )

set (LIBSC_INSTALL ${LIBSC_DIR}/${TESTBED_NAME}-install)
file(MAKE_DIRECTORY ${LIBSC_INSTALL})

set (LIBSC_INSTALL_INC ${LIBSC_INSTALL}/include)
file(MAKE_DIRECTORY ${LIBSC_INSTALL_INC})

set (LIBSC_INSTALL_LIB ${LIBSC_INSTALL}/lib)
file(MAKE_DIRECTORY ${LIBSC_INSTALL_LIB})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBSC_INSTALL_LIB})

set (LIBSC_INSTALL_BIN ${LIBSC_INSTALL}/testing-bin)
file(MAKE_DIRECTORY ${LIBSC_INSTALL_BIN})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIBSC_INSTALL_BIN})

file(MAKE_DIRECTORY ${LIBSC_INSTALL}/testLogs)
add_definitions(-DTESTLIBDIR="${LIBSC_INSTALL}/testLogs")

#OpenBLAS libs and includes (gold1 build)
set (OBLAS_PREFIX "/home/users/panastas/Lib_install/OpenBLAS")
set (OBLAS_INCLUDE_DIRS ${OBLAS_PREFIX}/include)
include_directories(${OBLAS_INCLUDE_DIRS} )
message( "Library Scheduler - Included OpenBLAS: " ${OBLAS_INCLUDE_DIRS} )
set (OBLAS_LD ${OBLAS_PREFIX}/lib)
set (OBLAS_LINK  "-L${OBLAS_LD} -lopenblas" )
message( "Library Scheduler - Linking OpenBLAS: " ${OBLAS_LINK} )


add_library(cocopelia SHARED 
${LIBSC_DIR_SRC}/Dgemm.cu
${LIBSC_DIR_SRC}/TileScheduler.cu
${LIBSC_DIR_SRC}/BLAS3Wrappers.cu
)
target_link_libraries(cocopelia ${INTERNAL_LINK_LIB} autotuning_runtime unihelpers ${OBLAS_LINK})
configure_file(${LIBSC_DIR_INC}/CoCoPeLia.hpp ${LIBSC_INSTALL_INC}/CoCoPeLia.hpp COPYONLY)
configure_file(${LIBSC_DIR_INC}/cuBLASXtWrapped.hpp ${LIBSC_INSTALL_INC}/cuBLASXtWrapped.hpp COPYONLY)

add_executable (CoCopeLiaDgemmRunner ${LIBSC_DIR}/testing/CoCopeLiaDgemmRunner.cpp )
add_executable (CoCopeLiaDgemmTester ${LIBSC_DIR}/testing/CoCopeLiaDgemmTester.cpp )

target_link_libraries(CoCopeLiaDgemmRunner ${INTERNAL_LINK_LIB} cocopelia autotuning_runtime unihelpers)
target_link_libraries(CoCopeLiaDgemmTester ${INTERNAL_LINK_LIB} cocopelia autotuning_runtime unihelpers)


set (LIBSC_INSTALL_INC ${LIBSC_INSTALL_INC} PARENT_SCOPE)
message( "->\nLibrary Scheduler - Shared Library: " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
message( "Library Scheduler - Shared Library headers: " ${LIBSC_INSTALL_INC} )
message( "Library Scheduler - Library testing bin DIR: " ${LIBSC_INSTALL_BIN} )
message( "------------------------------------------------------------------------------------------------")


