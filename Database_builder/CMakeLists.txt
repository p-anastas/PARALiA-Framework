cmake_minimum_required (VERSION 3.10)
### Initialize the project
project (CoCopeLia-Deployment VERSION ${PROJECT_VERSION} DESCRIPTION "The Deployment component of the CoCoPeLia framework" LANGUAGES CXX CUDA)
message( "------------------------------------------------------------------------------------------------")
message( "Database Builder - PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR} )

set ( DEPLOYMENT_DIR ${PROJECT_SOURCE_DIR}/${BACKEND} )
message( "Database Builder - DEPLOYMENT_DIR: " ${DEPLOYMENT_DIR} )

include_directories( ${DEPLOYMENT_DIR}/include)
message( "Database Builder - Included DIR: " ${DEPLOYMENT_DIR}/include )

include_directories(${UNIHELP_INSTALL_INC})
message( "Database Builder - Included DIR: " ${UNIHELP_INSTALL_INC} )

set(NVEMAPI_PREFIX "${DEPLOYMENT_DIR}/nvidia-energy-measure")
set (NVEMAPI_INCLUDE_DIRS ${NVEMAPI_PREFIX}/include) # -L${CUDA_LD}
include_directories(${NVEMAPI_INCLUDE_DIRS} )
message( "Database Builder - NVEMAPI_INCLUDE_DIRS: " ${NVEMAPI_INCLUDE_DIRS} )
ADD_SUBDIRECTORY (CuCuBLAS/nvidia-energy-measure)

## TODO: Choose between ->
## i) Automatic micro-benchmarking (Population checked within 95% CI) using boost (recommended if boost available).
message("Database Builder - Using Boost for microbenchmark statistics")
add_definitions(-DAUTO_BENCH_USE_BOOST)
set(BOOSTO_PREFIX "/home/users/panastas/Lib_install/boost-1.69")
set (BOOSTO_INCLUDE_DIRS ${BOOSTO_PREFIX}/include)
include_directories(${BOOSTO_INCLUDE_DIRS})
message("Database Builder - Included (Boost) DIR: " ${BOOSTO_INCLUDE_DIRS})
## OR
## ii) Benchmark with a predifined itteration number for each problem size.
#add_definitions(-DITER=100)
#message("Database Builder - Performing predefined microbenchmark itterations")

set (DEPLOYMENT_DIR_SRC ${DEPLOYMENT_DIR}/src)
message( "Database Builder - Source DIR: " ${DEPLOYMENT_DIR_SRC} )

set(DEPLOYMENT_INSTALL_PREFIX ${CMAKE_INSTALL_PROJECT}/Deployment_phase)
file(MAKE_DIRECTORY ${DEPLOYMENT_INSTALL_PREFIX})

set (DEPLOYMENT_DIR_BIN ${DEPLOYMENT_INSTALL_PREFIX}/bin )
file(MAKE_DIRECTORY ${DEPLOYMENT_DIR_BIN})
message( "->\nDatabase Builder - Microbenchmark bin DIR: " ${DEPLOYMENT_DIR_BIN} )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DEPLOYMENT_DIR_BIN})

set (DEPLOYMENT_DIR_DB ${DEPLOYMENT_INSTALL_PREFIX}/Database )
file(MAKE_DIRECTORY ${DEPLOYMENT_DIR_DB})
## Database location variable set at CoCoPeLia-Framework Project (parent) scope for result input.
set (DEPLOYMENT_DIR_DB ${DEPLOYMENT_DIR_DB} PARENT_SCOPE)
message( "Database Builder - Database DIR: " ${DEPLOYMENT_DIR_DB} )

add_definitions(-DDEPLOYDB="${DEPLOYMENT_DIR_DB}")

add_executable (RunCuCuBlasDdotBench ${DEPLOYMENT_DIR_SRC}/ddot_microbench.cu )
add_executable (RunCuCuBlasDaxpyBench ${DEPLOYMENT_DIR_SRC}/daxpy_microbench.cu )
add_executable (RunCuCuBlasDgemmBench ${DEPLOYMENT_DIR_SRC}/dgemm_microbench.cu )
add_executable (RunCuCuBlasSgemmBench ${DEPLOYMENT_DIR_SRC}/sgemm_microbench.cu )
add_executable (RunCuCuBlasLinkBench ${DEPLOYMENT_DIR_SRC}/transfer_link_microbench.cpp )

target_link_libraries(RunCuCuBlasDdotBench ${INTERNAL_LINK_LIB} unihelpers nvidia_powa)
target_link_libraries(RunCuCuBlasDaxpyBench ${INTERNAL_LINK_LIB} unihelpers nvidia_powa)
target_link_libraries(RunCuCuBlasDgemmBench ${INTERNAL_LINK_LIB} unihelpers nvidia_powa)
target_link_libraries(RunCuCuBlasSgemmBench ${INTERNAL_LINK_LIB} unihelpers nvidia_powa)
target_link_libraries(RunCuCuBlasLinkBench ${INTERNAL_LINK_LIB} unihelpers nvidia_powa)

configure_file(${PROJECT_SOURCE_DIR}/RunMicrobenchmarks.in ${DEPLOYMENT_INSTALL_PREFIX}/RunMicrobenchmarks.sh @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/ProcessDatabase.in ${DEPLOYMENT_INSTALL_PREFIX}/ProcessDatabase.sh @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/Python-DataManage/BenchToLookupTables.py.in ${DEPLOYMENT_INSTALL_PREFIX}/BenchToLookupTables.py @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/Python-DataManage/LinkPreprocessing.py.in ${DEPLOYMENT_INSTALL_PREFIX}/LinkPreprocessing.py @ONLY)

message( "------------------------------------------------------------------------------------------------")
