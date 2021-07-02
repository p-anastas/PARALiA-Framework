cmake_minimum_required (VERSION 3.10)
### Initialize the project 
project (CoCopeLia-Deployment CXX CUDA)
message( "------------------------------------------------------------------------------------------------")
message( "Deployment Phase - PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR} )
#message( "Deployment Phase - CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR} )
#message( "Deployment Phase - CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR} )

set ( DEPLOYMENT_DIR ${PROJECT_SOURCE_DIR}/${BACKENED} )
message( "Deployment Phase - DEPLOYMENT_DIR: " ${DEPLOYMENT_DIR} )

include_directories( ${DEPLOYMENT_DIR}/include)
message( "Deployment Phase - Included DIR: " ${DEPLOYMENT_DIR}/include )

include_directories(${UNIHELP_INSTALL_INC})
message( "Deployment Phase - Included DIR: " ${UNIHELP_INSTALL_INC} )

set (DEPLOYMENT_DIR_SRC ${DEPLOYMENT_DIR}/src)
message( "Deployment Phase - Source DIR: " ${DEPLOYMENT_DIR_SRC} )

file(MAKE_DIRECTORY ${DEPLOYMENT_DIR}/${TESTBED_NAME}-install)

file(MAKE_DIRECTORY ${DEPLOYMENT_DIR}/${TESTBED_NAME}-install/bin)
set ( DEPLOYMENT_DIR_BIN ${DEPLOYMENT_DIR}/${TESTBED_NAME}-install/bin )
message( "->\nDeployment Phase - Microbenchmark bin DIR: " ${DEPLOYMENT_DIR_BIN} )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DEPLOYMENT_DIR_BIN})

file(MAKE_DIRECTORY ${DEPLOYMENT_DIR}/${TESTBED_NAME}-install/Database)
set ( DEPLOYMENT_DIR_DB ${DEPLOYMENT_DIR}/${TESTBED_NAME}-install/Database )
## Database location variable set at CoCoPeLia-Framework Project (parent) scope for result input.
set (DEPLOYMENT_DIR_DB ${DEPLOYMENT_DIR_DB} PARENT_SCOPE)
message( "Deployment Phase - Database DIR: " ${DEPLOYMENT_DIR_DB} )

add_definitions(-DDEPLOYDIR="${DEPLOYMENT_DIR}")

## Choose between ->
## i) Automatic micro-benchmarking (Population checked within 95% CI) using boost (recommended if boost available). 
#add_definitions(-DAUTO_BENCH_USE_BOOST)
## OR 
## ii) Benchmark with a predifined itteration number for each problem size. 
add_definitions(-DITER=100)

add_executable (RunCuCuBlasDaxpyBench ${DEPLOYMENT_DIR_SRC}/daxpy_microbench_gpu.cu )
add_executable (RunCuCuBlasDgemmBench ${DEPLOYMENT_DIR_SRC}/dgemm_microbench_gpu.cu )
add_executable (RunCuCuBlasSgemmBench ${DEPLOYMENT_DIR_SRC}/sgemm_microbench_gpu.cu )
add_executable (RunCuCuBlasTransferBench ${DEPLOYMENT_DIR_SRC}/transfers_microbench_gpu.cu )

target_link_libraries(RunCuCuBlasDaxpyBench ${INTERNAL_LINK_LIB} unihelpers)
target_link_libraries(RunCuCuBlasDgemmBench ${INTERNAL_LINK_LIB} unihelpers)
target_link_libraries(RunCuCuBlasSgemmBench ${INTERNAL_LINK_LIB} unihelpers)
target_link_libraries(RunCuCuBlasTransferBench ${INTERNAL_LINK_LIB} unihelpers)

message( "------------------------------------------------------------------------------------------------")


