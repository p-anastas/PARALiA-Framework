cmake_minimum_required (VERSION 3.10)
### Initialize the project
project (CoCopeLia-Autotune VERSION ${PROJECT_VERSION} DESCRIPTION "Auto-tuning function lib for CoCoPeLia" LANGUAGES CXX)
message( "------------------------------------------------------------------------------------------------")
message( "Autotuner - PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR} )
#message( "Autotuner - CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR} )
#message( "Autotuner - CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR} )

## Define which cost function to use for device selection
add_definitions(-DDEV_COST_FUNC="FULL-OVERLAP")
#add_definitions(-DDEV_COST_FUNC="ZERO-OVERLAP")

## Define an extra percentile overhead for Tile sizes which lead to padding and possible imbalance
## See autotuner->CoCoPeLiaModel->CoCopeLiaPredictReuseHetero
add_definitions(-DTILE_IMBALANCE_PENALTY=0.05)

## Define an extra percentile overhead for Tile sizes which lead to padding and possible imbalance
## See autotuner->CoCoPeLiaModel->CoCopeLiaPredictReuseHetero
add_definitions(-DREDUCE_PENALTY=0.1)

## Define which mode to use when calculating relative performance for a problem
## See autotuner->CoCoPeLiaModel->CoCopeLiaPredictReuseHetero
add_definitions(-DREL_PERF_MODE="PERCENTILE")
#add_definitions(-DREL_PERF_MODE="ROOT-PROBLEM")

set ( AUTOTUNE_DIR ${PROJECT_SOURCE_DIR} )
message( "Autotuner - AUTOTUNE_DIR: " ${AUTOTUNE_DIR} )

set (AUTOTUNE_DIR_INC ${AUTOTUNE_DIR}/include)
include_directories( ${AUTOTUNE_DIR_INC})
message( "Autotuner - Included DIR: " ${AUTOTUNE_DIR_INC} )

include_directories(${UNIHELP_INSTALL_INC})
message( "Autotuner - Included DIR: " ${UNIHELP_INSTALL_INC} )

set (AUTOTUNE_DIR_SRC ${AUTOTUNE_DIR}/src)
message( "Autotuner - Source DIR: " ${AUTOTUNE_DIR_SRC} )

set (AUTOTUNE_INSTALL ${CMAKE_INSTALL_PROJECT}/Autotuner)
file(MAKE_DIRECTORY ${AUTOTUNE_INSTALL})

set (AUTOTUNE_INSTALL_INC ${AUTOTUNE_INSTALL}/include)
file(MAKE_DIRECTORY ${AUTOTUNE_INSTALL_INC})

set (AUTOTUNE_INSTALL_LIB ${AUTOTUNE_INSTALL}/lib)
file(MAKE_DIRECTORY ${AUTOTUNE_INSTALL_LIB})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${AUTOTUNE_INSTALL_LIB})

set (AUTOTUNE_INSTALL_BIN ${AUTOTUNE_INSTALL}/bin)
file(MAKE_DIRECTORY ${AUTOTUNE_INSTALL_BIN})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${AUTOTUNE_INSTALL_BIN})

file(MAKE_DIRECTORY ${AUTOTUNE_INSTALL}/testLogs)
add_definitions(-DTESTDIR="${AUTOTUNE_INSTALL}/testLogs")

add_library(autotuner SHARED
#${AUTOTUNE_DIR_SRC}/TransferLinks.cpp
${AUTOTUNE_DIR_SRC}/LinkMap.cpp
${AUTOTUNE_DIR_SRC}/CoModel.cpp
${AUTOTUNE_DIR_SRC}/GPUexec_lookup.cpp
${AUTOTUNE_DIR_SRC}/Model_functions.cpp
${AUTOTUNE_DIR_SRC}/ModelLvl1.cpp
${AUTOTUNE_DIR_SRC}/ModelLvl3.cpp
${AUTOTUNE_DIR_SRC}/WerkhovenModel.cpp
${AUTOTUNE_DIR_SRC}/Subkernel_distributions.cpp
${AUTOTUNE_DIR_SRC}/Autotuner.cpp
${AUTOTUNE_DIR_SRC}/Modeler.cpp
)
target_link_libraries(autotuner ${INTERNAL_LINK_LIB} unihelpers)
configure_file(${AUTOTUNE_DIR_INC}/CoModel.hpp ${AUTOTUNE_INSTALL_INC}/CoModel.hpp COPYONLY)
configure_file(${AUTOTUNE_DIR_INC}/Autotuner.hpp ${AUTOTUNE_INSTALL_INC}/Autotuner.hpp COPYONLY)

#add_executable (CoCoPeLiaPredictTileTester ${AUTOTUNE_DIR}/testing/CoCoPeLiaPredictTileTester.cpp )
#target_link_libraries(CoCoPeLiaPredictTileTester ${INTERNAL_LINK_LIB} autotuner unihelpers)

set (AUTOTUNE_INSTALL_INC ${AUTOTUNE_INSTALL_INC} PARENT_SCOPE)
message( "->\nAutotuner - Shared Library: " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
message( "Autotuner - Shared Library headers: " ${AUTOTUNE_INSTALL_INC} )
message( "Autotuner - Model testing bin DIR: " ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} )
message( "------------------------------------------------------------------------------------------------")
