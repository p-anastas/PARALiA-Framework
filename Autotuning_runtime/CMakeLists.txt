cmake_minimum_required (VERSION 3.10)
### Initialize the project
project (CoCopeLia-Autotune VERSION ${PROJECT_VERSION} DESCRIPTION "Auto-tuning function lib for CoCoPeLia" LANGUAGES CXX)
message( "------------------------------------------------------------------------------------------------")
message( "Autotuning Runtime exp - PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR} )
#message( "Autotuning Runtime exp - CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR} )
#message( "Autotuning Runtime exp - CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR} )

## Define which cost function to use for device selection
add_definitions(-DDEV_COST_FUNC="FULL-OVERLAP")
#add_definitions(-DDEV_COST_FUNC="ZERO-OVERLAP")

## Define an extra percentile overhead for Tile sizes which lead to padding and possible imbalance
## See Autotuning_runtime->CoCoPeLiaModel->CoCopeLiaPredictReuseHetero
add_definitions(-DTILE_IMBALANCE_PENALTY=0.05)

## Define an extra percentile overhead for Tile sizes which lead to padding and possible imbalance
## See Autotuning_runtime->CoCoPeLiaModel->CoCopeLiaPredictReuseHetero
add_definitions(-DREDUCE_PENALTY=0.1)

## Define which mode to use when calculating relative performance for a problem
## See Autotuning_runtime->CoCoPeLiaModel->CoCopeLiaPredictReuseHetero
add_definitions(-DREL_PERF_MODE="PERCENTILE")
#add_definitions(-DREL_PERF_MODE="ROOT-PROBLEM")

set ( AUTOTUNE_DIR ${PROJECT_SOURCE_DIR} )
message( "Autotuning Runtime exp - AUTOTUNE_DIR: " ${AUTOTUNE_DIR} )

set (AUTOTUNE_DIR_INC ${AUTOTUNE_DIR}/include)
include_directories( ${AUTOTUNE_DIR_INC})
message( "Autotuning Runtime exp - Included DIR: " ${AUTOTUNE_DIR_INC} )

include_directories(${UNIHELP_INSTALL_INC})
message( "Autotuning Runtime exp - Included DIR: " ${UNIHELP_INSTALL_INC} )

set (AUTOTUNE_DIR_SRC ${AUTOTUNE_DIR}/src)
message( "Autotuning Runtime exp - Source DIR: " ${AUTOTUNE_DIR_SRC} )

set (AUTOTUNE_INSTALL ${CMAKE_INSTALL_PROJECT}/Autotuning_Runtime_exp)
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

add_library(autotuning_runtime SHARED
${AUTOTUNE_DIR_SRC}/CoCoPeLiaLinks.cpp
${AUTOTUNE_DIR_SRC}/CoCoPeLiaCoModel.cpp
${AUTOTUNE_DIR_SRC}/CoCoPeLiaGPUexec.cpp
${AUTOTUNE_DIR_SRC}/CoCoPeLiaModelWrap.cpp
${AUTOTUNE_DIR_SRC}/CoCoPeLiaModelLvl1.cpp
${AUTOTUNE_DIR_SRC}/CoCoPeLiaModelLvl3.cpp
${AUTOTUNE_DIR_SRC}/WerkhovenModel.cpp
${AUTOTUNE_DIR_SRC}/Distributions.cpp
)
target_link_libraries(autotuning_runtime ${INTERNAL_LINK_LIB} unihelpers)
configure_file(${AUTOTUNE_DIR_INC}/CoCoPeLiaCoModel.hpp ${AUTOTUNE_INSTALL_INC}/CoCoPeLiaCoModel.hpp COPYONLY)
configure_file(${AUTOTUNE_DIR_INC}/CoCoPeLiaModel.hpp ${AUTOTUNE_INSTALL_INC}/CoCoPeLiaModel.hpp COPYONLY)

#add_executable (CoCoPeLiaPredictTileTester ${AUTOTUNE_DIR}/testing/CoCoPeLiaPredictTileTester.cpp )
#target_link_libraries(CoCoPeLiaPredictTileTester ${INTERNAL_LINK_LIB} autotuning_runtime unihelpers)

set (AUTOTUNE_INSTALL_INC ${AUTOTUNE_INSTALL_INC} PARENT_SCOPE)
message( "->\nAutotuning Runtime exp - Shared Library: " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
message( "Autotuning Runtime exp - Shared Library headers: " ${AUTOTUNE_INSTALL_INC} )
message( "Autotuning Runtime exp - Model testing bin DIR: " ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} )
message( "------------------------------------------------------------------------------------------------")
