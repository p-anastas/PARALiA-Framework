cmake_minimum_required (VERSION 3.10)
### Initialize the project 
project (CoCopeLia-Autotune VERSION ${PROJECT_VERSION} DESCRIPTION "Auto-tuning function lib for CoCoPeLia" LANGUAGES CXX)
message( "------------------------------------------------------------------------------------------------")
message( "Autotuning Runtime - PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR} )
#message( "Autotuning Runtime - CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR} )
#message( "Autotuning Runtime - CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR} )

set ( AUTOTUNE_DIR ${PROJECT_SOURCE_DIR} )
message( "Autotuning Runtime - AUTOTUNE_DIR: " ${AUTOTUNE_DIR} )

set (AUTOTUNE_DIR_INC ${AUTOTUNE_DIR}/include)
include_directories( ${AUTOTUNE_DIR_INC})
message( "Autotuning Runtime - Included DIR: " ${AUTOTUNE_DIR_INC} )

include_directories(${UNIHELP_INSTALL_INC})
message( "Autotuning Runtime - Included DIR: " ${UNIHELP_INSTALL_INC} )

set (AUTOTUNE_DIR_SRC ${AUTOTUNE_DIR}/src)
message( "Autotuning Runtime - Source DIR: " ${AUTOTUNE_DIR_SRC} )

set (AUTOTUNE_INSTALL ${CMAKE_INSTALL_PROJECT}/Autotuning_Runtime)
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
${AUTOTUNE_DIR_SRC}/CoCoPeLiaCoModel.cpp
${AUTOTUNE_DIR_SRC}/CoCoPeLiaGPUexec.cpp
${AUTOTUNE_DIR_SRC}/CoCoPeLiaModel.cpp
${AUTOTUNE_DIR_SRC}/WerkhovenModel.cpp
)
target_link_libraries(autotuning_runtime ${INTERNAL_LINK_LIB} unihelpers)
configure_file(${AUTOTUNE_DIR_INC}/CoCoPeLiaCoModel.hpp ${AUTOTUNE_INSTALL_INC}/CoCoPeLiaCoModel.hpp COPYONLY)
configure_file(${AUTOTUNE_DIR_INC}/CoCoPeLiaModel.hpp ${AUTOTUNE_INSTALL_INC}/CoCoPeLiaModel.hpp COPYONLY)

add_executable (CoCoPeLiaPredictTileTester ${AUTOTUNE_DIR}/testing/CoCoPeLiaPredictTileTester.cpp )
target_link_libraries(CoCoPeLiaPredictTileTester ${INTERNAL_LINK_LIB} autotuning_runtime unihelpers)

set (AUTOTUNE_INSTALL_INC ${AUTOTUNE_INSTALL_INC} PARENT_SCOPE)
message( "->\nAutotuning Runtime - Shared Library: " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
message( "Autotuning Runtime - Shared Library headers: " ${AUTOTUNE_INSTALL_INC} )
message( "Autotuning Runtime - Model testing bin DIR: " ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} )
message( "------------------------------------------------------------------------------------------------")


