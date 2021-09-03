cmake_minimum_required (VERSION 3.10)
### Initialize the project 
project (CoCopeLia-Library VERSION ${PROJECT_VERSION} DESCRIPTION "Library wrappers & Tile scheduler for CoCoPeLia" LANGUAGES CXX CUDA)
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

set (LIBSC_INSTALL ${CMAKE_INSTALL_PROJECT}/Library_scheduler)
file(MAKE_DIRECTORY ${LIBSC_INSTALL})

set (LIBSC_INSTALL_INC ${LIBSC_INSTALL}/include)
file(MAKE_DIRECTORY ${LIBSC_INSTALL_INC})

set (LIBSC_INSTALL_LIB ${LIBSC_INSTALL}/lib)
file(MAKE_DIRECTORY ${LIBSC_INSTALL_LIB})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBSC_INSTALL_LIB})

add_library(cocopelia SHARED 
${LIBSC_DIR_SRC}/Dgemm.cpp
${LIBSC_DIR_SRC}/TileScheduler.cpp
)
target_link_libraries(cocopelia ${INTERNAL_LINK_LIB} ${XKBLAS_LINK} ${BLASX_LINK} ${OBLAS_LINK} autotuning_runtime unihelpers )
configure_file(${LIBSC_DIR_INC}/CoCoPeLia.hpp ${LIBSC_INSTALL_INC}/CoCoPeLia.hpp COPYONLY)

set (LIBSC_INSTALL_INC ${LIBSC_INSTALL_INC} PARENT_SCOPE)
message( "->\nLibrary Scheduler - Shared Library: " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
message( "Library Scheduler - Shared Library headers: " ${LIBSC_INSTALL_INC} )
message( "------------------------------------------------------------------------------------------------")


