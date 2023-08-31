cmake_minimum_required (VERSION 3.10)
### Initialize the project
project (PARALiA-Library VERSION ${PROJECT_VERSION} DESCRIPTION "Experiments for advancing Library wrappers & Tile scheduler for PARALiA" LANGUAGES CXX CUDA)
message( "------------------------------------------------------------------------------------------------")
message( "Library Scheduler - PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR} )
#message( "Library Scheduler - CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR} )
#message( "Library Scheduler - CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR} )

set ( LIBSC_DIR ${PROJECT_SOURCE_DIR} )
message( "Library Scheduler - LIBSC_DIR: " ${LIBSC_DIR} )

set (LIBSC_DIR_INC ${LIBSC_DIR}/include)
include_directories( ${LIBSC_DIR_INC})
message( "Library Scheduler - Included DIR: " ${LIBSC_DIR_INC} )

include_directories(${LINKMAP_INSTALL_INC})
message( "Library Scheduler - Included DIR: " ${LINKMAP_INSTALL_INC} )

include_directories(${AUTOTUNE_INSTALL_INC})
message( "Library Scheduler - Included DIR: " ${AUTOTUNE_INSTALL_INC} )

include_directories(${CACHING_INSTALL_INC})
message( "Library Scheduler - Included DIR: " ${CACHING_INSTALL_INC} )

set (LIBSC_DIR_SRC ${LIBSC_DIR}/src)
message( "Library Scheduler - Source DIR: " ${LIBSC_DIR_SRC} )

set ( BACKEND_LIBSC_DIR ${LIBSC_DIR}/${BACKEND} )
message( "Library Scheduler - BACKEND_LIBSC_DIR: " ${BACKEND_LIBSC_DIR} )

set (LIBSC_INSTALL ${CMAKE_INSTALL_PROJECT}/Library_scheduler)
file(MAKE_DIRECTORY ${LIBSC_INSTALL})

set (LIBSC_INSTALL_INC ${LIBSC_INSTALL}/include)
file(MAKE_DIRECTORY ${LIBSC_INSTALL_INC})

set (LIBSC_INSTALL_LIB ${LIBSC_INSTALL}/lib)
file(MAKE_DIRECTORY ${LIBSC_INSTALL_LIB})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBSC_INSTALL_LIB})

configure_file(${LIBSC_DIR_INC}/Subkernel.hpp ${LIBSC_INSTALL_INC}/Subkernel.hpp COPYONLY)

add_library(paralia SHARED
${LIBSC_DIR_SRC}/Subkernel.cpp
${LIBSC_DIR_SRC}/Functions/dgemm.cpp
#${LIBSC_DIR_SRC}/Functions/sgemm.cpp
#${LIBSC_DIR_SRC}/Functions/dgemv.cpp
#${LIBSC_DIR_SRC}/Functions/daxpy.cpp
#${LIBSC_DIR_SRC}/Functions/ddot.cpp
)
target_link_libraries(paralia ${INTERNAL_LINK_LIB} autotuner linkmap caching)
configure_file(${LIBSC_DIR_INC}/PARALiA.hpp ${LIBSC_INSTALL_INC}/PARALiA.hpp COPYONLY)

set (LIBSC_INSTALL_INC ${LIBSC_INSTALL_INC} PARENT_SCOPE)
message( "->\nLibrary Scheduler - Shared Library: " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
message( "Library Scheduler - Shared Library headers: " ${LIBSC_INSTALL_INC} )
message( "------------------------------------------------------------------------------------------------")
