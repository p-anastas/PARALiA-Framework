cmake_minimum_required (VERSION 3.10)
### Initialize the project
project (Software-Caching VERSION ${PROJECT_VERSION} DESCRIPTION "Software caching implementation for automated multi-GPU data movement" LANGUAGES CXX CUDA)
message( "------------------------------------------------------------------------------------------------")
message( "Software Caching - PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR} )
#message( "Library Scheduler - CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR} )
#message( "Library Scheduler - CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR} )

set ( CACHING_DIR ${PROJECT_SOURCE_DIR} )
message( "Software Caching - CACHING_DIR: " ${CACHING_DIR} )

set (CACHING_DIR_INC ${CACHING_DIR}/include)
include_directories( ${CACHING_DIR_INC})
message( "Software Caching - Included DIR: " ${CACHING_DIR_INC} )

include_directories(${UNIHELP_INSTALL_INC})
message( "Software Caching - Included DIR: " ${UNIHELP_INSTALL_INC} )

include_directories(${LIBSC_INSTALL_INC})
message( "Software Caching - Included DIR: " ${LIBSC_INSTALL_INC} )

set (CACHING_DIR_SRC ${CACHING_DIR}/src)
message( "Software Caching - Source DIR: " ${CACHING_DIR_SRC} )

set (CACHING_INSTALL ${CMAKE_INSTALL_PROJECT}/Library_scheduler_exp/Software_caching)
file(MAKE_DIRECTORY ${CACHING_INSTALL})

set (CACHING_INSTALL_INC ${CACHING_INSTALL}/include)
file(MAKE_DIRECTORY ${CACHING_INSTALL_INC})

set (CACHING_INSTALL_LIB ${CACHING_INSTALL}/lib)
file(MAKE_DIRECTORY ${CACHING_INSTALL_LIB})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CACHING_INSTALL_LIB})

add_library(caching SHARED
${CACHING_DIR_SRC}/DataCaching.cpp
)

set (CACHE_VAL 0)
add_definitions(-DCACHE_SCHEDULING_POLICY=${CACHE_VAL})
set (CACHE_VAL ${CACHE_VAL} PARENT_SCOPE)

target_link_libraries(caching ${INTERNAL_LINK_LIB} unihelpers )
configure_file(${CACHING_DIR_INC}/DataCaching.hpp ${CACHING_INSTALL_INC}/DataCaching.hpp COPYONLY)

set (CACHING_INSTALL_INC ${CACHING_INSTALL_INC} PARENT_SCOPE)
message( "->\nSoftware Caching - Shared Library: " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
message( "Software Caching - Shared Library headers: " ${CACHING_INSTALL_INC} )
message( "------------------------------------------------------------------------------------------------")
