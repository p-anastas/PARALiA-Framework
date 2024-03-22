cmake_minimum_required (VERSION 3.10)
### Initialize the project
project (CoCopeLia-Helpers VERSION ${PROJECT_VERSION} DESCRIPTION "Universal Helper function lib for CoCoPeLia" LANGUAGES CXX CUDA)
message( "------------------------------------------------------------------------------------------------")
message( "Universal Helpers - PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR} )
#message( "Universal Helpers - CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR} )

set ( UNIHELP_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
message( "Universal Helpers - UNIHELP_DIR: " ${UNIHELP_DIR} )

set (UNIHELP_DIR_INC ${UNIHELP_DIR}/include)
message( "Universal Helpers - Included DIR: " ${UNIHELP_DIR_INC} )
include_directories(${UNIHELP_DIR_INC})

set (UNIHELP_DIR_SRC ${UNIHELP_DIR}/src)
message( "Universal Helpers - Source DIR: " ${UNIHELP_DIR_SRC} )

set (BACKEND_UNIHELP_DIR_SRC ${UNIHELP_DIR}/${BACKEND})
message( "Universal Helpers - BACKEND DIR: " ${BACKEND_UNIHELP_DIR_SRC} )
message( "Universal Helpers - Included DIR: " ${BACKEND_UNIHELP_DIR_SRC} )
include_directories(${BACKEND_UNIHELP_DIR_SRC})

set (UNIHELP_INSTALL ${CMAKE_INSTALL_PROJECT}/Universal_helpers)
file(MAKE_DIRECTORY ${UNIHELP_INSTALL})

set (UNIHELP_INSTALL_LIB ${UNIHELP_INSTALL}/lib)
file(MAKE_DIRECTORY ${UNIHELP_INSTALL_LIB})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${UNIHELP_INSTALL_LIB})

set (UNIHELP_INSTALL_INC ${UNIHELP_INSTALL}/include)
file(MAKE_DIRECTORY ${UNIHELP_INSTALL_INC})

## Enable using lazy events; e.g. event backends are created on first record on the corresponding device.
#FIXME: Current CoCoPeLia version only work with lazy events.
add_definitions(-DENABLE_LAZY_EVENTS)

if(${BUILD_BLAS_WRAPPERS_FLAG} EQUAL 1)
    add_library(unihelpers SHARED
    ${UNIHELP_DIR_SRC}/general_utils.cpp
    ${UNIHELP_DIR_SRC}/data_management.cpp
    ${BACKEND_UNIHELP_DIR_SRC}/backend_util_wrappers.cu
    ${BACKEND_UNIHELP_DIR_SRC}/backend_data_wrappers.cu
    ${BACKEND_UNIHELP_DIR_SRC}/backend_class_wrappers.cu
    ${BACKEND_UNIHELP_DIR_SRC}/backend_lib_wrappers.cu
    ${BACKEND_UNIHELP_DIR_SRC}/backend_routine_wrappers.cu
    )
else()
    add_library(unihelpers SHARED
    ${UNIHELP_DIR_SRC}/general_utils.cpp
    ${UNIHELP_DIR_SRC}/data_management.cpp    
    ${BACKEND_UNIHELP_DIR_SRC}/backend_util_wrappers.cu
    ${BACKEND_UNIHELP_DIR_SRC}/backend_data_wrappers.cu
    ${BACKEND_UNIHELP_DIR_SRC}/backend_class_wrappers.cu
    )
endif()

target_link_libraries(unihelpers ${INTERNAL_LINK_LIB} ${OPENBLAS_LINK})
configure_file(${UNIHELP_DIR_INC}/unihelpers.hpp ${UNIHELP_INSTALL_INC}/unihelpers.hpp COPYONLY)

## FIXME: This is a temporary cheat for deployment, must be removed at some point (?). It has to do with TransposeTranslate.
configure_file(${BACKEND_UNIHELP_DIR_SRC}/backend_wrappers.hpp ${UNIHELP_INSTALL_INC}/backend_wrappers.hpp COPYONLY)

set (UNIHELP_INSTALL_INC ${UNIHELP_INSTALL_INC} PARENT_SCOPE)
message( "->\nUniversal Helpers - Shared Library: " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
message( "Universal Helpers - Shared Library headers: " ${UNIHELP_INSTALL_INC} )
message( "------------------------------------------------------------------------------------------------")

#set_target_properties(unihelpers PROPERTIES VERSION ${PROJECT_VERSION})
#set_target_properties(unihelpers PROPERTIES SOVERSION 0)
#set_target_properties(unihelpers PROPERTIES PUBLIC_HEADER ${PROJECT_BACKEND_LIB_DIR_INC}/unihelpers.hpp)
#target_include_directories(mylib PRIVATE .)
#target_include_directories(mylib PRIVATE include)
#target_include_directories(mylib PRIVATE src)

#include(GNUInstallDirs)
#install(TARGETS unihelpers
#    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#configure_file(unihelpers.pc.in unihelpers.pc @ONLY)
#install(FILES ${CMAKE_BINARY_DIR}/unihelpers.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
