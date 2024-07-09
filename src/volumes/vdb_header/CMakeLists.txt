cmake_minimum_required(VERSION 3.18)
project(vdb_header LANGUAGES C CXX)

# NanoVDB header files
set(VDB_INCLUDE_FILES
  defines.h
  impl_drjit.h
#   monolith_pnano.h
)

add_library(vdb_header INTERFACE)
# target_sources(vdb_header INTERFACE impl_drjit.h)
target_include_directories(vdb_header INTERFACE .)

set(VDB_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/vdb_header)

install(FILES ${VDB_INCLUDE_FILES} DESTINATION ${VDB_INSTALL_INCLUDEDIR})