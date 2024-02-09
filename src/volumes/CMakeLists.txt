set(MI_PLUGIN_PREFIX "volumes")

add_plugin(constvolume  const.cpp)
add_plugin(gridvolume   grid.cpp)
add_plugin(gridtest     gridtest.cpp)

set(NANOVDB_USE_BLOSC OFF CACHE BOOL "use Blosc" FORCE)
add_subdirectory(nanovdb/nanovdb)

set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
