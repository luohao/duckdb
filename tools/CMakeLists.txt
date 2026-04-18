if(NOT SUN AND BUILD_SHELL)
  add_subdirectory(shell)
endif()

add_executable(plan_bench utils/plan_bench.cpp)
target_link_libraries(plan_bench duckdb_static ${DUCKDB_EXTRA_LINK_FLAGS})
link_extension_libraries(plan_bench "")

add_executable(kway_bench utils/kway_bench.cpp)
target_link_libraries(kway_bench duckdb_static ${DUCKDB_EXTRA_LINK_FLAGS})
link_extension_libraries(kway_bench "")
