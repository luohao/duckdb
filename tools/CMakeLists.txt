if(NOT SUN AND BUILD_SHELL)
  add_subdirectory(shell)
endif()

add_executable(plan_bench utils/plan_bench.cpp)
target_link_libraries(plan_bench duckdb_static ${DUCKDB_EXTRA_LINK_FLAGS})
link_extension_libraries(plan_bench "")

add_executable(kway_bench utils/kway_bench.cpp)
target_link_libraries(kway_bench duckdb_static ${DUCKDB_EXTRA_LINK_FLAGS})
link_extension_libraries(kway_bench "")

add_executable(hash_dump utils/hash_dump/hash_dump.cpp)
target_link_libraries(hash_dump duckdb_static ${DUCKDB_EXTRA_LINK_FLAGS})
link_extension_libraries(hash_dump "")

add_executable(inner_bench utils/inner_bench/inner_bench.cpp)
target_link_libraries(inner_bench duckdb_static ${DUCKDB_EXTRA_LINK_FLAGS})
link_extension_libraries(inner_bench "")
