cmake_minimum_required(VERSION 3.15)

message(NOTICE "IBBV_BENCHMARK enabled, target included")
include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.4
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(ibbv_bench EXCLUDE_FROM_ALL bench.cpp)
add_compile_options(ibbv_bench PRIVATE -fno-exceptions -fno-rtti)
target_link_libraries(ibbv_bench ibbv benchmark::benchmark)
