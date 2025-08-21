cmake_minimum_required(VERSION 3.15)

option(IBBV_BENCHMARK "Enable ibbv benchmarking with Google Benchmark" OFF)
if(IBBV_BENCHMARK)
  message(NOTICE "IBBV_BENCHMARK enabled, target included")
  include(FetchContent)
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
  )

  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googlebenchmark)

  add_executable(ibbv_bench EXCLUDE_FROM_ALL ${CMAKE_CURRENT_LIST_DIR}/bench.cpp)
  target_include_directories(ibbv_bench PUBLIC
    ${PROJECT_SOURCE_DIR}/include
  )
  target_link_libraries(ibbv_bench benchmark::benchmark)
else()
  message(NOTICE "IBBV_BENCHMARK disabled")
endif()