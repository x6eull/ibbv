cmake_minimum_required(VERSION 3.15)

if(CMAKE_PROJECT_NAME STREQUAL ibbv)
  include(FetchContent)
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
  )

  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googlebenchmark)

  add_executable(ibbv_bench ${CMAKE_CURRENT_LIST_DIR}/bench.cpp)
  target_link_libraries(ibbv_bench benchmark::benchmark)
endif()