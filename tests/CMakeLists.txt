cmake_minimum_required(VERSION 3.15)

include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.4
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(ibbv_bench EXCLUDE_FROM_ALL bench.cpp)
target_compile_options(ibbv_bench PRIVATE -fno-exceptions -fno-rtti)
target_link_libraries(ibbv_bench ibbv benchmark::benchmark)


FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
add_executable(ibbv_test EXCLUDE_FROM_ALL test.cpp)
target_link_libraries(ibbv_test ibbv GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(ibbv_test)