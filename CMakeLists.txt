cmake_minimum_required(VERSION 3.19)
project(ibbv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(src)

add_library(ibbv INTERFACE 
  ${CMAKE_CURRENT_LIST_DIR}/include/IndexedBlockBitVector.h
  ${CMAKE_CURRENT_LIST_DIR}/include/SIMDHelper.h
)
set_target_properties(ibbv PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(ibbv INTERFACE -march=native)
target_include_directories(ibbv INTERFACE
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
)

install(TARGETS ibbv EXPORT ibbvTargets)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION include)
install(EXPORT ibbvTargets
    FILE ibbvConfig.cmake
    NAMESPACE ibbv::
    DESTINATION lib/cmake/ibbv
)

option(IBBV_BENCHMARK "Enable ibbv benchmarking with Google Benchmark" OFF)
if(IBBV_BENCHMARK)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests)
else()
  message(NOTICE "IBBV_BENCHMARK disabled")
endif()