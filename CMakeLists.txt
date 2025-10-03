cmake_minimum_required(VERSION 3.19)
project(ibbv LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(ibbv STATIC
  ${CMAKE_CURRENT_LIST_DIR}/include/AdaptiveBitVector.hpp
)
set_target_properties(ibbv PROPERTIES LINKER_LANGUAGE CXX)
# target_compile_definitions(ibbv INTERFACE IBBV_COUNT_OP=1)
target_compile_options(ibbv INTERFACE -march=native)
target_include_directories(ibbv INTERFACE
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
)

find_package(mimalloc 2.2 REQUIRED)
target_link_libraries(ibbv INTERFACE mimalloc-static)

set(IBBV_IMPL "ibbv" CACHE STRING "Set the implementation of the bit vector.")
set_property(CACHE IBBV_IMPL PROPERTY STRINGS "ibbv" "abv" "roaring" "wah" "concise")
add_compile_definitions(IBBV_IMPL_IBBV=0 IBBV_IMPL_ABV=1 IBBV_IMPL_ROARING=2 IBBV_IMPL_WAH=3 IBBV_IMPL_CONCISE=4)

if(IBBV_IMPL STREQUAL "concise")
  add_compile_definitions(IBBV_IMPL=4)
  message(STATUS "Using Concise")
elseif(IBBV_IMPL STREQUAL "wah")
  add_compile_definitions(IBBV_IMPL=3)
  message(STATUS "Using WAH")
elseif(IBBV_IMPL STREQUAL "roaring")
  message(STATUS "RoaringBitmap is enabled and acts as the underlying bit vector implementation.")
  add_compile_definitions(ROARING_EXCEPTIONS=0)
  target_include_directories(ibbv PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include/Roaring)
  target_sources(ibbv PRIVATE ${CMAKE_CURRENT_LIST_DIR}/lib/Roaring/roaring.c)
  add_compile_definitions(IBBV_IMPL=2)
  message(STATUS "Using Roaring")
elseif(IBBV_IMPL STREQUAL "abv")
  add_compile_definitions(IBBV_IMPL=1)
  message(STATUS "AdaptiveBitVector is enabled")
elseif(IBBV_IMPL STREQUAL "ibbv")
  add_compile_definitions(IBBV_IMPL=0)
  message(STATUS "AdaptiveBitVector is disabled. Using IBBV")
else()
  message(FATAL_ERROR "Unknown IBBV_IMPL value: ${IBBV_IMPL}. Supported values are: ibbv, abv, roaring.")
endif()

install(TARGETS ibbv EXPORT ibbvTargets)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION include)
install(EXPORT ibbvTargets
  FILE ibbvConfig.cmake
  NAMESPACE ibbv::
  DESTINATION lib/cmake/ibbv
)

option(IBBV_TESTS "Enable ibbv benchmarking with Google Benchmark" OFF)
if(IBBV_TESTS)
  message(NOTICE "IBBV_TESTS enabled, target included")
  include(CTest)
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests)
else()
  message(NOTICE "IBBV_TESTS disabled")
endif()
