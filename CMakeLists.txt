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

option(IBBV_USE_ROARING "Use RoaringBitmap/CRoaring as bit vector implementation." OFF)
option(IBBV_ENABLE_ABV "Enable AdaptiveBitVector. If disabled, AdaptiveBitVector will be an alias for IndexedBlockBitVector." ON)

if(IBBV_USE_ROARING)
  message(STATUS "RoaringBitmap is enabled and acts as the underlying bit vector implementation.")
  add_compile_definitions(IBBV_USE_ROARING=1)
  add_compile_definitions(ROARING_EXCEPTIONS=0)
  target_sources(ibbv PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include/roaring.c)
elseif(IBBV_ENABLE_ABV)
  add_compile_definitions(IBBV_ENABLE_ABV=1)
  message(STATUS "AdaptiveBitVector is enabled.")
else()
  message(STATUS "AdaptiveBitVector is disabled.")
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
