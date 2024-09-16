cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR "In-source build is not supported")
endif()

set(PROJECT_NAME cpplm)

project(${PROJECT_NAME} VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED true)

include(FetchContent)
FetchContent_Declare(
	googletest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG main
	FIND_PACKAGE_ARGS NAMES googletest
)
FetchContent_MakeAvailable(googletest)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# uncomment for testing
enable_testing()
include(GoogleTest)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils.cmake)

create_tests_from_sources(
	SOURCES tests/modules.test.cpp
	LIBRARIES torch
)
