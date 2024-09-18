cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR "In-source build is not supported")
endif()

set(PROJECT_NAME cpplm)

project(${PROJECT_NAME} VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED true)

if("${CMAKE_BUILD_TYPE}" STREQUAL Debug)
	set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

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

create_test_from_file(
	FILE tests/modules.test.cpp
	SOURCES cpplm/modules.cpp cpplm/modules.hpp
	LIBRARIES torch
)

create_test_from_file(
	FILE tests/model.test.cpp 
	SOURCES cpplm/model.hpp cpplm/model.cpp cpplm/modules.hpp cpplm/modules.cpp 
	LIBRARIES torch
)
