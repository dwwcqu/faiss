# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(faiss
  VERSION 1.7.3
  DESCRIPTION "A library for efficient similarity search and clustering of dense vectors."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
  LANGUAGES CXX HIP)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Valid values are "generic", "avx2".
option(FAISS_OPT_LEVEL "" "generic")
option(FAISS_ENABLE_GPU "Enable support for GPU indexes." ON)
option(FAISS_ENABLE_PYTHON "Build Python extension." OFF)
option(FAISS_ENABLE_C_API "Build C API." OFF)
option(BUILD_TESTING "Build Test." OFF)

# ROCm configuration
if(FAISS_ENABLE_GPU)
  list(APPEND CMAKE_PREFIX_PATH "/opt/rocm/")
  set(AMDGPU_TARGETS "gfx916" CACHE STRING "Choose the ROCm architectures of compiler to build")
  set(HIP_ARCHITECTURES "gfx906;gfx916" CACHE STRING "Choose the ROCm architectures of compiler to build")
  set(CMAKE_CXX_COMPILER $ENV{CXX} CACHE STRING
      "Choose the type of compiler to build: Default point to hipcc")
endif()

# if debug compile, set the optimazation 0
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-O0)
endif()
add_subdirectory(faiss)

if(FAISS_ENABLE_GPU)
  add_subdirectory(faiss/gpu)
endif()

if(FAISS_ENABLE_PYTHON)
  add_subdirectory(faiss/python)
endif()

if(FAISS_ENABLE_C_API)
  add_subdirectory(c_api)
endif()

add_subdirectory(demos)
add_subdirectory(benchs)
add_subdirectory(tutorial/cpp)

# CTest must be included in the top level to enable `make test` target.
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)

  if(FAISS_ENABLE_GPU)
    add_subdirectory(faiss/gpu/test)
  endif()
endif()
