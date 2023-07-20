# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

set(FAISS_LANGUAGES CXX)

if(FAISS_ENABLE_GPU)
  list(APPEND FAISS_LANGUAGES CUDA)
endif()

if(FAISS_ENABLE_RAFT)
include(cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(faiss)
rapids_cuda_init_architectures(pyfaiss)
rapids_cuda_init_architectures(faiss_c_library)
endif()

project(faiss
  VERSION 1.7.4
  DESCRIPTION "A library for efficient similarity search and clustering of dense vectors."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
<<<<<<< HEAD
  LANGUAGES ${FAISS_LANGUAGES})
=======
  LANGUAGES CXX HIP)
>>>>>>> port_to_rocm/v1.7.3
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Valid values are "generic", "avx2".
option(FAISS_OPT_LEVEL "" "generic")
option(FAISS_ENABLE_GPU "Enable support for GPU indexes." ON)
<<<<<<< HEAD
option(FAISS_ENABLE_RAFT "Enable RAFT for GPU indexes." OFF)
option(FAISS_ENABLE_PYTHON "Build Python extension." ON)
=======
option(FAISS_ENABLE_PYTHON "Build Python extension." OFF)
>>>>>>> port_to_rocm/v1.7.3
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

<<<<<<< HEAD
if(FAISS_ENABLE_RAFT)
  rapids_cpm_init()
  include(cmake/thirdparty/get_raft.cmake)
  include(cmake/thirdparty/get_cutlass.cmake)
endif()

=======
# if debug compile, set the optimazation 0
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-O0)
endif()
>>>>>>> port_to_rocm/v1.7.3
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
