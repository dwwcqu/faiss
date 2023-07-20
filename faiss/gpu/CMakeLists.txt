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

set(FAISS_GPU_SRC
  GpuAutoTune.cpp
  GpuCloner.cpp
<<<<<<< HEAD
  GpuDistance.cu
  GpuIcmEncoder.cu
  GpuIndex.cu
  GpuIndexBinaryFlat.cu
  GpuIndexFlat.cu
  GpuIndexIVF.cu
  GpuIndexIVFFlat.cu
  GpuIndexIVFPQ.cu
  GpuIndexIVFScalarQuantizer.cu
=======
  GpuClonerOptions.cpp
  GpuDistance.cpp
  GpuIcmEncoder.cpp
  GpuIndex.cpp
  GpuIndexBinaryFlat.cpp
  GpuIndexFlat.cpp
  GpuIndexIVF.cpp
  GpuIndexIVFFlat.cpp
  GpuIndexIVFPQ.cpp
  GpuIndexIVFScalarQuantizer.cpp
>>>>>>> port_to_rocm/v1.7.3
  GpuResources.cpp
  StandardGpuResources.cpp
  impl/BinaryDistance.cpp
  impl/BinaryFlatIndex.cpp
  impl/BroadcastSum.cpp
  impl/Distance.cpp
  impl/FlatIndex.cpp
  impl/IndexUtils.cpp
  impl/IVFAppend.cpp
  impl/IVFBase.cpp
  impl/IVFFlat.cpp
  impl/IVFFlatScan.cpp
  impl/IVFInterleaved.cpp
  impl/IVFPQ.cpp
  impl/IVFUtils.cpp
  impl/IVFUtilsSelect1.cpp
  impl/IVFUtilsSelect2.cpp
  impl/InterleavedCodes.cpp
  impl/L2Norm.cpp
  impl/L2Select.cpp
  impl/PQScanMultiPassPrecomputed.cpp
  impl/RemapIndices.cpp
  impl/VectorResidual.cpp
  impl/scan/IVFInterleaved1.cpp
  # impl/scan/IVFInterleaved32.cpp
  impl/scan/IVFInterleaved64.cpp
  impl/scan/IVFInterleaved128.cpp
  impl/scan/IVFInterleaved256.cpp
  impl/scan/IVFInterleaved512.cpp
  impl/scan/IVFInterleaved1024.cpp
  impl/scan/IVFInterleaved2048.cpp
  impl/IcmEncoder.cpp
  utils/BlockSelectFloat.cpp
  utils/DeviceUtils.cpp
  utils/StackDeviceMemory.cpp
  utils/Timer.cpp
  utils/WarpSelectFloat.cpp
  utils/blockselect/BlockSelectFloat1.cpp
  # utils/blockselect/BlockSelectFloat32.cpp
  utils/blockselect/BlockSelectFloat64.cpp
  utils/blockselect/BlockSelectFloat128.cpp
  utils/blockselect/BlockSelectFloat256.cpp
  utils/blockselect/BlockSelectFloatF512.cpp
  utils/blockselect/BlockSelectFloatF1024.cpp
  utils/blockselect/BlockSelectFloatF2048.cpp
  utils/blockselect/BlockSelectFloatT512.cpp
  utils/blockselect/BlockSelectFloatT1024.cpp
  utils/blockselect/BlockSelectFloatT2048.cpp
  utils/warpselect/WarpSelectFloat1.cpp
  utils/warpselect/WarpSelectFloat32.cpp
  utils/warpselect/WarpSelectFloat64.cpp
  utils/warpselect/WarpSelectFloat128.cpp
  utils/warpselect/WarpSelectFloat256.cpp
  utils/warpselect/WarpSelectFloatF512.cpp
  utils/warpselect/WarpSelectFloatF1024.cpp
  utils/warpselect/WarpSelectFloatF2048.cpp
  utils/warpselect/WarpSelectFloatT512.cpp
  utils/warpselect/WarpSelectFloatT1024.cpp
  utils/warpselect/WarpSelectFloatT2048.cpp
)

set(FAISS_GPU_HEADERS
  GpuAutoTune.h
  GpuCloner.h
  GpuClonerOptions.h
  GpuDistance.h
  GpuIcmEncoder.h
  GpuFaissAssert.h
  GpuIndex.h
  GpuIndexBinaryFlat.h
  GpuIndexFlat.h
  GpuIndexIVF.h
  GpuIndexIVFFlat.h
  GpuIndexIVFPQ.h
  GpuIndexIVFScalarQuantizer.h
  GpuIndicesOptions.h
  GpuResources.h
  StandardGpuResources.h
  impl/BinaryDistance.h
  impl/BinaryFlatIndex.h
  impl/BroadcastSum.h
  impl/Distance.h
  impl/DistanceUtils.h
  impl/FlatIndex.h
  impl/GeneralDistance.h
  impl/GpuScalarQuantizer.h
  impl/IndexUtils.h
  impl/IVFAppend.h
  impl/IVFBase.h
  impl/IVFFlat.h
  impl/IVFFlatScan.h
  impl/IVFInterleaved.h
  impl/IVFPQ.h
  impl/IVFUtils.h
  impl/InterleavedCodes.h
  impl/L2Norm.h
  impl/L2Select.h
  impl/PQCodeDistances-inl.h
  impl/PQCodeDistances.h
  impl/PQCodeLoad.h
  impl/PQScanMultiPassNoPrecomputed-inl.h
  impl/PQScanMultiPassNoPrecomputed.h
  impl/PQScanMultiPassPrecomputed.h
  impl/RemapIndices.h
  impl/VectorResidual.h
  impl/scan/IVFInterleavedImpl.h
  impl/IcmEncoder.h
  utils/BlockSelectKernel.h
  utils/Comparators.h
  utils/ConversionOperators.h
  utils/CopyUtils.h
  utils/DeviceDefs.h
  utils/DeviceTensor-inl.h
  utils/DeviceTensor.h
  utils/DeviceUtils.h
  utils/DeviceVector.h
  utils/Float16.h
  utils/HostTensor-inl.h
  utils/HostTensor.h
  utils/Limits.h
  utils/LoadStoreOperators.h
  utils/MathOperators.h
  utils/MatrixMult-inl.h
  utils/MatrixMult.h
  utils/MergeNetworkBlock.h
  utils/MergeNetworkUtils.h
  utils/MergeNetworkWarp.h
  utils/NoTypeTensor.h
  utils/Pair.h
  utils/PtxUtils.h
  utils/ReductionOperators.h
  utils/Reductions.h
  utils/Select.h
  utils/StackDeviceMemory.h
  utils/StaticUtils.h
  utils/Tensor-inl.h
  utils/Tensor.h
  utils/ThrustUtils.h
  utils/Timer.h
  utils/Transpose.h
  utils/WarpPackedBits.h
  utils/WarpSelectKernel.h
  utils/WarpShuffles.h
  utils/blockselect/BlockSelectImpl.h
  utils/warpselect/WarpSelectImpl.h
)

if(FAISS_ENABLE_RAFT)
  list(APPEND FAISS_GPU_HEADERS
          impl/RaftFlatIndex.cuh)
  list(APPEND FAISS_GPU_SRC
          impl/RaftFlatIndex.cu)

  target_compile_definitions(faiss PUBLIC USE_NVIDIA_RAFT=1)
  target_compile_definitions(faiss_avx2 PUBLIC USE_NVIDIA_RAFT=1)
endif()

# Export FAISS_GPU_HEADERS variable to parent scope.
set(FAISS_GPU_HEADERS ${FAISS_GPU_HEADERS} PARENT_SCOPE)

target_sources(faiss PRIVATE ${FAISS_GPU_SRC})
target_sources(faiss_avx2 PRIVATE ${FAISS_GPU_SRC})

foreach(header ${FAISS_GPU_HEADERS})
  get_filename_component(dir ${header} DIRECTORY )
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/gpu/${dir}
  )
endforeach()
<<<<<<< HEAD

find_package(CUDAToolkit REQUIRED)
target_link_libraries(faiss PRIVATE CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_RAFT}>:raft::raft> $<$<BOOL:${FAISS_ENABLE_RAFT}>:nvidia::cutlass::cutlass>)
target_link_libraries(faiss_avx2 PRIVATE CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_RAFT}>:raft::raft> $<$<BOOL:${FAISS_ENABLE_RAFT}>:nvidia::cutlass::cutlass>)
target_compile_options(faiss PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all --expt-extended-lambda --expt-relaxed-constexpr>)
target_compile_options(faiss_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all --expt-extended-lambda --expt-relaxed-constexpr>)
=======
find_package(hip REQUIRED)
find_package(rocthrust REQUIRED)
find_package(hipblas REQUIRED)
find_package(hiprand REQUIRED)
find_package(rocrand REQUIRED)
target_link_libraries(faiss PRIVATE hip::host hip::device roc::rocthrust roc::hipblas hip::hiprand roc::rocrand)
target_link_libraries(faiss_avx2 PRIVATE hip::host hip::device roc::rocthrust roc::hipblas hip::hiprand roc::rocrand)
# target_compile_options(faiss PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all>)
# target_compile_options(faiss_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all>)
>>>>>>> port_to_rocm/v1.7.3
