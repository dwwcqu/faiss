/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/Tensor.h>
#ifdef __HIP_PLATFORM_AMD__
#include <hip/amd_detail/amd_hip_fp16.h>
#endif
namespace faiss {
namespace gpu {

void runL2Norm(
        Tensor<float, 2, true>& input,
        bool inputRowMajor,
        Tensor<float, 1, true>& output,
        bool normSquared,
        hipStream_t stream);

void runL2Norm(
        Tensor<half, 2, true>& input,
        bool inputRowMajor,
        Tensor<float, 1, true>& output,
        bool normSquared,
        hipStream_t stream);

} // namespace gpu
} // namespace faiss
