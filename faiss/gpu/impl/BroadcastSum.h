/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#ifdef __HIP_PALTFORM_NVIDIA__
#else
#include <hip/hip_fp16.h>
#endif
#include <faiss/gpu/utils/Tensor.h>

namespace faiss {
namespace gpu {

// output[x][i] += input[i] for all x
void runSumAlongColumns(
        Tensor<float, 1, true>& input,
        Tensor<float, 2, true>& output,
        hipStream_t stream);

void runSumAlongColumns(
        Tensor<half, 1, true>& input,
        Tensor<half, 2, true>& output,
        hipStream_t stream);

// output[x][i] = input[i] for all x
void runAssignAlongColumns(
        Tensor<float, 1, true>& input,
        Tensor<float, 2, true>& output,
        hipStream_t stream);

void runAssignAlongColumns(
        Tensor<half, 1, true>& input,
        Tensor<half, 2, true>& output,
        hipStream_t stream);

// output[i][x] += input[i] for all x
// If zeroClamp, output[i][x] = max(output[i][x] + input[i], 0) for all x
void runSumAlongRows(
        Tensor<float, 1, true>& input,
        Tensor<float, 2, true>& output,
        bool zeroClamp,
        hipStream_t stream);

void runSumAlongRows(
        Tensor<half, 1, true>& input,
        Tensor<half, 2, true>& output,
        bool zeroClamp,
        hipStream_t stream);

} // namespace gpu
} // namespace faiss
