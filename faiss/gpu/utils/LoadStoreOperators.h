/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/Float16.h>

#ifndef __HALF2_TO_UI
// cuda_fp16.hpp doesn't export this
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
#endif

//
// Templated wrappers to express load/store for different scalar and vector
// types, so kernels can have the same written form but can operate
// over half and float, and on vector types transparently
//

namespace faiss {
namespace gpu {

template <typename T>
struct LoadStore {
    static inline __device__ T load(void* p) {
        return *((T*)p);
    }

    static inline __device__ void store(void* p, const T& v) {
        *((T*)p) = v;
    }
};

template <>
struct LoadStore<Half4> {
    static inline __device__ Half4 load(void* p) {
        Half4 out;
#ifdef __HIP_PLATFORM_NVIDIA__
    #if CUDA_VERSION >= 9000
        asm("ld.global.v2.u32 {%0, %1}, [%2];"
            : "=r"(__HALF2_TO_UI(out.a)), "=r"(__HALF2_TO_UI(out.b))
            : "l"(p));
    #else
        asm("ld.global.v2.u32 {%0, %1}, [%2];"
            : "=r"(out.a.x), "=r"(out.b.x)
            : "l"(p));
    #endif
#else
    uint16_t *ptr = reinterpret_cast<uint16_t*>(p);
    __half2_raw temp_a;
    __half2_raw temp_b;
    temp_a.x = *ptr; ++ptr;
    temp_a.y = *ptr; ++ptr;
    temp_b.x = *ptr; ++ptr;
    temp_b.y = *ptr;

    out.a = temp_a;
    out.b = temp_b;
#endif
    return out;
}

    static inline __device__ void store(void* p, Half4& v) {
#ifdef __HIP_PLATFORM_NVIDIA__
#if CUDA_VERSION >= 9000
        asm("st.v2.u32 [%0], {%1, %2};"
            :
            : "l"(p), "r"(__HALF2_TO_UI(v.a)), "r"(__HALF2_TO_UI(v.b)));
#else
        asm("st.v2.u32 [%0], {%1, %2};" : : "l"(p), "r"(v.a.x), "r"(v.b.x));
#endif
#else
    uint16_t *ptr = reinterpret_cast<uint16_t*>(p);
    __half2_raw temp_a = v.a;
    __half2_raw temp_b = v.b;
    *ptr = temp_a.x; ++ptr;
    *ptr = temp_a.y; ++ptr;
    *ptr = temp_b.x; ++ptr;
    *ptr = temp_b.y;
#endif
    }
};

template <>
struct LoadStore<Half8> {
    static inline __device__ Half8 load(void* p) {
        Half8 out;
#ifdef __HIP_PLATFORM_NVIDIA__
#if CUDA_VERSION >= 9000
        asm("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(__HALF2_TO_UI(out.a.a)),
              "=r"(__HALF2_TO_UI(out.a.b)),
              "=r"(__HALF2_TO_UI(out.b.a)),
              "=r"(__HALF2_TO_UI(out.b.b))
            : "l"(p));
#else
        asm("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(out.a.a.x), "=r"(out.a.b.x), "=r"(out.b.a.x), "=r"(out.b.b.x)
            : "l"(p));
#endif
#else
    uint16_t *ptr = reinterpret_cast<uint16_t*>(p);
    __half2_raw temp_a_a;
    __half2_raw temp_a_b;
    __half2_raw temp_b_a;
    __half2_raw temp_b_b;
    temp_a_a.x = *ptr; ++ptr;
    temp_a_a.y = *ptr; ++ptr;
    temp_a_b.x = *ptr; ++ptr;
    temp_a_b.y = *ptr; ++ptr;
    temp_b_a.x = *ptr; ++ptr;
    temp_b_a.y = *ptr; ++ptr;
    temp_b_b.x = *ptr; ++ptr;
    temp_b_b.y = *ptr;

    out.a.a = temp_a_a;
    out.a.b = temp_a_b;
    out.b.a = temp_b_a;
    out.b.b = temp_b_b;
#endif
        return out;
    }

    static inline __device__ void store(void* p, Half8& v) {
#ifdef __HIP_PLATFORM_NVIDIA__
#if CUDA_VERSION >= 9000
        asm("st.v4.u32 [%0], {%1, %2, %3, %4};"
            :
            : "l"(p),
              "r"(__HALF2_TO_UI(v.a.a)),
              "r"(__HALF2_TO_UI(v.a.b)),
              "r"(__HALF2_TO_UI(v.b.a)),
              "r"(__HALF2_TO_UI(v.b.b)));
#else
        asm("st.v4.u32 [%0], {%1, %2, %3, %4};"
            :
            : "l"(p), "r"(v.a.a.x), "r"(v.a.b.x), "r"(v.b.a.x), "r"(v.b.b.x));
#endif
#else
    uint16_t *ptr = reinterpret_cast<uint16_t*>(p);
    __half2_raw temp_a_a = v.a.a;
    __half2_raw temp_a_b = v.a.b;
    __half2_raw temp_b_a = v.b.a;
    __half2_raw temp_b_b = v.b.b;
    *ptr = temp_a_a.x; ++ptr;
    *ptr = temp_a_a.y; ++ptr;
    *ptr = temp_a_b.x; ++ptr;
    *ptr = temp_a_b.y; ++ptr;
    *ptr = temp_b_a.x; ++ptr;
    *ptr = temp_b_a.y; ++ptr;
    *ptr = temp_b_b.x; ++ptr;
    *ptr = temp_b_b.y;
#endif
    }
};

} // namespace gpu
} // namespace faiss
