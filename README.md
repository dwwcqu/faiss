<<<<<<< HEAD
# Faiss

Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/numpy. Some of the most useful algorithms are implemented on the GPU. It is developed primarily at Meta's [Fundamental AI Research](https://ai.facebook.com/) group.

## News

See [CHANGELOG.md](CHANGELOG.md) for detailed information about latest features.

## Introduction

Faiss contains several methods for similarity search. It assumes that the instances are represented as vectors and are identified by an integer, and that the vectors can be compared with L2 (Euclidean) distances or dot products. Vectors that are similar to a query vector are those that have the lowest L2 distance or the highest dot product with the query vector. It also supports cosine similarity, since this is a dot product on normalized vectors.

Some of the methods, like those based on binary vectors and compact quantization codes, solely use a compressed representation of the vectors and do not require to keep the original vectors. This generally comes at the cost of a less precise search but these methods can scale to billions of vectors in main memory on a single server. Other methods, like HNSW and NSG add an indexing structure on top of the raw vectors to make searching more efficient.

The GPU implementation can accept input from either CPU or GPU memory. On a server with GPUs, the GPU indexes can be used a drop-in replacement for the CPU indexes (e.g., replace `IndexFlatL2` with `GpuIndexFlatL2`) and copies to/from GPU memory are handled automatically. Results will be faster however if both input and output remain resident on the GPU. Both single and multi-GPU usage is supported.

## Installing

Faiss comes with precompiled libraries for Anaconda in Python, see [faiss-cpu](https://anaconda.org/pytorch/faiss-cpu) and [faiss-gpu](https://anaconda.org/pytorch/faiss-gpu). The library is mostly implemented in C++, the only dependency is a [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) implementation. Optional GPU support is provided via CUDA, and the Python interface is also optional. It compiles with cmake. See [INSTALL.md](INSTALL.md) for details.

## How Faiss works

Faiss is built around an index type that stores a set of vectors, and provides a function to search in them with L2 and/or dot product vector comparison. Some index types are simple baselines, such as exact search. Most of the available indexing structures correspond to various trade-offs with respect to

- search time
- search quality
- memory used per index vector
- training time
- adding time
- need for external data for unsupervised training

The optional GPU implementation provides what is likely (as of March 2017) the fastest exact and approximate (compressed-domain) nearest neighbor search implementation for high-dimensional vectors, fastest Lloyd's k-means, and fastest small k-selection algorithm known. [The implementation is detailed here](https://arxiv.org/abs/1702.08734).

## Full documentation of Faiss

The following are entry points for documentation:

- the full documentation can be found on the [wiki page](http://github.com/facebookresearch/faiss/wiki), including a [tutorial](https://github.com/facebookresearch/faiss/wiki/Getting-started), a [FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ) and a [troubleshooting section](https://github.com/facebookresearch/faiss/wiki/Troubleshooting)
- the [doxygen documentation](https://faiss.ai/) gives per-class information extracted from code comments
- to reproduce results from our research papers, [Polysemous codes](https://arxiv.org/abs/1609.01882) and [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734), refer to the [benchmarks README](benchs/README.md). For [
Link and code: Fast indexing with graphs and compact regression codes](https://arxiv.org/abs/1804.09996), see the [link_and_code README](benchs/link_and_code)

## Authors

The main authors of Faiss are:
- [Hervé Jégou](https://github.com/jegou) initiated the Faiss project and wrote its first implementation
- [Matthijs Douze](https://github.com/mdouze) implemented most of the CPU Faiss
- [Jeff Johnson](https://github.com/wickedfoo) implemented all of the GPU Faiss
- [Lucas Hosseini](https://github.com/beauby) implemented the binary indexes and the build system
- [Chengqi Deng](https://github.com/KinglittleQ) implemented NSG, NNdescent and much of the additive quantization code.
- [Alexandr Guzhva](https://github.com/alexanderguzhva) many optimizations: SIMD, memory allocation and layout, fast decoding kernels for vector codecs, etc.

## Reference

Reference to cite when you use Faiss in a research paper:

=======
# faiss library ROCm GPU-backend
该 *faiss* 库 *fork* 自 [*facebook faiss*](https://github.com/facebookresearch/faiss)，本库主要是将其从 *CUDA GPU backend* 移植到 *ROCm GPU backend*。

目前，移植的 *faiss* 版本是 v1.7.3。
## Build
```bash
# /path/to/faiss
# Debug 版本
CXX=/path/to/your/hipcc cmake -DCMAKE_BUILD_TYPE=Debug -DFAISS_ENABLE_GPU=ON -B build .
make -C build -j
# /path/to/faiss
# Release 版本
CXX=/path/to/your/hipcc cmake -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=ON -B build .
make -C build -j
>>>>>>> port_to_rocm/v1.7.3
```

<<<<<<< HEAD
We monitor the [issues page](http://github.com/facebookresearch/faiss/issues) of the repository.
You can report bugs, ask questions, etc.

## Legal

Faiss is MIT-licensed, refer to the [LICENSE file](https://github.com/facebookresearch/faiss/blob/main/LICENSE) in the top level directory.

Copyright © Meta Platforms, Inc. See the [Terms of Use](https://opensource.fb.com/legal/terms/) and [Privacy Policy](https://opensource.fb.com/legal/privacy/) for this project.
=======
## Test
```bash
# /path/to/faiss
# Debug 版本
CXX=/path/to/your/hipcc cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON -DFAISS_ENABLE_GPU=ON -B build .
make -C build -j
make -C build test
# /path/to/faiss
# Release 版本
CXX=/path/to/your/hipcc cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DFAISS_ENABLE_GPU=ON -B build .
make -C build -j
make -C build test
```

## Install
```bash
# /path/to/faiss
# Debug 版本
CXX=/path/to/your/hipcc cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/path/to/install/ -DFAISS_ENABLE_GPU=ON -B build .
make -C build -j
make -C build install
# /path/to/faiss
# Release 版本
CXX=/path/to/your/hipcc cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/install/ -DFAISS_ENABLE_GPU=ON -B build .
make -C build -j
make -C build install
```
关于详细的源码编译、测试和安装，请参考[*Install.md*](./INSTALL.md)，关于 *faiss* 库的了解，请参考 [*FAISS_README.md*](./FAISS_README.md)。
## Port Summary
*faiss* 库的 GPU 加速部分，移植适配 ROCm-GPU 的主要内容汇总如下：

+ *k-select* 算法中，*warpSize* 大小的线程束内的双调排序部分实现，从 CUDA-backend 的 32 warp 大小调整为 ROCm-backend 的 64 warp 大小；且在进行合并时，需要进行大小为 32 长度的合并操作(在 CUDA-backend 是 16 大小的合并操作)；
+ 对于 *k-select* 算法中，一个 warp 中会有一个 warp 队列，当 `k<=32` 时，CUDA-backend 会设置 32 的队列大小，ROCm-backend 需要设置为 64 的队列大小。这里有一个例外是 WarpSelect 实现，其 `N_WARP_Q` 大小 32 是不影响的，而对于 BlockSelect, IVFInterleavedScan, IVFInterleavedScan2 等都需要将 `N_WARP_Q` 大小设置为 64；
+ 在关于 `IVFPQ` 实现部分，CUDA-backend 需要用一个 warpSize=32 去编码 32 个向量，对应到 ROCm-backend 下，就需要用一个 warpSize=64 去编码 64 个向量；
+ 最后一部分，就是 *Warp-coalesced parallel reading and writing of packed bits*，这里需要将 CUDA-backend 的 32 个 warp 线程去读写 4-bits, 5-bits, 6-bits 的操作，改为 ROCm-backend 下 64 个 warp 线程去读写 4-bits, 5-bits, 6-bits 的操作；
+ 更多具体的移植过程，可以参考 git 的 commit 记录对边，进行具体了解；

## Unsupport
目前，针对 *python* 接口部分，目前还未完全移植完毕，这是后续的移植任务。
>>>>>>> port_to_rocm/v1.7.3
