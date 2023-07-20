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
```

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