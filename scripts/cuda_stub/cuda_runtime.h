// Minimal CUDA stand-ins for scripts/cuda_syntax_check.sh: lets clang's CUDA front end parse the
// rasterizer sources on a machine without the CUDA toolkit. Declarations only; nothing links.
#pragma once
#define CUDA_VERSION 12000
#define __CUDACC_VER_MAJOR__ 12
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>
#define CUDART_INF_F __builtin_huge_valf()
struct float2 { float x, y; }; struct float3 { float x, y, z; }; struct float4 { float x, y, z, w; };
struct int2 { int x, y; }; struct uint2 { unsigned x, y; }; struct uint3 { unsigned x, y, z; };
struct dim3 { unsigned x, y, z; __host__ __device__ dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
__host__ __device__ inline float2 make_float2(float a, float b) { return { a, b }; }
__host__ __device__ inline float3 make_float3(float a, float b, float c) { return { a, b, c }; }
__host__ __device__ inline float4 make_float4(float a, float b, float c, float d) { return { a, b, c, d }; }
__host__ __device__ inline int2 make_int2(int a, int b) { return { a, b }; }
__host__ __device__ inline uint2 make_uint2(unsigned a, unsigned b) { return { a, b }; }
typedef int cudaError_t; typedef void* cudaStream_t;
enum cudaMemcpyKind { cudaMemcpyDeviceToHost };
__host__ cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
__host__ cudaError_t cudaGetLastError();
__host__ cudaError_t cudaDeviceSynchronize();
__host__ const char* cudaGetErrorString(cudaError_t);
__device__ float atomicAdd(float*, float); __device__ int atomicAdd(int*, int);
__device__ void __syncthreads(); __device__ int __syncthreads_count(int);
__device__ float __expf(float); __device__ float __logf(float); __device__ float __powf(float, float);
__device__ float __fdividef(float, float); __device__ float rsqrtf(float);
#include "__clang_cuda_builtin_vars.h"
template <class A, class B> __host__ __device__ inline auto min(A a, B b) { return a < b ? a : b; }
template <class A, class B> __host__ __device__ inline auto max(A a, B b) { return a > b ? a : b; }
__device__ float fmaxf(float, float); __device__ float fminf(float, float);
__device__ float expf(float); __device__ float logf(float); __device__ float sqrtf(float); __device__ float powf(float, float);
__device__ float fabsf(float); __device__ float acosf(float); __device__ float cosf(float); __device__ float ceilf(float);
__device__ float floorf(float); __device__ float log1pf(float); __device__ int isfinite(float);
__device__ float sqrt(float); __device__ double sqrt(double); __device__ float exp(float); __device__ float log(float);
__device__ float pow(float, float);
extern "C" int printf(const char*, ...); __device__ void __trap();
__device__ float max(float, float); __device__ float min(float, float);
extern "C" unsigned __cudaPushCallConfiguration(dim3, dim3, size_t = 0, void* = 0);
__device__ float floor(float); __device__ float ceil(float); __host__ float floor(float); __host__ float ceil(float);
#define cudaSuccess 0
__host__ cudaError_t cudaMemset(void*, int, size_t);
namespace cub {
struct DeviceScan {
	template <class... A> static cudaError_t InclusiveSum(A...);
	template <class... A> static cudaError_t ExclusiveSum(A...);
};
struct DeviceRadixSort {
	template <class... A> static cudaError_t SortPairs(A...);
};
}
