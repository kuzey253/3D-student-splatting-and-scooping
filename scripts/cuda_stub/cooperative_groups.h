#pragma once
namespace cooperative_groups {
struct thread_block { __device__ void sync(); __device__ dim3 group_index(); __device__ dim3 thread_index(); __device__ unsigned thread_rank(); __device__ unsigned size(); };
struct grid_group { __device__ unsigned long long thread_rank(); __device__ unsigned long long size(); };
__device__ thread_block this_thread_block(); __device__ grid_group this_grid();
}
