// Host stand-ins for the CUDA names mip_optionb.h uses, so tests/test_optionb_header.py can
// compile it as plain C++ (float32 arithmetic, denormals on, as nvcc without fast math).
#pragma once
#include <cmath>
#include <algorithm>
using std::fmaxf; using std::fminf;
#define __device__
#define __forceinline__ inline
#define __constant__ static
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
inline float3 transformPoint4x3(const float3& p, const float* m)
{ return { m[0]*p.x+m[4]*p.y+m[8]*p.z+m[12], m[1]*p.x+m[5]*p.y+m[9]*p.z+m[13], m[2]*p.x+m[6]*p.y+m[10]*p.z+m[14] }; }
