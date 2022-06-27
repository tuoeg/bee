/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#include "layer_norm_plugin.h"
#include <cub/cub.cuh>

using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

// __global__ void layerNormKernel(float *pInput,float *w,float *b, float *pOutput)
// {
//     const int tx = threadIdx.x, index = blockIdx.x * 256 + threadIdx.x;

//     __shared__ float temp[128];

//     float value0 = pInput[index];
//     float value1 = pInput[index + 128];
//     temp[tx] = value0 + value1;
//     __syncthreads();

//     for (int stride = 64; stride >= 1; stride /= 2)
//     {
//         if (tx < stride)
//         {
//             temp[tx] += temp[tx + stride];
//         }
//         __syncthreads();
//     }
//     float mean = temp[0] / 256;
//     __syncthreads();

//     temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean);
//     __syncthreads();

//     for (int stride = 64; stride >= 1; stride /= 2)
//     {
//         if (tx < stride)
//         {
//             temp[tx] += temp[tx + stride];
//         }
//         __syncthreads();
//     }
//     float var = temp[0] / 256;

//     pOutput[index]       = (value0 - mean) * rsqrtf(var + 6e-6)*w[tx]+b[tx];
//     pOutput[index + 128] = (value1 - mean) * rsqrtf(var + 6e-6)*w[tx+128]+b[tx+128];

// }

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};
// typedef struct{
//     float a;
//     float b;
//     float c; 
// }float3;
/*
template<typename T>
struct Sum
{	
__host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const
{return a + b;}
};

template<>
struct Sum<float2>
{
__host__ __device__ __forceinline__ float2 operator()(const float2 & a, const float2 & b) const
{
        return {a.x + b.x, a.y + b.y};
}
};
*/
template <>
__host__ __device__ __forceinline__ float2 cub::Sum::operator()(const float2 &a, const float2 &b) const
{
	//float2 res;
	//res.x = a.x + b.x;
	//res.y = a.y + b.y;
	return make_float2(a.x + b.x, a.y + b.y);
	//return res;
}
template<>
struct BytesToType<12>
{
    using type = float3;
};
template<int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

template<typename T,int TPB,int VPT>
__global__ void layerNormKernel(float *input,float *gama,float *beta, float *output)
{ 
    const int idx=blockIdx.x*TPB*VPT + threadIdx.x*VPT;
    T localX[VPT],localGama[VPT],localBeta[VPT];
    copy<sizeof(T)*VPT>(&input[idx],localX);
    float2 localFloat2 ={0.f,0.f};
    const float rld=float(1)/float(768);
#pragma unroll
    for(int it=0;it<VPT;it++)
    {
        const float tmp=rld * (float)localX[it];
        localFloat2.x+=tmp;
        localFloat2.y+=tmp*(float)localX[it];
    }
   
    copy<sizeof(T)*VPT>(&beta[threadIdx.x*VPT],localBeta);
    copy<sizeof(T)*VPT>(&gama[threadIdx.x*VPT],localGama);
    using BlockReduce =cub::BlockReduce<float2,TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float mu;
    __shared__ float rsigma;
    //const float2 sumKV=BlockReduce(temp_storage).Reduce(localFloat2,cub::Sum());
    const float2 sumKV=BlockReduce(temp_storage).Reduce(localFloat2, cub::Sum());
   
    if(threadIdx.x==0)
    {
        //const float rld=float(1)/float(768);
	mu=sumKV.x;
        rsigma=rsqrt(sumKV.y - mu*mu + 1e-5);
    }
    __syncthreads();
#pragma unroll
    for(int it=0;it<VPT;it++)
    {
        localX[it]=(float)localGama[it]*((float)localX[it]-mu)*rsigma+(float)localBeta[it];
    }
    copy<sizeof(T)*VPT>(localX,&output[idx]);
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    (layerNormKernel<float,384,2>)<<<nBlock, 384, 0, stream>>>((float *)inputs[0],(float *)inputs[1],(float *)inputs[2], (float *)outputs[0]);
    //layerNormKernel<<<nBlock, 128, 0, stream>>>((float *)inputs[0],(float *)inputs[1],(float *)inputs[2], (float *)outputs[0]);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);



