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
 
#include "gather_new_plugin.h"
#include <cuda.h>

using namespace nvinfer1;

PluginFieldCollection GatherNewPluginCreator::fc_{};
std::vector<PluginField> GatherNewPluginCreator::attr_;

__global__ void GatherNewKernel(const float* data,const int* indices, int need_data_dims, float* output)
{ 
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    // axis = 0
    
    int x1 = blockIdx.x/need_data_dims;
    int data_axis_index = indices[x1];
    int x2 = blockIdx.x%need_data_dims;
    int ret = blockIdx.x%(blockIdx.x/need_data_dims);
    int data_index = data_axis_index * blockDim.x * need_data_dims + x2 * blockDim.x + threadIdx.x;
    
    // check 
    // if (blockIdx.x == 1 && threadIdx.x == 1)
    // {
    //     printf("axis index %d",data_axis_index);
    //     printf("data index %d", data_index);
    //     printf("data %f", data[data_index]);
    // }
    output[tx] = data[data_index];

}

int32_t GatherNewPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int blocks = 1;
    for (int i = 0; i < inputDesc[1].dims.nbDims; i ++)
    {
        blocks *= inputDesc[1].dims.d[i];
        // printf("i1 = %d dim = %d", i, inputDesc[1].dims.d[i]);
    }

    int need_data_dims = 1;
    for (int i = 1; i < inputDesc[0].dims.nbDims - 1; i ++)
    {
        need_data_dims *= inputDesc[0].dims.d[i];
        
    }
    blocks *= need_data_dims;
    
    // last dim is always last dim
    int threads = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];


    // printf("blocks = %d, threads = %d， need_data_dims=%d", blocks, threads, need_data_dims);
    const float* data = (const float*)inputs[0];
    const int* indices = (const int*)inputs[1];
    float* out = (float*)outputs[0];
    GatherNewKernel<<<blocks, threads, 0, stream>>>(data, indices, need_data_dims, out);

    return 0;
}

REGISTER_TENSORRT_PLUGIN(GatherNewPluginCreator);



