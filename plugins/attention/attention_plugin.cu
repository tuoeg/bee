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
 
#include "attention_plugin.h"
#include "cublas_v2.h"
#include <cuda.h>
#include <assert.h>
#include "helper_cuda.h"

using namespace nvinfer1;

PluginFieldCollection AttentionPluginCreator::fc_{};
std::vector<PluginField> AttentionPluginCreator::attr_;


template<typename T,int TPB,int VPT>
__global__ void AttentionKernel(float *x1,float *x2,float *bias, float *output)
{ 
    

}

int32_t AttentionPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    
    cudaError_t cuda_err;
    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    // 3D covert 2d
    assert(inputDesc[0].dims.d[2] == inputDesc[1].dims.d[0] && "k must be the same\n");
    int n = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    int m = inputDesc[1].dims.d[1];
    int k = inputDesc[0].dims.d[2];
    printf("m = %d, n = %d, k = %d\n",m, n, k);

    const float alpha = 1, beta = 0;
    // cudalas column major
    const void *A = inputs[1];
    const void *B = inputs[0];
    void *C = outputs[0];

    // should not printf val in device memory 
    //printf("a0 = %f\n", ((float*)A)[0]);
    //printf("b0 = %f\n", ((float*)B)[0]);
    checkCudaErrors(cublasGemmEx(handle,
                 CUBLAS_OP_N, 
                 CUBLAS_OP_N,
                 m,
                 n,
                 k,
                 &alpha,
                 A,
                 CUDA_R_32F,
                 m,
                 B,
                 CUDA_R_32F,
                 k,
                 &beta,
                 C,
                 CUDA_R_32F,
                 m,
                 CUBLAS_COMPUTE_32F,                                   
                 CUBLAS_GEMM_DEFAULT));
    
    printf("end");
    cublasDestroy(handle);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(AttentionPluginCreator);



