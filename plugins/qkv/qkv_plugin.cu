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
 
#include "qkv_plugin.h"
#include "cublas_v2.h"
#include <cuda.h>
#include <assert.h>
#include "helper_cuda.h"

using namespace nvinfer1;

PluginFieldCollection QKVPluginCreator::fc_{};
std::vector<PluginField> QKVPluginCreator::attr_;

__global__ void QKVKernel(const float* bias_param,const int* trans_param, int dim_a,int dim_b,int dim_c,int dim_d,float* output)
{ 
    // change 64 * 12 to 12 * 64
    const int tx = blockIdx.x  * blockDim.x + threadIdx.x;
    int i0 = threadIdx.x / 64;
    int i1 = threadIdx.x - 64 * i0;

    int new_tid = i1 * 12 + i0;
    const int tmp = output[blockIdx.x  * blockDim.x + new_tid];
    __syncthreads();
    output[tx] = tmp;
    // add bias
    
    /*
    int _3_index = tx / (dim_a * dim_b * dim_c * dim_d);
    output[tx] += bias_parame[tx];
    __syncthreads();
    // out now is 3 * 6 * 709 * 12 * 64
    // a = 6, b = 709, c = 12, d = 64. when transpose, a * b * c * d will be (for exapmle) b * a * d *c .
    //  tx = index_a * b * c * d + index_b * c * d + index c * d + index_d
    // out [tx] = out [index_b * a * d * c + index_a * d * c + index_d * c + index_c] 
    
    int _6_index = tx / (dim_b * dim_c * dim_d) - _3_index * dim_a;
    int _709_index = tx / (dim_c * dim_d) - _3_index * dim_a * dim_b - _6_index * dim_b;
    int _12_index = tx / (dim_d) - _3_index * dim_a * dim_b * dim_c - _6_index * dim_b * dim_c - _709_index * dim_c;
    int _64_index = tx - _3_index * dim_a * dim_b * dim_c * dim_d - _6_index * dim_b * dim_c * dim_d - _709_index * dim_c * dim_d - _12_index * dim_d;

    
    int dim_l[4] = {dim_a, dim_b, dim_c, dim_d};
    int a_dim_index = trans_param[_3_index * 4];
    int b_dim_index = trans_param[_3_index * 4 + 1];
    int c_dim_index = trans_param[_3_index * 4 + 2];
    int d_dim_index = trans_param[_3_index * 4 + 3];

    
    int a = dim_l[a_dim_index];
    int b = dim_l[b_dim_index];
    int c = dim_l[c_dim_index];
    int d = dim_l[d_dim_index];

    int _6_index = tx / (b * c * d) - _3_index * a;
    int _709_index = tx / (c * d) - _3_index * a * b - _6_index * b;
    int _12_index = tx / (d) - _3_index * a * b * c - _6_index * b * c - _709_index * c;
    int _64_index = tx - _3_index * a * b * c * d - _6_index * b * c * d - _709_index * c * d - _12_index * d;


    int index_l[4] = {_6_index, _709_index, _12_index, _64_index};
    int a_index = index_l[a_dim_index];
    int b_index = index_l[b_dim_index];
    int c_index = index_l[c_dim_index];
    int d_index = index_l[d_dim_index];

    int index = _3_index * a * b * c *d  + a_index * b * c * d + b_index * c * d + c_index * d + d_index;
    //printf("index = %d\n", index);
    float tmp = output[index];
    __syncthreads();
    output[tx] = tmp;
    */
}

int32_t QKVPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // input 0 = data (3, 6, 709, 768)
    // input 1 = matmal_param (3, 768, 768)
    // input 2 = bias_param (3, 6, 709, 768)
    // input 3 = transpose_param (3, 4)
    // out 0 = (3, 6, 709, 12, 64)
    
    printf("start\n");
    cudaError_t cuda_err;
    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    int m = inputDesc[1].dims.d[2];
    int n = inputDesc[0].dims.d[1] * inputDesc[0].dims.d[2];
    int k = inputDesc[1].dims.d[1];

    const float alpha = 1, beta = 0;
    const void* A = inputs[1];
    const void* B = inputs[0];
    void *C = outputs[0];

    long long int stride_a = 768 * 768;

    // long long int stride_a = 6;


    long long int stride_b = 6 * 709 * 768;
    // long long int stride_b = 12;
    long long int stride_c = 6 * 709 * 768;

    // long long int stride_c = 8;
    int batch_count = inputDesc[0].dims.d[0];

    // stride batch 的 768的那一维变成了 709 * 6 * 64 * 12
    checkCudaErrors(cublasGemmStridedBatchedEx(handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    m, n, k,
                                    &alpha,
                                    A,
                                    CUDA_R_32F,
                                    m,
                                    stride_a,
                                    B,
                                    CUDA_R_32F,
                                    k,
                                    stride_b,
                                    &beta,
                                    C,
                                    CUDA_R_32F,
                                    m,
                                    stride_c,
                                    batch_count,
                                    CUBLAS_COMPUTE_32F,
                                    CUBLAS_GEMM_DEFAULT));
    printf("gemmbatch end\n");
    int dim_a = 6, dim_b = 709, dim_c = 12 , dim_d = 64;
    int blocks = 3*dim_a*dim_b;
    int threads = dim_c*dim_d;

    const float* bias_param = (const float*)inputs[2];
    const int* trans_param = (const int*)inputs[3];

    float* output = (float*)outputs[0];
    printf("qkv start \n");
    printf("bias shape = (%d, %d, %d, %d)\n", inputDesc[2].dims.d[0], inputDesc[2].dims.d[1], inputDesc[2].dims.d[2], inputDesc[2].dims.d[3]);
    printf("trans shape = (%d, %d)\n", inputDesc[3].dims.d[0], inputDesc[3].dims.d[1]);

    QKVKernel<<<blocks, threads, 0, stream>>>(bias_param, trans_param, dim_a, dim_b, dim_c, dim_d, output);

    return 0;
}

REGISTER_TENSORRT_PLUGIN(QKVPluginCreator);



