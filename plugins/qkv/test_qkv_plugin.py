#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
import ctypes
import numpy as np
from cuda import cudart  # 使用 cuda runtime API
import tensorrt as trt
import random
soFilePath      = './QKV.so'
nBS             = 6
nSL             = 709
nEmbedding      = 768
epsilon         = 6e-6

np.random.seed(97)

def check(a, b, weak = False):
    print(a.sum(), b.sum())
    if weak:
        return np.all( np.abs(a - b) < epsilon)
    else:
        return np.all( a == b )

# batch matmul biad transpose
def qkv_cpu(buffer_h):
    
    # 3 inputs for qkv
    _data = buffer_h[0]
    q_i, k_i, v_i = np.vsplit(_data, 3)

    # 3 params for matmul
    _matmul_params = buffer_h[1]
    q_p, k_p, v_p = np.vsplit(_matmul_params, 3)

    # 3 bias params for bias
    _bias_params = buffer_h[2]
    q_bp, k_bp, v_bp = np.vsplit(_bias_params, 3)

    # 3 transpose params for transpose
    _transpose_params = buffer_h[3]
    print(_transpose_params)
    q_tp, k_tp, v_tp = np.vsplit(_transpose_params, 3)
    print(q_i.shape, q_p.shape, q_bp.shape, q_tp.shape)

    # q = np.transpose((q_i.reshape(-1, 768).dot(q_p.reshape(768, -1))).reshape(1, 6, 709, 768) + q_bp, tuple(q_tp[0])).reshape(1, 6, 709, 12, 64)
    # k = np.transpose((k_i.reshape(-1, 768).dot(k_p.reshape(768, -1))).reshape(1, 6, 709, 768) + k_bp, tuple(k_tp[0])).reshape(1, 6, 709, 12, 64)
    # v = np.transpose((v_i.reshape(-1, 768).dot(v_p.reshape(768, -1))).reshape(1, 6, 709, 768) + v_bp, tuple(v_tp[0])).reshape(1, 6, 709, 12, 64)
    q = np.transpose((q_i.reshape(-1, 768).dot(q_p.reshape(768, -1))).reshape(1, 6, 709, 768)).reshape(1, 6, 709, 12, 64)
    k = np.transpose((k_i.reshape(-1, 768).dot(k_p.reshape(768, -1))).reshape(1, 6, 709, 768)).reshape(1, 6, 709, 12, 64)
    v = np.transpose((v_i.reshape(-1, 768).dot(v_p.reshape(768, -1))).reshape(1, 6, 709, 768)).reshape(1, 6, 709, 12, 64)

    # q = np.transpose((q_i.reshape(-1, 3).dot(q_p.reshape(3, -1))).reshape(1,2,2,2)).reshape(1,2,2,2)
    # k = np.transpose((k_i.reshape(-1, 3).dot(k_p.reshape(3, -1))).reshape(1,2,2,2)).reshape(1,2,2,2)
    # v = np.transpose((v_i.reshape(-1, 3).dot(v_p.reshape(3, -1))).reshape(1,2,2,2)).reshape(1,2,2,2)

    print(q.shape, k.shape, v.shape)

    _total = np.concatenate((q, k, v))
    return _total

def getQKVPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'QKV':
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def run():
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    builder         = trt.Builder(logger)
    network         = builder.create_network(1<<0)
    config          = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
    config.flags    = 0

    inputTensorList = []
    inputTensorList.append( network.add_input('data', trt.float32, [3, -1, nSL,nEmbedding]) )
    # inputTensorList.append( network.add_input('data', trt.float32, [3, -1, 2,3]) )

    inputTensorList.append( network.add_input('matmul_p', trt.float32, [3, nEmbedding, nEmbedding]) )
    # inputTensorList.append( network.add_input('matmul_p', trt.float32, [3, 3, 2]) )
    
    inputTensorList.append( network.add_input('bias_p', trt.float32, [3, -1, nSL, nEmbedding]) )
    inputTensorList.append( network.add_input('transpose_p', trt.int32, [3, 4]) )


    profile = builder.create_optimization_profile()
    profile.set_shape('data',[3, 1,nSL,nEmbedding],[3, 4, nSL,nEmbedding],[3, 6, nSL,nEmbedding])
    # profile.set_shape('data',[3, 1,2,3],[3, 2, 2,3],[3, 3, 2,3])

    profile.set_shape('bias_p',[3, 1, nSL, nEmbedding],[3, 4, nSL, nEmbedding],[3, 6, nSL, nEmbedding])
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getQKVPlugin())

    network.mark_output(pluginLayer.get_output(0))

    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0,[3, nBS, nSL, nEmbedding])
    context.set_binding_shape(2,[3, nBS, nSL, nEmbedding])
    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))

    bufferH = []
    # data
    bufferH.append( np.random.rand(3, nBS,nSL,nEmbedding).astype(np.float32).reshape(3, nBS,nSL,nEmbedding))
    # bufferH.append( np.random.randint(0, 10, size = (3, 2,2,3)).astype(np.float32).reshape(3, 2,2,3))


    # matmul_p
    bufferH.append( np.random.rand(3, nEmbedding,nEmbedding).astype(np.float32).reshape(3, nEmbedding,nEmbedding))
    # bufferH.append( np.random.randint(0, 10, size = (3, 3,2)).astype(np.float32).reshape(3, 3,2))

    # bias_p
    bufferH.append( np.random.rand(3, nBS, nSL, nEmbedding).astype(np.float32).reshape(3, nBS, nSL, nEmbedding))

    # transpose_p
    # generator random 0,1,2,3
    bufferH.append( np.vstack(tuple([sorted(iter(np.arange(4)),key= lambda k: random.random())for i in range(3)])))
    
    #output
    bufferH.append(np.empty((3, nBS, nSL, 12, 64),dtype=trt.nptype(engine.get_binding_dtype(0))))
    # bufferH.append(np.empty((3, 2, 2, 2),dtype=trt.nptype(engine.get_binding_dtype(0))))

    
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
        print(bufferH[i].nbytes)

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    print('cuda result:\n')
    print(bufferH[0])
    print(bufferH[1])
    print(bufferH[-1])
    print("check result:")
    temp1 = bufferH[-1]
    temp2 = qkv_cpu(bufferH[:4])
    print(temp2)
    
    print(check(temp1,temp2,True), "max diff=%f"%(np.max(np.abs(temp1 - temp2))) )
    
    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == '__main__':
    os.system("rm -f ./*.plan")
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    run()
