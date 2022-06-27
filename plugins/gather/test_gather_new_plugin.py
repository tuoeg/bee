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
import ctypes
import numpy as np
from cuda import cudart  # 使用 cuda runtime API
import tensorrt as trt
soFilePath      = './GatherNew.so'
nBS             = 4
nSL             = 709
nEmbedding      = 768
epsilon         = 6e-6

np.random.seed(97)

def check(a, b, weak = False):
    if weak:
        return np.all( np.abs(a - b) < epsilon)
    else:
        return np.all( a == b )

# just matmul
def gather_new_cpu(buffer_h):
    _data = buffer_h[0]
    _indices = buffer_h[1]

    _1 = np.take(_data, _indices, axis=0)
    print(_1)
    return _1

def getGatherNewPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'GatherNew':
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
    inputTensorList.append( network.add_input('data', trt.float32, [-1,nSL,nEmbedding]) )
    inputTensorList.append( network.add_input('indices', trt.int32, [-1, nEmbedding]) )

    profile = builder.create_optimization_profile()
    profile.set_shape('data',[1,nSL,nEmbedding],[4, nSL,nEmbedding],[6, nSL,nEmbedding])
    profile.set_shape('indices',[1,nEmbedding],[4, nEmbedding],[6, nEmbedding])
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getGatherNewPlugin())

    network.mark_output(pluginLayer.get_output(0))

    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0,[nBS,nSL,nEmbedding])
    context.set_binding_shape(1,[nBS,nEmbedding])
    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))

    bufferH = []
    # data
    bufferH.append( np.random.rand(nBS,nSL,nEmbedding).astype(np.float32).reshape(nBS,nSL,nEmbedding))
    # x2
    bufferH.append((np.random.rand(nBS,nEmbedding) * nBS).astype(np.int32).reshape(nBS, nEmbedding))
    #output
    bufferH.append(np.empty((nBS, nEmbedding, nSL, nEmbedding),dtype=trt.nptype(engine.get_binding_dtype(0))))
    # bufferH.append(np.empty((4*768, 768),dtype=trt.nptype(engine.get_binding_dtype(0))))

    bufferD = []
    
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
        print(bufferH[i].nbytes)

    print('buffer D = ', bufferD, '\n')
    print('buffer H = ', bufferH, '\n')
    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    print('result:\n')
    print(bufferH[-1])
    print("check result:")
    temp1 = bufferH[-1]
    temp2 = gather_new_cpu(bufferH[:2])
    
    print(check(temp1,temp2,True), "max diff=%f"%(np.max(np.abs(temp1 - temp2))) )
    
    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == '__main__':
    os.system("rm -f ./*.plan")
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    run()
