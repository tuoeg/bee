from common.data_loader import DataLoader
from common.base_m import BaseM
import tensorrt as trt
import time
import os
G_LOGGER = trt.Logger(trt.Logger.ERROR)
import ctypes
import copy
ctypes.cdll.LoadLibrary('./plugins/layernorm/LayerNorm.so')
class MyAlgorithmSelector(trt.IAlgorithmSelector):

    def __init__(self, keepAll=True):
        super(MyAlgorithmSelector, self).__init__()
        self.keepAll = keepAll

    def select_algorithms(self, layerAlgorithmContext, layerAlgorithmList):
        print(layerAlgorithmContext.name, len(layerAlgorithmList))
        result = list((range(len(layerAlgorithmList))))
        # if layerAlgorithmContext.name == 'MatMul_14':
        #     print('-------------------------------------------------------------')
        #     for index, algorithm in enumerate(layerAlgorithmList):
        #         print(algorithm.algorithm_variant.tactic)
        #         print(algorithm.algorithm_variant.implementation)
        if 'ForeignNode' in layerAlgorithmContext.name:
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            for index, algorithm in enumerate(layerAlgorithmList):
                print(algorithm.algorithm_variant.tactic)
                print(algorithm.algorithm_variant.implementation)
            # result = [0]
        # if 'Clip_44' in layerAlgorithmContext.name:
        #     print('++++++++++++++++++++++++++++++++++++++' + layerAlgorithmContext.name + '++++++++++++++++++')
        #     for index, algorithm in enumerate(layerAlgorithmList):
        #         print(algorithm.algorithm_variant.tactic)
        #         print(algorithm.algorithm_variant.implementation)
        #     # result = [13]
        # if 'Clip_51' in layerAlgorithmContext.name:
        #     print('++++++++++++++++++++++++++++++++++++++' + layerAlgorithmContext.name + '++++++++++++++++++')
        #     for index, algorithm in enumerate(layerAlgorithmList):
        #         print(algorithm.algorithm_variant.tactic)
        #         print(algorithm.algorithm_variant.implementation)
        #     # result = [16]
        #     # result = [ index for index,algorithm in enumerate(layerAlgorithmList) if algorithm.algorithm_variant.implementation == 0x1fc87d7eb370bb7a ]
        return result

    def report_algorithms(self, modelAlgorithmContext, modelAlgorithmList):

        for i in range(len(modelAlgorithmContext)):
            context = modelAlgorithmContext[i]
            algorithm = modelAlgorithmList[i]

            print("Layer%4d:%s" % (i, context.name))
            nInput = context.num_inputs
            nOutput = context.num_outputs
            for j in range(nInput):
                ioInfo = algorithm.get_algorithm_io_info(j)
                print("    Input [%2d]:%s,%s,%s,%s" % (
                j, context.get_shape(j), ioInfo.dtype, ioInfo.strides, ioInfo.tensor_format))
            for j in range(nOutput):
                ioInfo = algorithm.get_algorithm_io_info(j + nInput)
                print("    Output[%2d]:%s,%s,%s,%s" % (
                j, context.get_shape(j + nInput), ioInfo.dtype, ioInfo.strides, ioInfo.tensor_format))
            print("    algorithm:[implementation:%d,tactic:%d,timing:%fms,workspace:%dMB]" % \
                  (algorithm.algorithm_variant.implementation,
                   algorithm.algorithm_variant.tactic,
                   algorithm.timing_msec,
                   algorithm.workspace_size))




class Fp16Trt(BaseM):
    def __init__(self):
        super().__init__()
        self.batch_list = [1, 2, 4, 8]
        self.total_num = 54
        self.time_list = []
        self.data_loader = DataLoader()
        self.trt_file = './plan/fp16.plan'
    
    def generate_batch(self,length, n):
        for i in range(0, length, n):
            yield [i, i + n]    
    
    def __surgen(self):
        pass
    def __build_trt(self):
        if os.path.exists(self.trt_file) is False:
            
            builder = trt.Builder(G_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            profile = builder.create_optimization_profile()
            config = builder.create_builder_config()
            config.max_workspace_size = 3 << 30
            parser = trt.OnnxParser(network, G_LOGGER)
            with open('./onnx/layernorm_plugin.onnx', 'rb') as model:
                if not parser.parse(model.read()):
                    print("Failed parsing onnx file!")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    exit()
            print("Succeeded parsing onnx file!")


            input_ids = network.get_input(0)
            bbox = network.get_input(1)
            images = network.get_input(2)
            attention_mask = network.get_input(3)

            # profile.set_shape(input_ids.name, [6,709,768], [6,709,768], [6,709,768])
            # profile.set_shape(bbox.name, [6,1,1,709], [6,1,1,709], [6,1,1,709])
            # profile.set_shape(images.name, [6,12,709,709], [6,12,709,709], [6,12,709,709])
            # profile.set_shape(attention_mask.name, [6,12,709,709], [6,12,709,709], [6,12,709,709])
            profile.set_shape(input_ids.name, [1, 512], [4, 512], [8, 512])
            profile.set_shape(bbox.name, [1, 512, 4], [4, 512, 4], [8, 512, 4])
            profile.set_shape(images.name, [1, 3, 224, 224], [4, 3, 224, 224], [8, 3, 224, 224])
            profile.set_shape(attention_mask.name, [1,709], [4,709], [8,709])
            config.add_optimization_profile(profile)
            config.set_flag(trt.BuilderFlag.DEBUG)
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            config.clear_flag(trt.BuilderFlag.TF32)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            config.flags = config.flags | (1 << int(trt.BuilderFlag.STRICT_TYPES)) | (1 << int(trt.BuilderFlag.FP16))
            config.algorithm_selector = MyAlgorithmSelector(True)  # set algorithm_selector
            engineString = builder.build_serialized_network(network, config)
            with open(self.trt_file, 'wb') as f:
                f.write(engineString)
    
    def infer(self):
        self.__build_trt()
        
        # first time is wrong
        for i in self.batch_list:
            self.infer_batch(self.trt_file, i)
        
        for i in self.batch_list:
            self.infer_batch(self.trt_file, i)
        
        self.trt_matric(self.trt_file)
        self.print()