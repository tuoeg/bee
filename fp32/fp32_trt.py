from common.surgen import surgen, layernorm_surgen
from common.data_loader import DataLoader
from common.base_m import BaseM
import tensorrt as trt
import time
import os
G_LOGGER = trt.Logger(trt.Logger.ERROR)

import ctypes
import copy
ctypes.cdll.LoadLibrary('./plugins/layernorm/LayerNorm.so')
class Fp32Trt(BaseM):
    def __init__(self):
        super().__init__()
        self.batch_list = [1, 2, 4, 8]
        self.total_num = 54
        self.time_list = []
        self.data_loader = DataLoader()
        self.trt_file = './plan/fp32.plan'
    
    def generate_batch(self,length, n):
        for i in range(0, length, n):
            yield [i, i + n]    
    
    def __surgen(self):
        surgen()
        layernorm_surgen()

    def __build_trt(self):
        if os.path.exists(self.trt_file) is True:
            return
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
        # config.flags = config.flags | (1 << int(trt.BuilderFlag.STRICT_TYPES)) | (1 << int(trt.BuilderFlag.FP16))
        # config.algorithm_selector = MyAlgorithmSelector(True)  # set algorithm_selector
        engineString = builder.build_serialized_network(network, config)
        with open(self.trt_file, 'wb') as f:
            f.write(engineString)
    
    def infer(self):
        self.__surgen()
        self.__build_trt()
        
        # first time is wrong
        for i in self.batch_list:
            self.infer_batch(self.trt_file, i)
        
        for i in self.batch_list:
            self.infer_batch(self.trt_file, i)
        
        self.trt_matric(self.trt_file)
        self.print()