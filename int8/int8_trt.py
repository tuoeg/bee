from common.surgen import surgen, layernorm_surgen
from common.data_loader import DataLoader
from common.base_m import BaseM
import tensorrt as trt
import time
import os
G_LOGGER = trt.Logger(trt.Logger.ERROR)

import ctypes
ctypes.cdll.LoadLibrary('./plugins/layernorm/LayerNorm.so')

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np



class Calibrator(trt.IInt8EntropyCalibrator2):
    '''calibrator
        IInt8EntropyCalibrator2
        IInt8LegacyCalibrator
        IInt8EntropyCalibrator
        IInt8MinMaxCalibrator
    '''

    def __init__(self, stream, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)
        # dataloader
        self.stream = stream

        self.d_input_input_ids = cuda.mem_alloc(self.stream.input_ids.nbytes)
        self.d_input_bbox = cuda.mem_alloc(self.stream.bbox.nbytes)
        self.d_input_images = cuda.mem_alloc(self.stream.images.nbytes)
        self.d_input_attention_mask = cuda.mem_alloc(self.stream.attention_mask.nbytes)

        self.cache_file = cache_file
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):

        batch = self.stream.next_batch()
        if not batch[0].size:
            return None

        cuda.memcpy_htod(self.d_input_input_ids, batch[0])
        cuda.memcpy_htod(self.d_input_bbox, batch[1])
        cuda.memcpy_htod(self.d_input_images, batch[2])
        cuda.memcpy_htod(self.d_input_attention_mask, batch[3])
        # return [int(self.d_input_input_ids), int(self.d_input_bbox), int(self.d_input_images)]
        return [int(self.d_input_input_ids),int(self.d_input_bbox),int(self.d_input_images),int(self.d_input_attention_mask)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print(f"[INFO] Using calibration cache to save time: {self.cache_file}")
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            print(f"[INFO] Caching calibration data for future use: {self.cache_file}")
            f.write(cache)


class DataLoader:
    def __init__(self,batch=1, batch_size=32):
        self.index = 0
        self.length = batch
        self.batch_size = batch_size
        self.input_ids = np.zeros((self.batch_size, 512), dtype=np.int32)
        self.bbox = np.zeros((self.batch_size, 512,4), dtype=np.int32)
        self.images = np.zeros((self.batch_size, 3, 224, 224), dtype=np.float32)
        self.attention_mask = np.zeros((self.batch_size, 709), dtype=np.int32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                bbox = np.load('./data/train_bbox.npy').astype(np.int32)
                images = np.load('./data/train_images.npy').astype(np.float32)
                input_ids = np.load('./data/train_input_ids.npy').astype(np.int32)
                attention_mask = np.load('./data/train_attention_mask.npy').astype(np.int32)
                self.input_ids[i] = input_ids[i + self.index * self.batch_size]
                self.bbox[i] = bbox[i + self.index * self.batch_size]
                self.images[i] = images[i + self.index * self.batch_size]
                self.attention_mask[i] = attention_mask[i + self.index * self.batch_size]

            self.index += 1
            return [np.ascontiguousarray(self.input_ids, dtype=np.int32),np.ascontiguousarray(self.bbox, dtype=np.int32),
                    np.ascontiguousarray(self.images, dtype=np.float32),np.ascontiguousarray(self.attention_mask, dtype=np.int32)]

        else:
            return [np.array([]),np.array([]),np.array([]),np.array([])]

    def __len__(self):
        return self.length



class Int8Trt(BaseM):
    def __init__(self):
        super().__init__()
        self.batch_list = [1, 2, 4, 8]
        self.total_num = 54
        self.time_list = []
        self.data_loader = DataLoader()
        self.trt_file = './plan/int8.plan'
    
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


        # build trt engine
        builder.max_batch_size = 54
        input_ids = network.get_input(0)
        bbox = network.get_input(1)
        images = network.get_input(2)
        attention_mask = network.get_input(3)
        profile.set_shape(input_ids.name, [1, 512], [4, 512], [8, 512])
        profile.set_shape(bbox.name, [1, 512, 4], [4, 512, 4], [8, 512, 4])
        profile.set_shape(images.name, [1, 3, 224, 224], [4, 3, 224, 224], [8, 3, 224, 224])
        profile.set_shape(attention_mask.name, [1, 709], [4, 709], [8, 709])
        
        config.add_optimization_profile(profile)
        # 不然会有警告
        config.set_calibration_profile(profile)
           

        config.set_flag(trt.BuilderFlag.INT8)

        calibration_stream = DataLoader()
        config.int8_calibrator = Calibrator(calibration_stream, '.cache')
        engineString = builder.build_serialized_network(network, config)

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