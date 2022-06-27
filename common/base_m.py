from datasets import load_metric
from .data_loader import DataLoader
import time
import numpy as np
import torch
import pycuda.autoinit
import tensorrt as trt

from cuda import cudart 

# import pycuda.driver as cuda

G_LOGGER = trt.Logger(trt.Logger.ERROR)
metric = load_metric("seqeval")
class BaseM(object):
    def __init__(self):
        self.runtime = trt.Runtime(G_LOGGER)
    
    def torch_matric(self):
        torch.cuda.empty_cache()
        predictions = np.ones((self.total_num,709),dtype=np.int32)
        for i, ii in self.generate_batch(self.total_num, 4):
            attention_mask = self.data_loader.attention_mask[i:ii]
            bbox = self.data_loader.bbox[i:ii]
            images = self.data_loader.images[i:ii]
            input_ids = self.data_loader.input_ids[i:ii]
            
            attention_mask = torch.from_numpy(attention_mask).to(self.device)
            bbox = torch.from_numpy(bbox).to(self.device)
            images = torch.from_numpy(images).to(self.device)
            input_ids = torch.from_numpy(input_ids).to(self.device)
            with torch.no_grad():
                pre = self.m(input_ids, bbox, images, attention_mask)
            
            pred = np.argmax(pre[0].cpu().detach().numpy(), axis=2)
            predictions[i:ii,:] = pred


        label_list = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
        labels = np.load('./data/eval_labels.npy').astype(np.int32)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        print(results["overall_precision"])
        print(results["overall_recall"])
        print(results["overall_f1"])

        self.precision = results['overall_precision']
        self.recall = results['overall_recall']
        self.f1 = results['overall_f1']
    
    def trt_matric(self, file):
        predictions = np.ones((self.total_num,709),dtype=np.int32)

        # with open(file, "rb") as f:
        #     engine = self.runtime.deserialize_cuda_engine(f.read())

        
        # context = engine.create_execution_context()
        # context.set_binding_shape(0,[2,512])
        # context.set_binding_shape(1,[2,512,4])
        # context.set_binding_shape(2,[2,3, 224, 224])
        # context.set_binding_shape(3,[2,709])

        for i_s, ii_e in self.generate_batch(self.total_num, 2):
            print(i_s, ii_e)
            torch.cuda.empty_cache()
            with open(file, "rb") as f:
                engine = self.runtime.deserialize_cuda_engine(f.read())

        
            context = engine.create_execution_context()
            context.set_binding_shape(0,[ii_e - i_s,512])
            context.set_binding_shape(1,[ii_e - i_s,512,4])
            context.set_binding_shape(2,[ii_e - i_s,3, 224, 224])
            context.set_binding_shape(3,[ii_e - i_s,709])

            print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    
            nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
            nOutput = engine.num_bindings - nInput
            for i in range(engine.num_bindings):
                print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))


            bufferH = []
            self.data_loader = DataLoader()

            # input_id
            bufferH.append( self.data_loader.input_ids[i_s:ii_e])
            # bbox
            bufferH.append( self.data_loader.bbox[i_s:ii_e])
            # images
            bufferH.append( self.data_loader.images[i_s:ii_e])
            # attention_mask
            bufferH.append( self.data_loader.attention_mask[i_s:ii_e])
            #output
            bufferH.append(np.empty((ii_e - i_s, 709,7),dtype=trt.nptype(engine.get_binding_dtype(2))))
            bufferD = []
        
            for i in range(engine.num_bindings):
                print(bufferH[i].nbytes)
                bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

            for i in range(nInput):
                cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

       

            for i in range(nInput, nInput + nOutput):
                cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)


            context.execute_v2(bufferD)
       
            for i in range(nInput, nInput + nOutput):
                cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            
            print(bufferH[-1].shape)
            pred = np.argmax(bufferH[-1], axis=2)
            print(pred.shape)
            predictions[i_s:ii_e,:] = pred

        label_list = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
        labels = np.load('./data/eval_labels.npy').astype(np.int32)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        print(results["overall_precision"])
        print(results["overall_recall"])
        print(results["overall_f1"])

        self.precision = results['overall_precision']
        self.recall = results['overall_recall']
        self.f1 = results['overall_f1']

    def infer_batch(self, file, batch):
        torch.cuda.empty_cache()
        with open(file, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        context.set_binding_shape(0,[batch,512])
        context.set_binding_shape(1,[batch,512,4])
        context.set_binding_shape(2,[batch,3, 224, 224])
        context.set_binding_shape(3,[batch,709])
        print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
   
        nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
        nOutput = engine.num_bindings - nInput
        for i in range(engine.num_bindings):
            print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))


        bufferH = []
        # input_id
        bufferH.append( self.data_loader.input_ids[0:batch])
        # bbox
        bufferH.append( self.data_loader.bbox[0:batch])
        # images
        bufferH.append( self.data_loader.images[0:batch])
        # attention_mask
        bufferH.append( self.data_loader.attention_mask[0:batch])
        #output
        bufferH.append(np.empty((batch, 709,7),dtype=trt.nptype(engine.get_binding_dtype(2))))
        bufferD = []
        
        for i in range(engine.num_bindings):
            print(bufferH[i].nbytes)
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # warm up
        for i in range(3):
            context.execute_v2(bufferD)

        for i in range(nInput, nInput + nOutput):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        print(bufferH[-1])
        # 100
        start_time = time.time()
        for i in range(10):
            context.execute_v2(bufferD)
        end_time = time.time()
        self.time_list.append((end_time - start_time) * 100)
        
        for i in range(nInput, nInput + nOutput):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        print(bufferH[-1])
    

    def infer_2_batch(self, file_1, file_2, batch):
        
        with open(file_1, "rb") as f:
            engine1 = self.runtime.deserialize_cuda_engine(f.read())
        
        with open(file_2, "rb") as f:
            engine2 = self.runtime.deserialize_cuda_engine(f.read())

        context1 = engine1.create_execution_context()
        context1.set_binding_shape(0,[batch,512])
        context1.set_binding_shape(1,[batch,512,4])
        context1.set_binding_shape(2,[batch,3, 224, 224])
        context1.set_binding_shape(3,[batch,709])
        print("Binding all? %s"%(["No","Yes"][int(context1.all_binding_shapes_specified)]))
   
        nInput1 = np.sum([ engine1.binding_is_input(i) for i in range(engine1.num_bindings) ])
        nOutput1 = engine1.num_bindings - nInput1
        for i in range(engine1.num_bindings):
            print("input ->" if engine1.binding_is_input(i) else "output->",engine1.get_binding_dtype(i),engine1.get_binding_shape(i),context.get_binding_shape(i))


        context2 = engine2.create_execution_context()
        context2.set_binding_shape(0,[batch,512])
        context2.set_binding_shape(1,[batch,512,4])
        context2.set_binding_shape(2,[batch,3, 224, 224])
        context2.set_binding_shape(3,[batch,709])
        print("Binding all? %s"%(["No","Yes"][int(context2.all_binding_shapes_specified)]))
   
        nInput2 = np.sum([ engine2.binding_is_input(i) for i in range(engine2.num_bindings) ])
        nOutput2 = engine2.num_bindings - nInput
        for i in range(engine2.num_bindings):
            print("input ->" if engine2.binding_is_input(i) else "output->",engine2.get_binding_dtype(i),engine2.get_binding_shape(i),context.get_binding_shape(i))


        bufferH = []
        # input_id
        bufferH.append( self.data_loader.input_ids[0:batch])
        # bbox
        bufferH.append( self.data_loader.bbox[0:batch])
        # images
        bufferH.append( self.data_loader.images[0:batch])
        # attention_mask
        bufferH.append( self.data_loader.attention_mask[0:batch])
        #output
        bufferH.append(np.empty((batch, 709,7),dtype=trt.nptype(engine1.get_binding_dtype(2))))
        bufferD = []
        
        for i in range(engine1.num_bindings):
            print(bufferH[i].nbytes)
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(nInput1):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # warm up
        for i in range(3):
            context1.execute_v2(bufferD)
            context2.execute_v2(bufferD)

        for i in range(nInput2, nInput2 + nOutput2):
            cudart.cudaMemcpy(bufferH1[i].ctypes.data, bufferD2[i], bufferH1[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        print(bufferH[-1])
        # 100
        start_time = time.time()
        for i in range(10):
            context1.execute_v2(bufferD)
            context2.execute_v2(bufferD)
        end_time = time.time()
        self.time_list.append((end_time - start_time) * 100)
        
        for i in range(nInput2, nInput2 + nOutput2):
            cudart.cudaMemcpy(bufferH1[i].ctypes.data, bufferD2[i], bufferH1[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        print(bufferH[-1])
        
    def print(self):
        print('1 batch = %d, 2 batch = %d,4 batch = %d,8 batch = %d,'% (self.time_list[4], self.time_list[5], self.time_list[6], self.time_list[7]))
        print('precison = %f, recall = %f, f1 = %f' % (self.precision, self.recall,self.f1))
        
