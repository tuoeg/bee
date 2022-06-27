import os
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer
)
import layoutlmft.data.funsd
from datasets import load_dataset
import sys
sys.path.append('..')
from common.data_loader import DataLoader
from common.base_m import BaseM
import torch
import numpy as np
from datasets import load_metric
import time
metric = load_metric("seqeval")
class Lmv3Model(BaseM):
    def __init__(self):

        config = AutoConfig.from_pretrained(
            'HYPJUDY/layoutlmv3-base-finetuned-funsd',
            num_labels=7,
            finetuning_task='ner',
            cache_dir=None,
            revision='main',
            input_size=224,
            use_auth_token=None
        )

        tokenizer = AutoTokenizer.from_pretrained(
            'HYPJUDY/layoutlmv3-base-finetuned-funsd',
            tokenizer_file=None,
            cache_dir=None,
            use_fast=True,
            add_prefix_space=True,
            revision='main',
            use_auth_token=None
        )

        self.m = AutoModelForTokenClassification.from_pretrained(
            'HYPJUDY/layoutlmv3-base-finetuned-funsd',
            from_tf=False,
            config=config,
            cache_dir=None,
            revision='main',
            use_auth_token=None
        )

        self.device = torch.device('cuda')
        self.m.to(self.device)
        self.m.eval()
        self.data_loader = DataLoader()

        self.batch_list = [1, 2, 4, 8]
        self.total_num = 54
        self.time_list = []
    def generate_batch(self,length, n):
        for i in range(0, length, n):
            yield [i, i + n]
    
    def infer(self):
        for i in self.batch_list:
            attention_mask = self.data_loader.attention_mask[0:i]
            bbox = self.data_loader.bbox[0:i]
            images = self.data_loader.images[0:i]
            input_ids = self.data_loader.input_ids[0:i]
            print(attention_mask) 
            attention_mask = torch.from_numpy(attention_mask).to(self.device)
            bbox = torch.from_numpy(bbox).to(self.device)
            images = torch.from_numpy(images).to(self.device)
            input_ids = torch.from_numpy(input_ids).to(self.device)

            with torch.no_grad():
                
                start_time = time.time()
                torch_res = self.m(input_ids, bbox, images, attention_mask)
                print(torch_res[0].cpu().detach().numpy().shape)
                end_time = time.time()
                self.time_list.append((end_time - start_time)*1000)

        # predictions = np.ones((self.total_num,709),dtype=np.int32)
        # for i, ii in self.generate_batch(self.total_num, 4):
        #     attention_mask = self.data_loader.attention_mask[i:ii]
        #     bbox = self.data_loader.bbox[i:ii]
        #     images = self.data_loader.images[i:ii]
        #     input_ids = self.data_loader.input_ids[i:ii]
            
        #     attention_mask = torch.from_numpy(attention_mask).to(self.device)
        #     bbox = torch.from_numpy(bbox).to(self.device)
        #     images = torch.from_numpy(images).to(self.device)
        #     input_ids = torch.from_numpy(input_ids).to(self.device)
        #     with torch.no_grad():
        #         pre = self.m(input_ids, bbox, images, attention_mask)
            
        #     pred = np.argmax(pre[0].cpu().detach().numpy(), axis=2)
        #     predictions[i:ii,:] = pred


        # label_list = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
        # labels = np.load('./data/eval_labels.npy').astype(np.int32)

        # true_predictions = [
        #     [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        #     for prediction, label in zip(predictions, labels)
        # ]
        # true_labels = [
        #     [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        #     for prediction, label in zip(predictions, labels)
        # ]
        # results = metric.compute(predictions=true_predictions, references=true_labels)
        # print(results["overall_precision"])
        # print(results["overall_recall"])
        # print(results["overall_f1"])

        # self.precision = results['overall_precision']
        # self.recall = results['overall_recall']
        # self.f1 = results['overall_f1']
        self.matric()
    def onnx(self):
        onnx_f = './onnx/original.onnx'
        self.data_loader = DataLoader()
        attention_mask = self.data_loader.attention_mask[0:6]
        bbox = self.data_loader.bbox[0:6]
        images = self.data_loader.images[0:6]
        input_ids = self.data_loader.input_ids[0:6]
        
        device = torch.device('cpu')
        self.m.to(device)
        
        attention_mask = torch.from_numpy(attention_mask).to(device)
        bbox = torch.from_numpy(bbox).to(device)
        images = torch.from_numpy(images).to(device)
        input_ids = torch.from_numpy(input_ids).to(device)
        

        with torch.no_grad():
            torch.onnx.export(
            self.m,
            args=(input_ids, bbox, images, attention_mask),
            f=onnx_f,
            input_names=['input_ids', 'bbox',  'images', 'attention_mask'],
            output_names=['out'],
            do_constant_folding=False,
            dynamic_axes={'input_ids': [0],
                          'bbox': [0],
                          'images': [0],
                          'attention_mask': [0],
                          'out': [0]
                          },
            opset_version=11
        )




