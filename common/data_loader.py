import numpy as np

class DataLoader(object):
    def __init__(self):
        self.attention_mask = np.load('./data/eval_attention_mask.npy').astype(np.int32)
        self.bbox = np.load('./data/eval_bbox.npy').astype(np.int32)
        self.images = np.load('./data/eval_images.npy').astype(np.float32)
        self.input_ids = np.load('./data/eval_input_ids.npy')

data_loader = DataLoader()