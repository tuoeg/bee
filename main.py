from torch_model.lmv3_model import Lmv3Model
from fp32.fp32_trt import Fp32Trt



a = Lmv3Model()

a.infer()
a.onnx()


b = Fp32Trt()
b.infer()
