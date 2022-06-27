from torch_model.lmv3_model import Lmv3Model
from fp32.fp32_trt import Fp32Trt
from fp16.fp16_trt import Fp16Trt
from int8.int8_trt import Int8Trt


torch  = Lmv3Model()

torch.infer()
# a.onnx()

fp32 = Fp32Trt()
fp32.infer()

fp16 = Fp16Trt()
fp16.infer()

int8 = Int8Trt()
int8.infer()

print('torch result')
torch.print()
print('fp32 result')
fp32.print()
print('fp16 result')
fp16.print()
print('int8 result')
int8.print()
