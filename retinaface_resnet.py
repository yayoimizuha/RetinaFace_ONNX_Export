from retinaface.pre_trained_models import get_model
from torchinfo import summary
from torchao import quantization
from torch import manual_seed, rand,compile

model = get_model("resnet50_2020-07-20", max_size=640)

manual_seed(42)

dummy_input = rand(size=(1, 3, 640, 640))

print(model.model)
summary(model=model.model, input_data=dummy_input)

resp_orig = model.model(dummy_input)

qmodel = quantization.autoquant(compile(model.model,mode="max-autotune"),qtensor_class_list=quantization.DEFAULT_AUTOQUANT_CLASS_LIST)

resp_q = qmodel(dummy_input)

