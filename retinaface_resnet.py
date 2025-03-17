from retinaface.pre_trained_models import get_model
from torchinfo import summary
from torchao import quantize_, quantization
from torch import manual_seed, rand, compile, inference_mode
from os import environ, pathsep

environ["VSLANG"] = "1033"
environ["LIB"] += pathsep + r"C:\Users\tomokazu\AppData\Local\Programs\Python\Python312\libs"

with (inference_mode()):
    model = get_model("resnet50_2020-07-20", max_size=640)
    model.eval()

    manual_seed(42)

    dummy_input = rand(size=(1, 3, 640, 640))

    # print(model.model)
    summary(model=model.model, input_data=dummy_input)

    resp_orig = model.model(dummy_input)

    qmodel = model.model

    quantization.quant_api.quantize_(qmodel, quantization.quant_api.Int4WeightOnlyConfig())

    resp_q = qmodel(dummy_input)
    summary(model=qmodel, input_data=dummy_input)
    print([r.shape for r in resp_orig])
