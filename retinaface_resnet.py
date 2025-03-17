import torch
from retinaface.pre_trained_models import get_model
from retinaface.network import RetinaFace
from torch.ao.quantization import default_dynamic_qconfig, QuantStub, DeQuantStub
from torchinfo import summary
from torchao.quantization import quantize_, int8_weight_only, int8_dynamic_activation_int8_weight, dynamic_quant
from torch import manual_seed, rand, compile, inference_mode, save, ao, quantization
from os import environ, pathsep

torch.backends.quantized.engine = 'fbgemm'


class QuantRetinaFace(torch.nn.Module):
    def __init__(self, model):
        super(QuantRetinaFace, self).__init__()
        self.model = model
        self.quant = QuantStub()
        self.de_quant = DeQuantStub()

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        q = self.quant(inputs)
        q_res = self.model(q)
        bboxes, prob, landmarks = self.de_quant(q_res)
        return {"bbox": bboxes, "prob": prob, "landmark": landmarks}


environ["VSLANG"] = "1033"
environ["LIB"] += pathsep + r"C:\Users\tomokazu\AppData\Local\Programs\Python\Python312\libs"

modules_to_fuse = [
    ['model.body.conv1', 'model.body.bn1'],
    # layer1
    ['model.body.layer1.0.conv1', 'model.body.layer1.0.bn1'],
    ['model.body.layer1.0.conv2', 'model.body.layer1.0.bn2'],
    ['model.body.layer1.0.conv3', 'model.body.layer1.0.bn3'],
    ['model.body.layer1.0.downsample.0', 'model.body.layer1.0.downsample.1'],
    ['model.body.layer1.1.conv1', 'model.body.layer1.1.bn1'],
    ['model.body.layer1.1.conv2', 'model.body.layer1.1.bn2'],
    ['model.body.layer1.1.conv3', 'model.body.layer1.1.bn3'],
    ['model.body.layer1.2.conv1', 'model.body.layer1.2.bn1'],
    ['model.body.layer1.2.conv2', 'model.body.layer1.2.bn2'],
    ['model.body.layer1.2.conv3', 'model.body.layer1.2.bn3'],
    # layer2
    ['model.body.layer2.0.conv1', 'model.body.layer2.0.bn1'],
    ['model.body.layer2.0.conv2', 'model.body.layer2.0.bn2'],
    ['model.body.layer2.0.conv3', 'model.body.layer2.0.bn3'],
    ['model.body.layer2.0.downsample.0', 'model.body.layer2.0.downsample.1'],
    ['model.body.layer2.1.conv1', 'model.body.layer2.1.bn1'],
    ['model.body.layer2.1.conv2', 'model.body.layer2.1.bn2'],
    ['model.body.layer2.1.conv3', 'model.body.layer2.1.bn3'],
    ['model.body.layer2.2.conv1', 'model.body.layer2.2.bn1'],
    ['model.body.layer2.2.conv2', 'model.body.layer2.2.bn2'],
    ['model.body.layer2.2.conv3', 'model.body.layer2.2.bn3'],
    ['model.body.layer2.3.conv1', 'model.body.layer2.3.bn1'],
    ['model.body.layer2.3.conv2', 'model.body.layer2.3.bn2'],
    ['model.body.layer2.3.conv3', 'model.body.layer2.3.bn3'],
    # layer3
    ['model.body.layer3.0.conv1', 'model.body.layer3.0.bn1'],
    ['model.body.layer3.0.conv2', 'model.body.layer3.0.bn2'],
    ['model.body.layer3.0.conv3', 'model.body.layer3.0.bn3'],
    ['model.body.layer3.0.downsample.0', 'model.body.layer3.0.downsample.1'],
    ['model.body.layer3.1.conv1', 'model.body.layer3.1.bn1'],
    ['model.body.layer3.1.conv2', 'model.body.layer3.1.bn2'],
    ['model.body.layer3.1.conv3', 'model.body.layer3.1.bn3'],
    ['model.body.layer3.2.conv1', 'model.body.layer3.2.bn1'],
    ['model.body.layer3.2.conv2', 'model.body.layer3.2.bn2'],
    ['model.body.layer3.2.conv3', 'model.body.layer3.2.bn3'],
    ['model.body.layer3.3.conv1', 'model.body.layer3.3.bn1'],
    ['model.body.layer3.3.conv2', 'model.body.layer3.3.bn2'],
    ['model.body.layer3.3.conv3', 'model.body.layer3.3.bn3'],
    ['model.body.layer3.4.conv1', 'model.body.layer3.4.bn1'],
    ['model.body.layer3.4.conv2', 'model.body.layer3.4.bn2'],
    ['model.body.layer3.4.conv3', 'model.body.layer3.4.bn3'],
    ['model.body.layer3.5.conv1', 'model.body.layer3.5.bn1'],
    ['model.body.layer3.5.conv2', 'model.body.layer3.5.bn2'],
    ['model.body.layer3.5.conv3', 'model.body.layer3.5.bn3'],
    # layer4
    ['model.body.layer4.0.conv1', 'model.body.layer4.0.bn1'],
    ['model.body.layer4.0.conv2', 'model.body.layer4.0.bn2'],
    ['model.body.layer4.0.conv3', 'model.body.layer4.0.bn3'],
    ['model.body.layer4.0.downsample.0', 'model.body.layer4.0.downsample.1'],
    ['model.body.layer4.1.conv1', 'model.body.layer4.1.bn1'],
    ['model.body.layer4.1.conv2', 'model.body.layer4.1.bn2'],
    ['model.body.layer4.1.conv3', 'model.body.layer4.1.bn3'],
    ['model.body.layer4.2.conv1', 'model.body.layer4.2.bn1'],
    ['model.body.layer4.2.conv2', 'model.body.layer4.2.bn2'],
    ['model.body.layer4.2.conv3', 'model.body.layer4.2.bn3'],
    # fpn
    ['model.fpn.output1.0', 'model.fpn.output1.1'],
    ['model.fpn.output2.0', 'model.fpn.output2.1'],
    ['model.fpn.output3.0', 'model.fpn.output3.1'],
    ['model.fpn.merge1.0', 'model.fpn.merge1.1'],
    ['model.fpn.merge2.0', 'model.fpn.merge2.1'],
    # model.ssh1
    ['model.ssh1.conv3X3.0', 'model.ssh1.conv3X3.1'],
    ['model.ssh1.conv5X5_1.0', 'model.ssh1.conv5X5_1.1'],
    ['model.ssh1.conv5X5_2.0', 'model.ssh1.conv5X5_2.1'],
    ['model.ssh1.conv7X7_2.0', 'model.ssh1.conv7X7_2.1'],
    ['model.ssh1.conv7x7_3.0', 'model.ssh1.conv7x7_3.1'],
    # model.ssh2
    ['model.ssh2.conv3X3.0', 'model.ssh2.conv3X3.1'],
    ['model.ssh2.conv5X5_1.0', 'model.ssh2.conv5X5_1.1'],
    ['model.ssh2.conv5X5_2.0', 'model.ssh2.conv5X5_2.1'],
    ['model.ssh2.conv7X7_2.0', 'model.ssh2.conv7X7_2.1'],
    ['model.ssh2.conv7x7_3.0', 'model.ssh2.conv7x7_3.1'],
    # model.ssh3
    ['model.ssh3.conv3X3.0', 'model.ssh3.conv3X3.1'],
    ['model.ssh3.conv5X5_1.0', 'model.ssh3.conv5X5_1.1'],
    ['model.ssh3.conv5X5_2.0', 'model.ssh3.conv5X5_2.1'],
    ['model.ssh3.conv7X7_2.0', 'model.ssh3.conv7X7_2.1'],
    ['model.ssh3.conv7x7_3.0', 'model.ssh3.conv7x7_3.1']
]

with (inference_mode()):
    model = get_model("resnet50_2020-07-20", max_size=640)
    model.model = QuantRetinaFace(model.model)
    model.eval()

    manual_seed(42)

    dummy_input = rand(size=(2, 3, 640, 640))

    print(model.model)

    summary(model=model.model, input_data=dummy_input)
    save(model.model.state_dict(), "base.pth")

    resp_orig: dict[str, torch.Tensor] = model.model(dummy_input)

    # qmodel = model.model

    # qmodel = compile(model=model.model, mode="max-autotune")
    qmodel = model.model
    qmodel.qconfig = quantization.get_default_qconfig(backend="fbgemm")
    qmodel.eval()
    fused = quantization.fuse_modules(qmodel, modules_to_fuse)
    quantization.prepare(fused, inplace=True)
    fused(dummy_input)
    q_fused = quantization.convert(fused, inplace=True)
    # q_fused = quantization.quantize_dynamic(fused, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.quint8)
    q_fused.eval()

    # resp_q: dict[str, torch.Tensor] = q_fused(dummy_input)
    print(q_fused)
    # summary(model=q_fused, input_data=dummy_input)
    save(q_fused.state_dict(), "quant.pth")
    # print([(k, v) for k, v in resp_orig.items()])
    # print([(k, v) for k, v in resp_q.items()])
