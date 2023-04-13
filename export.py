from backbone import get_network
from easydict import EasyDict
import torch
from loguru import logger
import onnxsim
import onnx

def export_onnx(net, model_save, input_shape: tuple):
    dummy_input = torch.autograd.Variable(
        torch.randn(1, 3, input_shape[0], input_shape[1])
    )
    torch.onnx.export(
        net,
        dummy_input,
        model_save,
        verbose=True,
        keep_initializers_as_inputs=True,
        opset_version=11,
        input_names=["data"],
        output_names=["output"],
    )
    logger.info("finished exporting onnx.")
    logger.info("start simplifying onnx.")
    input_data = {"data": dummy_input.detach().cpu().numpy()}
    model_sim, flag = onnxsim.simplify(model_save, input_data=input_data)
    if flag:
        onnx.save(model_sim, model_save)
        logger.info("simplify onnx successfully")
    else:
        logger.error("simplify onnx failed")
    logger.info(f"export onnx model to {model_save}")

cfg = dict(
        use_onenetwork=True,
        width_mult=1.0,
        num_verts=1787,
        input_size=256,
        task=3,
        network='resnet_jmlr',
        no_gap=False,
        use_arcface=False,
    )
config = EasyDict(cfg)
net = get_network(config)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.load_state_dict(torch.load("workplace/s2/best_model.pth", map_location=device))
net.eval()

export_onnx(net, "workplace/s2/best_model.onnx", (256, 256))