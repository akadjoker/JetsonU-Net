import torch
from train import LightweightUNet  

 
model = LightweightUNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load('lightweight_unet_jetson.pth', map_location='cpu'))
model.eval()
 
dummy_input = torch.randn(1, 3, 256, 256)

 
torch.onnx.export(
    model, 
    dummy_input, 
    "lightweight_unet_jetson.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11   
)

