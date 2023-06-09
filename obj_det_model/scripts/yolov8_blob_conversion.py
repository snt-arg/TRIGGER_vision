from ultralytics import YOLO
import blobconverter
import os
from pathlib import Path
import argparse

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
default_model = f'{script_directory}/../bottle_segmentation/yolov8n-seg_416/weights/best.pt'

parser = argparse.ArgumentParser(description='YOLOv8 Blob conversion script from ONNX and OPENVINO')
parser.add_argument('--model_path', type=str, default=default_model, help="Network size option")
parser.add_argument('--dynamic', action='store_true', help="Set onnx input size dynamic")
parser.add_argument('--no_simplify', action='store_true', help="Set onnx input size dynamic")
parser.add_argument('--device', type=int, default=0, help="CUDA device for conversion")
parser.add_argument('--opset', type=int, default=15, help="ONNX Opset")
parser.add_argument('--image_size', type=int, default=argparse.SUPPRESS, help="Input size (HxW)")

args = parser.parse_args()
assert args.model_path.endswith('.pt'), 'Model path has to be Pytorch checkpoint file'

model = YOLO(args.model_path)
model.export(format='onnx', imgsz=args.imgsz if hasattr(args, 'imgsz') else None,
             dynamic=args.dynamic, half=not args.dynamic,
             simplify=not args.no_simplify,
             device=args.device, opset=15)
try:
    blobconverter.from_onnx(
        model=os.path.join(os.path.dirname(args.model_path), Path(args.model_path).stem + '.onnx'),
        output_dir=os.path.dirname(args.model_path),
        data_type="FP16",
        shaves=6,
        use_cache=False,
        version = "2022.1",
        optimizer_params=["--mean_values=[0,0,0]", "--scale_values=[255,255,255]"]
    )
except Exception as e:
    print(e)
    print("Try the online converter at http://blobconverter.luxonis.com/ with options: --data_type=FP16 --mean_values=[0,0,0] --scale_values=[255,255,255] --reverse_input_channels --layout 'nchw->nhwc' ")

# Advanced OpenVINO CLI to convert onnx
# requires building OpenVINO from source (branch version 2022.1.1)
#  mo --input_model best.onnx --input_shape images[?,3,512,512] --compress_to_fp16 --mean_values=[0,0,0] --scale_values=[255,255,255] --reverse_input_channels --layout 'nchw->nhwc'
