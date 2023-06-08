import depthai as dai
from ultralytics import YOLO
import blobconverter
import os

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)

model = YOLO(f'{script_directory}/../bottle_segmentation/YOLOv8m-512/weights/best.pt')
model.export(format='onnx', imgsz=512, dynamic=True, half=False, simplify=True, device=0)

blobconverter.from_onnx(
    model=f'{script_directory}/../bottle_segmentation/YOLOv8m-512/weights/best.onnx',
    output_dir=f"{script_directory}/../bottle_segmentation/YOLOv8m-512/weights//model.blob",
    data_type="FP16",
    shaves=6,
    use_cache=False,
    optimizer_params=['--reverse_input_channels', "--layout 'nhwc->nchw'", "--mean_values 0", "--scale 255"]
)
