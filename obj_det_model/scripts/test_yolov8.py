from ultralytics import YOLO
import torch as T
from PIL import Image
import cv2
import os
from pathlib import Path
import argparse


script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)

default_model = 'yolov8m-seg.pt'

parser = argparse.ArgumentParser(description='YOLOv8 evaluation and test on image')
parser.add_argument('--model_path', type=str, default=default_model, help="Network size option")
parser.add_argument('--device', type=int, default=0, help="CUDA device for conversion")
parser.add_argument('--test_image_path', type=str, default=argparse.SUPPRESS, help="Image for testing the model output")
parser.add_argument('--eval', action='store_true', help="Eval model on standard metrics")

args = parser.parse_args()
assert args.model_path.endswith('.pt'), 'Model path has to be Pytorch checkpoint file'

model = YOLO(args.model_path)

if args.eval:
    # Evaluate the model's performance on the validation set
    results = model.val()

# Perform object detection on an image using the model
# from PIL
if hasattr(args, 'test_image_path') and os.path.isfile(args.test_image_path):
    im1 = Image.open(args.test_image_path)
    # results = model.predict(source=im1, save=True, show=True)  # save plotted images
    res = model(im1)
    res_plotted = res[0].plot()
    cv2.imshow("result", res_plotted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
