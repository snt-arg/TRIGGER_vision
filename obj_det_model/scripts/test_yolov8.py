from ultralytics import YOLO
import torch as T
from PIL import Image
import cv2
import os

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)

model = YOLO(f'{script_directory}/../bottle_segmentation/YOLOv8m-512/weights/best.pt')  # load a pretrained model
# Evaluate the model's performance on the validation set
results = model.val()
# Perform object detection on an image using the model
# from PIL
im1 = Image.open(f"{script_directory}/../datasets/roboflow_solidwaste/test/images/Test-Vid-bottles_mp4-312_jpg.rf.c26f7f45f13bb8877f6ae69427badd5f.jpg")
# results = model.predict(source=im1, save=True, show=True)  # save plotted images
res = model(im1)
res_plotted = res[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)
cv2.destroyAllWindows()
