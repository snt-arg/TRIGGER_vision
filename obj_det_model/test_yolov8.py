from ultralytics import YOLO
import torch as T
from PIL import Image
import cv2

model = YOLO('./runs/detect/train3/weights/best.pt')  # load a pretrained model
# Perform object detection on an image using the model
# from PIL
im1 = Image.open("bus.jpg")
# results = model.predict(source=im1, save=True, show=True)  # save plotted images
res = model(im1)
res_plotted = res[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)
cv2.destroyAllWindows()
