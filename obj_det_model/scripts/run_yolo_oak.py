import depthai as dai
import cv2
import numpy as np
import argparse
import time
import os
from ultralytics import YOLO

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
default_model = f'{script_directory}/../blobs/yolov8n_segm_6shave.blob'

model = YOLO(f'{script_directory}/../bottle_segmentation/yolov8n-seg_416/weights/best.pt')

parser = argparse.ArgumentParser(description='Run YOLOv8 on OAK')
parser.add_argument('--net_blob_path', type=str, default=default_model, help="Network BLOB path")
parser.add_argument('--image_size', type=int, default=416, help="Input size (HxW)")
parser.add_argument('--fps', type=int, default=15)

args = parser.parse_args()

pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2022_1)

cam = pipeline.createColorCamera()
cam.setPreviewSize(args.image_size, args.image_size)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
order = cam.getColorOrder()
cam.setFps(args.fps)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
maxFrameSize = cam.getPreviewHeight() * cam.getPreviewWidth() * 3
labelMap = ['bottle', 'can']
syncNN = True
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(args.net_blob_path)
nn.setNumInferenceThreads(2) # By default 2 threads are used
nn.setNumNCEPerInferenceThread(1) # By default, 1 NCE is used per thread

img_out = pipeline.create(dai.node.XLinkOut)
nn_out = pipeline.create(dai.node.XLinkOut)
img_out.setStreamName("rgb")
nn_out.setStreamName("nn")
# Linking
cam.preview.link(nn.input)
nn.out.link(nn_out.input)
if syncNN:
    nn.passthrough.link(img_out.input)
else:
    cam.preview.link(img_out.input)

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
# Properties
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
monoLeft.setResolution(monoResolution)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setFps(args.fps)
monoRight.setResolution(monoResolution)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setFps(args.fps)

depth = pipeline.create(dai.node.StereoDepth)
depth_out = pipeline.create(dai.node.XLinkOut)
depth_out.setStreamName("depth")
# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)

# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(depth_out.input)

config = depth.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 400
config.postProcessing.thresholdFilter.maxRange = 15000
config.postProcessing.decimationFilter.decimationFactor = 1
depth.initialConfig.set(config)

# # In this example we use 2 imageManips for splitting the original 1000x500
# # preview frame into 2 500x500 frames
# manip1 = pipeline.create(dai.node.ImageManip)
# manip1.initialConfig.setCropRect(0, 0, 0.5, 1)
# manip1.setMaxOutputFrameSize(maxFrameSize)
# cam.preview.link(manip1.inputImage)

# manip2 = pipeline.create(dai.node.ImageManip)
# manip2.initialConfig.setCropRect(0.5, 0, 1, 1)
# manip2.setMaxOutputFrameSize(maxFrameSize)
# cam.preview.link(manip2.inputImage)

# xout1 = pipeline.create(dai.node.XLinkOut)
# xout1.setStreamName('out1')
# manip1.out.link(xout1.input)

# xout2 = pipeline.create(dai.node.XLinkOut)
# xout2.setStreamName('out2')
# manip2.out.link(xout2.input)


# Upload the pipeline to the device
with dai.Device(pipeline) as device:
  # Print MxID, USB speed, and available cameras on the device
  print('MxId:',device.getDeviceInfo().getMxId())
  print('USB speed:',device.getUsbSpeed())
  print('Connected cameras:',device.getConnectedCameras())

  # Output queues will be used to get the rgb frames and nn data from the outputs defined above
  q_cam = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
  q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
  q_detph = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

  frame = None
  detections = []
  startTime = time.monotonic()
  counter = 0
  color2 = (255, 255, 255)

  # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
  def frameNorm(frame, bbox):
      normVals = np.full(len(bbox), frame.shape[0])
      normVals[::2] = frame.shape[1]
      return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

  def displayFrame(name, frame):
      color = (255, 0, 0)
      for detection in detections:
          bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
          cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
          cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
          cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
      # Show the frame
      cv2.imshow(name, frame)

  while True:

    if syncNN:
        cam_out = q_cam.get()
        nn_out = q_nn.get()
    else:
        cam_out = q_cam.tryGet()
        nn_out = q_nn.tryGet()

    if cam_out is not None:
        frame = cam_out.getCvFrame()
        cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                    (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)
        res = model(frame)
        res_plotted = res[0].plot()
        cv2.imshow("result", res_plotted)

    if nn_out is not None:
        layers = nn_out.getAllLayerNames()
        layer1Data = nn_out.getLayerFp16("Layer1_FP16")
        data = nn_out.getData()

    if frame is not None:
        displayFrame("rgb", frame)

    inDisparity = q_detph.tryGet()
    if inDisparity is not None:
        frame = inDisparity.getFrame()
        # Normalization for better visualization
        frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)
        displayFrame("depth", frame)

    if cv2.waitKey(1) == ord('q'):
        break
