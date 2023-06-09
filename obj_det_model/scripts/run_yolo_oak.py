import depthai as dai
import cv2
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description='Run YOLOv8 on OAK')
parser.add_argument('--net_blob_path', type=str, required=True, help="Network BLOB path")
parser.add_argument('--image_size', type=int, default=416, help="Input size (HxW)")
args = parser.parse_args()

pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2022_1)

cam = pipeline.createColorCamera()
cam.setPreviewSize(args.image_size, args.image_size)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
order = cam.getColorOrder()
cam.setFps(15)
# cam.setInterleaved(False)
# cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
maxFrameSize = cam.getPreviewHeight() * cam.getPreviewWidth() * 3
labelMap = ['bottle', 'can']
syncNN = True
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(args.net_blob_path)

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
  qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
  qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

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
        inRgb = qRgb.get()
        inDet = qDet.get()
    else:
        inRgb = qRgb.tryGet()
        inDet = qDet.tryGet()

    if inRgb is not None:
        frame = inRgb.getCvFrame()
        cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                    (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

    if inDet is not None:
        detections = inDet.detections
        counter += 1

    if frame is not None:
        displayFrame("rgb", frame)

    if cv2.waitKey(1) == ord('q'):
        break
