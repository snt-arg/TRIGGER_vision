from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from ultralytics import YOLO
import torch as T
from PIL import Image
import cv2


experiment = Experiment(
  api_key = "mfVQXEQ0maXqHp4pVrDZyxzS9",
  project_name = "YOLOv8m_segm_trash",
  workspace="claudiocimarelli",
  auto_output_logging='default',
)

# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "model": 'yolov8m-seg.pt',# path to model file, i.e. yolov8n.pt, yolov8n.yaml
    "data": 'data.yaml', # path to data file, i.e. coco128.yaml
    "epochs": 500, # number of epochs to train for
    "patience": 20, # epochs to wait for no observable improvement for early stopping of training
    "batch": 32, # number of images per batch (-1 for AutoBatch)
    "imgsz": 512, # size of input images as integer or w,h
    "save": True, # save train checkpoints and predict results
    "save_period": 20, # Save checkpoint every x epochs (disabled if < 1)
    "cache": True, # True/ram, disk or False. Use cache for data loading
    "device": 0, # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
    "workers": 8, # number of worker threads for data loading (per RANK if DDP)
    "project": 'bottle_segmentation', # project name
    "name": 'YOLOv8m-512',# experiment name, results saved to 'project/name' directory
    "exist_ok": False, # whether to overwrite existing experiment
    "pretrained": True, # whether to use a pretrained model
    "optimizer": 'AdamW', # optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    "verbose": True, # whether to print verbose output
    "seed": 0, # random seed for reproducibility
    "deterministic": True, # whether to enable deterministic mode
    "single_cls": False, # train multi-class data as single-class
    "rect": False, # rectangular training if mode='train' or rectangular validation if mode='val'
    "cos_lr": False, # use cosine learning rate scheduler
    "close_mosaic": 0, # (int) disable mosaic augmentation for final epochs
    "resume": False ,# resume training from last checkpoint
    "amp": True, # Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check
    "fraction": 1.0, # dataset fraction to train on (default is 1.0, all images in train set)
    "profile": False, # profile ONNX and TensorRT speeds during training for loggers
    "lr0": 0.004, # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    "lrf": 0.01, # final learning rate (lr0 * lrf)
    "momentum": 0.9, # SGD momentum/Adam beta1
    "weight_decay": 0.0005, # optimizer weight decay 5e-4
    "warmup_epochs": 2.0, # warmup epochs (fractions ok)
    # "warmup_momentum: 0.8 # warmup initial momentum
    # "warmup_bias_lr: 0.1 # warmup initial bias lr
    # "box: 7.5 # box loss gain
    # "cls: 0.5 # cls loss gain (scale with pixels)
    # "dfl: 1.5 # dfl loss gain
    # "pose: 12.0 # pose loss gain
    # "kobj: 1.0 # keypoint obj loss gain
    # "label_smoothing: 0.0 # label smoothing (fraction)
    # "nbs: 64 # nominal batch size
    # "hsv_h: 0.015 # image HSV-Hue augmentation (fraction)
    # "hsv_s: 0.7 # image HSV-Saturation augmentation (fraction)
    # "hsv_v: 0.4 # image HSV-Value augmentation (fraction)
    "degrees": 5, # image rotation (+/- deg)
    "translate": 0.1, # image translation (+/- fraction)
    "scale": 0.5, # image scale (+/- gain)
    "shear": 0.01, # image shear (+/- deg)
    # "perspective": 0.00001, # image perspective (+/- fraction), range 0-0.001
    "flipud": 0.5, # image flip up-down (probability)
    "fliplr": 0.5, # image flip left-right (probability)
    # "mosaic: 1.0 # image mosaic (probability)
    # "mixup: 0.0 # image mixup (probability)
    # "copy_paste: 0.0 # segment copy-paste (probability)
}
# experiment.log_parameters(hyper_params)

# Load a model
model = YOLO('yolov8m-seg.pt')  # load a pretrained model

model.train(**hyper_params)
# Evaluate the model's performance on the validation set
results = model.val()

# Seamlessly log your Pytorch model
# log_model(experiment, model.model, model_name="YoloV8")
