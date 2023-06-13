from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from ultralytics import YOLO
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='YOLOv8 train CLI script')
parser.add_argument('--model', type=str, default=argparse.SUPPRESS, help="Model checkpoint path or yaml file")
parser.add_argument('--net_size', type=str, choices=['m', 's', 'n'], default='s', help="Network size option")
parser.add_argument('--dataset', type=str, required=True, help="Dataset YAML file path or Ultralytics dataset, e.g., coco.yaml")
parser.add_argument('--input_size', type=int, default=640, help="Input image size (integer)")
parser.add_argument('--workers', type=int, default=4, help="Num workers (cpu threads)")
parser.add_argument('--epochs', type=int, default=100, help="Num of epochs")
parser.add_argument('--save_period', type=int, default=50, help="Epochs before saving (-1 to disable)")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser.add_argument('--patience', type=int, default=10, help="Patience before early stopping")
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'], help="Optimize Algorithm")
parser.add_argument('--lr0', type=float, default=1e-3, help="Init learning rate")
parser.add_argument('--lrf', type=float, default=1e-2, help="Final learning rate (lr0 * lrf)")
parser.add_argument('--lr_decay', type=float, default=1e-4, help="Learning rate constant decay rate")
parser.add_argument('--momentum', type=float, default=.9, help="Lr Momemntum")
parser.add_argument('--warmup_epochs', type=float, default=0., help="Warmup epochs number (fraction ok)")
parser.add_argument('--warmup_momentum', type=float, default=.8, help="Warmup lr momentum")
parser.add_argument('--warmup_bias', type=float, default=1e-3, help="Warmup lr bias")
parser.add_argument('--degrees', type=int, default=30, help="Max degress of rotation for image augmentation")
parser.add_argument('--mosaic', type=float, default=0, help="Probability of mosaic augmentation")
parser.add_argument('--translate', type=float, default=.1, help="Image translation (+/- fraction)")
parser.add_argument('--resume', action='store_true', help="Resume training from last checkpoint")
parser.add_argument('--no_pretrain', action='store_true', help="Pretrained weights initialization")
parser.add_argument('--device', type=int, default=0, help="ID CUDA device")
parser.add_argument('--api_key', type=str, default=argparse.SUPPRESS, help="COMETML API Key (personal use)")

args = parser.parse_args()

net_size = args.net_size
input_size = args.input_size

model_name = f'yolov8{net_size}-seg'
experiment = None
if hasattr(args, 'imgsz'):
  try:
    experiment = Experiment(
      api_key = args.api_key,
      project_name = f"YOLOV8{net_size}-segm-{input_size}_bottle-person",
      workspace="claudiocimarelli",
      auto_output_logging='default',
    )
  except Exception as e:
    print(e)
    print ('Comet experiment disabled')

# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "model": args.model if hasattr(args, 'model') else f"{model_name}.pt",
    "data": args.dataset, # path to data file, i.e. coco128.yaml
    "epochs": args.epochs, # number of epochs to train for
    "patience": args.patience, # epochs to wait for no observable improvement for early stopping of training
    "batch": args.batch_size, # number of images per batch (-1 for AutoBatch)
    "imgsz": input_size, # size of input images as integer or w,h
    "save": True, # save train checkpoints and predict results
    "save_period": args.save_period, # Save checkpoint every x epochs (disabled if < 1)
    "cache": True, # True/ram, disk or False. Use cache for data loading
    "device": args.device, # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
    "workers": args.workers, # number of worker threads for data loading (per RANK if DDP)
    "project": 'bottle_segmentation', # project name
    "name": f"{model_name}_{input_size}_{Path(args.dataset).stem}",# experiment name, results saved to 'project/name' directory
    "exist_ok": False, # whether to overwrite existing experiment
    "pretrained": not args.no_pretrain, # whether to use a pretrained model
    "optimizer": args.optimizer, # optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    "verbose": True, # whether to print verbose output
    "seed": 0, # random seed for reproducibility
    "deterministic": True, # whether to enable deterministic mode
    "single_cls": False, # train multi-class data as single-class
    "rect": False, # rectangular training if mode='train' or rectangular validation if mode='val'
    "cos_lr": False, # use cosine learning rate scheduler
    "close_mosaic": 0, # (int) disable mosaic augmentation for final epochs
    "resume": args.resume ,# resume training from last checkpoint
    "amp": True, # Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check
    "fraction": 1.0, # dataset fraction to train on (default is 1.0, all images in train set)
    "profile": False, # profile ONNX and TensorRT speeds during training for loggers
    "lr0": args.lr0, # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    "lrf": args.lrf, # final learning rate (lr0 * lrf)
    "momentum": args.momentum, # SGD momentum/Adam beta1
    "weight_decay": args.lr_decay, # optimizer weight decay 5e-4
    "warmup_epochs": args.warmup_epochs, # warmup epochs (fractions ok)
    "warmup_momentum": args.warmup_momentum, # warmup initial momentum
    "warmup_bias_lr": args.warmup_bias, # warmup initial bias lr
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
    "degrees": args.degrees, # image rotation (+/- deg)
    "translate": args.translate, # image translation (+/- fraction)
    "scale": 0.5, # image scale (+/- gain)
    "shear": 0.01, # image shear (+/- deg)
    # "perspective": 0.00001, # image perspective (+/- fraction), range 0-0.001
    "flipud": 0.5, # image flip up-down (probability)
    "fliplr": 0.5, # image flip left-right (probability)
    "mosaic": args.mosaic # image mosaic (probability)
    # "mixup: 0.0 # image mixup (probability)
    # "copy_paste: 0.0 # segment copy-paste (probability)
}

if experiment is not None:
  experiment.log_parameters(hyper_params)

# Load a model
model = YOLO(args.model if hasattr(args, 'model') else f"{model_name}.pt")  # load a pretrained model
model.train(**hyper_params)

# Seamlessly log your Pytorch model
# log_model(experiment, model.model, model_name="YoloV8")
