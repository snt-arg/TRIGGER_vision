import os
import shutil
import tqdm
import argparse
import yaml
from ultralytics.yolo.utils.downloads import download
from pathlib import Path

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
default_ds_path = os.path.join(os.path.dirname(script_directory), 'datasets')

parser = argparse.ArgumentParser(description='YOLOv8 evaluation and test on image')
parser.add_argument('--dataset_path', type=str, default=default_ds_path, help="dataset path where to download COCO")
parser.add_argument('--target_folder', type=str, required=True, help="target folder under dataset_path")
parser.add_argument('--coco_classes', type=str, default='cup, bottle, apple', nargs="+", help="Filtered set of COCO classes passed as label names")


def process_image(image_path, label_path, target_image_path, target_label_path, selected_ids):
    lines = []
    with open(label_path, 'r') as label_file:
        lines = label_file.readlines()

    if len(lines)==0:
        return

    # Filter lines and modify object IDs
    filtered_lines = []
    for line in lines:
        fields = line.split(' ')
        object_id = int(fields[0])
        try:
            fields[0] = str(selected_ids.index(object_id))
            filtered_lines.append(' '.join(fields))
        except ValueError:
            continue

    if len(filtered_lines)==0:
        # sanity checks.
        if os.path.exists(target_image_path):
            os.remove(target_image_path)
        if os.path.exists(target_label_path):
            os.remove(target_label_path)
        return

    # Copy image file
    if not os.path.exists(target_image_path):
        shutil.copyfile(image_path, target_image_path)

    # Save modified label file
    with open(target_label_path, 'w') as f:
        f.writelines(filtered_lines)

def process_folder(source_folder, target_folder, selected_ids):

    for split_folder in ['val2017', 'train2017']:
        source_image_folder = os.path.join(source_folder, 'images', split_folder)
        source_label_folder = os.path.join(source_folder, 'labels', split_folder)
        target_image_folder = os.path.join(target_folder, 'images', split_folder)
        os.makedirs(target_image_folder, exist_ok=True)
        target_label_folder = os.path.join(target_folder, 'labels', split_folder)
        os.makedirs(target_label_folder, exist_ok=True)

        image_files = os.listdir(source_image_folder)
        for image_file in tqdm.tqdm(sorted(image_files), total=len(image_files)):
            image_id = image_file[:-4]  # Remove the '.jpg' extension
            label_file = image_id + '.txt'
            source_image_path = os.path.join(source_image_folder, image_file)
            source_label_path = os.path.join(source_label_folder, label_file)

            if os.path.exists(source_label_path):
                target_image_path = os.path.join(target_image_folder, image_file)
                target_label_path = os.path.join(target_label_folder, label_file)
                process_image(source_image_path, source_label_path, target_image_path, target_label_path, selected_ids)

        with open(os.path.join(target_folder, split_folder + '.txt'), 'w') as target_file:
            lines = [f'./images/{split_folder}/{f}\n' for f in os.listdir(target_image_folder)]
            target_file.writelines(lines)

def download_coco(ds_path, segments=True):

    ds_dir = Path(ds_path)/'coco'  # dataset root dir
    ds_dir.mkdir(parents=True, exist_ok=True)
    # Download labels
    if not (ds_dir.parent/'coco.yaml').exists():
        download('https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/datasets/coco.yaml')
    url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
    urls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels
    if not (ds_dir.parent/('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')).exists():
        download(urls, dir=ds_dir.parent)
    # Download data
    urls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
            'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images
            'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)
    if not all([(ds_dir.parent/u).exists() for u in urls]):
        download(urls, dir=ds_dir / 'images', threads=3)

def filter_yaml_names(file_path, dest_file_path, selected_names):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    name_mapping = {name: idx for idx, name in data['names'].items()}
    selected_ids = sorted([int(name_mapping[name]) for name in selected_names if name in name_mapping])

    data['names'] = {str(idx): name for idx, name in enumerate(selected_names)}

    with open(dest_file_path, 'w') as f:
        yaml.dump(data, f)

    return selected_ids

def main():
    args = parser.parse_args()

    download_coco(args.dataset_path)
    selected_ids = filter_yaml_names(os.path.join(args.dataset_path, 'coco.yaml'),
                                     os.path.join(args.dataset_path, args.target_folder+'.yaml'),
                                     args.coco_classes)
    source_folder = os.path.join(args.dataset_path, 'coco')
    target_folder = os.path.join(args.dataset_path, args.target_folder)
    process_folder(source_folder, target_folder, selected_ids)

if __name__ == '__main__':
    main()
