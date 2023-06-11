import os
import shutil
import tqdm

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)

def process_image(image_path, label_path, target_image_path, target_label_path):
    if os.path.exists(target_image_path) and os.path.exists(target_label_path):
        return

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
        if object_id == 0 or object_id == 39:
            if object_id == 39:
                fields[0] = '1'  # Change object ID from 39 to 1
            filtered_lines.append(' '.join(fields))

    if len(filtered_lines)==0:
        return

    # Copy image file
    shutil.copyfile(image_path, target_image_path)

    # Save modified label file
    with open(target_label_path, 'w') as target_label_file:
        target_label_file.writelines(filtered_lines)

def process_folder(source_folder, target_folder):

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
                process_image(source_image_path, source_label_path, target_image_path, target_label_path)

def main():
# Main script
    source_folder = os.path.join(os.path.dirname(script_directory), 'datasets', 'coco')
    target_folder = os.path.join(os.path.dirname(script_directory), 'datasets', 'coco_custom')

    process_folder(source_folder, target_folder)

if __name__ == '__main__':
    main()
