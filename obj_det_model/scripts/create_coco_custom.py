import os
import shutil
import tqdm

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)

def process_image(image_path, label_path, target_image_path, target_label_path):
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
            fields[0] = str([0, 39, 40, 41, 45, 47, 75].index(object_id))
            filtered_lines.append(' '.join(fields))
        except ValueError:
            continue

    if len(filtered_lines)==0:
        return

    # Copy image file
    if not os.path.exists(target_image_path):
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

        with open(os.path.join(target_folder, split_folder + '.txt'), 'w') as target_file:
            lines = [f'./images/{split_folder}/{f}\n' for f in os.listdir(target_image_folder)]
            target_file.writelines(lines)

def main():
# Main script
    source_folder = os.path.join(os.path.dirname(script_directory), 'datasets', 'coco')
    target_folder = os.path.join(os.path.dirname(script_directory), 'datasets', 'coco_custom')

    process_folder(source_folder, target_folder)

if __name__ == '__main__':
    main()
