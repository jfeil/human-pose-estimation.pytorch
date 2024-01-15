import os
import shutil
import zipfile
import json
import glob


def init_paths(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def create_subset(input_files, output_dir, mode, multiply=1, input_base_name='person_keypoints_',
                  output_base_name='person_keypoints_'):
    annotation_data = {}
    annotation_data['images'] = []
    annotation_data['annotations'] = []
    temp_dir = 'tmp'

    init_paths(os.path.join(output_dir, 'images', mode + '2017'))
    init_paths(temp_dir)
    for file_name in input_files:
        base_index = len(annotation_data['images'])

        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(temp_dir))

        with open(os.path.join(temp_dir, 'annotations', input_base_name + 'default.json')) as file:
            json_data = json.load(file)

        if not 'licenses' in annotation_data:
            annotation_data['licenses'] = json_data['licenses']
        if not 'info' in annotation_data:
            annotation_data['info'] = json_data['info']
        if not 'categories' in annotation_data:
            annotation_data['categories'] = json_data['categories']

        image_ids = sorted(set([x['image_id'] + base_index for x in json_data['annotations']]))
        for image in json_data['images']:
            if image['id'] + base_index not in image_ids:
                continue
            new_image = dict(image)
            new_image['id'] = len(annotation_data['images']) + 1
            new_name = f"{new_image['id']:012d}{os.path.splitext(new_image['file_name'])[1]}"
            shutil.move(os.path.join(temp_dir, 'images', new_image['file_name']),
                        os.path.join(output_dir, 'images', mode + '2017', new_name))
            new_image['file_name'] = new_name
            annotation_data['images'] += [new_image]
        for annotation in json_data['annotations']:
            new_annotation = dict(annotation)
            new_annotation['image_id'] = image_ids.index(new_annotation['image_id'] + base_index) + 1 + base_index
            new_annotation['id'] = len(annotation_data['annotations']) + 1
            annotation_data['annotations'] += [new_annotation]
    annotation_data['annotations'] *= multiply
    with open(os.path.join(output_dir, 'annotations', output_base_name + mode + '2017.json'), 'w+') as file:
        json.dump(annotation_data, file)


def create_dataset(input_dataset, output_dataset, multiply_train=10):
    init_paths(os.path.join(output_dataset, 'annotations'))
    for mode in [f.name for f in os.scandir(input_dataset) if f.is_dir()]:
        if mode == 'train':
            multiply = multiply_train
        else:
            multiply = 1
        input_files = glob.glob(os.path.join(input_dataset, mode, '*.zip'))
        create_subset(input_files, output_dataset, mode, multiply)