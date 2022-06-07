from research.fer.dataset_modules.dataset_functions_imbased import sbir_border
import torchvision.transforms as transforms
from argusutil.annotation import annotation
from argusutil.task.dal.tools import ToolingDAO
from argusutil.task.context import simple_context
import json
import cv2
import numpy as np
from PIL import Image
import torchvision
import os
from pathlib import Path



color_palette = ['pink', 'purple', 'deep-purple', 'indigo', 'blue', 'light-blue', 'cyan', 'teal', 'green', 'yellow',
                 'orange', 'brown', 'grey', 'lime']

keypoints_indices = {'mouth': '32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48',
                     'eyes': '0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30'}


def convert_applaud_truth_table_to_annotation(truth_table, start_frame):
    current_annotation = annotation.AnnotationOfIntervals(annotation.Unit.INDEX)
    current_status = 'start'
    for it in range(len(truth_table)):
        if truth_table[it] == 1 and current_status == 'start':
            current_start = it + start_frame
            current_status = 'stop'
        if truth_table[it] == 0 and current_status == 'stop':
            current_stop = it + start_frame
            current_annotation.add(annotation.Interval(current_start, current_stop))
            current_status = 'start'
    return current_annotation


def convert_indexed_annotation_to_ms_annotation(current_annotation, data_id):
    pipeline_db_path = 'postgresql://luigi@localhost:5000/sbir_eval_2021'
    rgb_param_family = 'ParseRGBVideoRecord'
    with simple_context(pipeline_db_path) as ctx:
        dao = ToolingDAO(ctx.session())

        rgb_parameters = dao.get_tasks_by_uuid_and_family(data_id, rgb_param_family)[0].task.loadOutput()
        current_record_length = len(rgb_parameters['rgb_video_record'].timestamps)
        frame_time_stamps = rgb_parameters['rgb_video_record'].timestamps

    current_ms_time_stamps = [int(ts * 1000) for ts in frame_time_stamps]
    current_transform_index_to_ms = annotation.indexedToMillisecondAnnotationTransform(current_ms_time_stamps)

    ms_annotation = current_transform_index_to_ms(current_annotation)
    return ms_annotation


def write_error_images(imgs, y_true, y_pred, data_id, ref_pos, annot_indices, au_index=0, transform=None):

    rootdir = Path('error_images')
    fp = rootdir / str(annot_indices[au_index]) / 'FP'
    fp.mkdir(parents=True, exist_ok=True)
    tp = rootdir / str(annot_indices[au_index]) / 'TP'
    tp.mkdir(parents=True, exist_ok=True)
    fn = rootdir / str(annot_indices[au_index]) / 'FN'
    fn.mkdir(parents=True, exist_ok=True)

    for i in range(y_true.shape[0]):
        if isinstance(imgs, list):
            img = imgs[0][i]
        else:
            img = imgs[i]

        if transform:
            img = transform(img)
        img = img.permute((1, 2, 0))

        if y_pred[i, au_index] == 1 and y_true[i, au_index] != y_pred[i, au_index]:
            folder_name = 'error_images/' + str(annot_indices[au_index]) + '/FP/'
            file_name = folder_name + data_id[i] + '_' + str(ref_pos[i]) + '.jpg'
            cv2.imwrite(file_name, img.cpu().numpy() * 255)

        if y_pred[i, au_index] == 1 and y_true[i, au_index] == y_pred[i, au_index]:
            folder_name = 'error_images/' + str(annot_indices[au_index]) + '/TP/'
            file_name = folder_name + data_id[i] + '_' + str(ref_pos[i]) + '.jpg'
            cv2.imwrite(file_name, img.cpu().numpy() * 255)

        if y_pred[i, au_index] == 0 and y_true[i, au_index] != y_pred[i, au_index]:
            folder_name = 'error_images/' + str(annot_indices[au_index]) + '/FN/'
            file_name = folder_name + data_id[i] + '_' + str(ref_pos[i]) + '.jpg'
            cv2.imwrite(file_name, img.cpu().numpy() * 255)


def create_grid_from_images(list_of_images, target, annot_indices):

    #list_of_images = [cv2.putText(list_of_images[i], "GT: AU: " + ",".join(map(str, np.array(annot_indices)[target[i].astype('int')==1])),
    #                              (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255) for i in range(list_of_images.shape[0])]

    #list_of_images2 = [list_of_images[i] for i in range(list_of_images.shape[0])]
    transform = inv_transform
    list_of_images2 = [transform(Image.fromarray(np.uint8(img))) for img in list_of_images]

    return torchvision.utils.make_grid(list_of_images2)


def convert_to_annotation(prediction, threshold, person_id):
    annotation_path_root = '/data/emotion/annotations/'
    AU_INDICES = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
    prediction_vectors = {}

    for it in range(len(prediction)):
        for ij in range(len(prediction[it])):
            for idx, current_au in enumerate(AU_INDICES):
                if current_au not in prediction_vectors.keys():
                    prediction_vectors[current_au] = []
                if prediction[it][ij][idx] > threshold:
                    prediction_vectors[current_au].append(1)
                else:
                    prediction_vectors[current_au].append(0)
    col_idx = 0
    output_annot_json = {}
    for au_key in prediction_vectors.keys():
        current_prediction_annotation = convert_applaud_truth_table_to_annotation(prediction_vectors[au_key],
                                                                                  sbir_border[person_id][0])
        current_ms_prediction_annotation = convert_indexed_annotation_to_ms_annotation(current_prediction_annotation,
                                                                                       person_id)
        serialized_indexed_extracted_annotation = annotation_serialize_conversion(current_ms_prediction_annotation)
        output_dict_descriptors = {}

        output_dict_descriptors['name'] = au_key
        output_dict_descriptors['color'] = color_palette[col_idx]
        output_dict_descriptors['data'] = serialized_indexed_extracted_annotation
        output_annot_json[str(col_idx)] = output_dict_descriptors
        col_idx += 1
    with open(annotation_path_root + person_id + '_annotation.json', 'w') as f:
        json.dump(output_annot_json, f)


def annotation_serialize_conversion(annotation):
    serialized_annotation = {}
    serialized_annotation['unit'] = 'MILLISECOND'
    serialized_annotation['intervals'] = []
    for it in range(len(annotation)):
        current_interval = {}
        current_interval['start'] = annotation[it].start
        current_interval['stop'] = annotation[it].stop
        serialized_annotation['intervals'].append(current_interval)
    return serialized_annotation


# advanced preprocessing for ResNet
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
    ], p=0.7),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

inv_transform = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                    std=[1., 1., 1.])])
