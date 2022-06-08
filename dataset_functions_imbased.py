import cv2
import numpy as np
import re
import pickle
import bisect
import random
import pickle as pkl
from PIL import Image
from typing import Dict

from emotion import EM_INDICES
from database_functions import DatabaseManagement
from jittering import jitter_points

random.seed(823)
# dict for start, stop frames
sbir_border = {
    '00ad44fd-cec2-446b-85b8-7922d172a45d': [15000, 29999],
    '2c0930ad-46c6-4a7a-9720-8736487e7b7b': [29988, 44979],
    '81b5da37-311e-4f94-a1e6-8f66486ec483': [37489, 52485],
    'b21e9c66-561a-48f8-9611-ab69c811fb35': [50965, 65965],
    '872fd1cf-a43b-4e1e-a998-db4f8e93ad9a': [41965, 56961],
    '4de55bad-84d9-4f51-a199-87410ee38a7f': [55958, 70958]
}

# fer2013
train_subjs_ids = ['2F01', '2F03', '2F05', '2F07', '2F09', '2F11', '2F13',
                   '2F15', '2F17', '2F19', '2F21', '2F23', '2M01', '2M03',
                   '2M05', '2M07', '2M09', '2M11', '2M13', '2M15', '2M17']

all_subjs_ids = ['2F01', '2F02', '2F03', '2F04', '2F05', '2F06', '2F07', '2F08', '2F09', '2F10', '2F11', '2F12',
                 '2F13', '2F14', '2F15', '2F16', '2F17', '2F18', '2F19', '2F20', '2F21', '2F22', '2F23',
                 '2M01', '2M02', '2M03', '2M04', '2M05', '2M06', '2M07', '2M08', '2M09', '2M10', '2M11',
                 '2M12', '2M13', '2M14', '2M15', '2M16', '2M17', '2M18']
                 
ddcf_valid_ids = ['S122', 'S11', 'S26', 'S33']

childefes_valid_ids = ['35', '32']

class AuDataManagementPrototype():
    def __init__(self, db_path : str, current_dataset : str, data_type : str, annot_type : str, way_of_split : float,
                annot_indices, list_of_data_needed : Dict[str,str], transform = None):
        self.DM = DatabaseManagement(db_path)
        data_ids = self.DM.get_ids_by_dataset_name(current_dataset)
        self.annot_type = annot_type
        self.current_dataset = current_dataset
        self.list_of_data_needed = list_of_data_needed
        self.annot_indices = annot_indices
        self.transform = transform

        if data_type == 'train':
            self.data_ids = [video for video in data_ids if video[0:video.find('_')] in train_subjs_ids]  #fera2015

        if data_type == 'test':
            self.data_ids = [video for video in data_ids if video[0:video.find('_')] not in train_subjs_ids]  #fera2015

        if data_type == 'train_split':
            self.data_ids = data_ids[:round(len(data_ids) * way_of_split)]  # regular split

        if data_type == 'test_split':
            self.data_ids = data_ids[round(len(data_ids) * way_of_split):]  # regular split

        if data_type == 'cv':
            self.data_ids = np.array([video for video in data_ids if video[0:video.find('_')]
                                      in np.array(all_subjs_ids)[way_of_split]])  # cross-validation split
        if data_type == 'cv_sbir':
            self.data_ids = np.array(data_ids)[way_of_split]
                                      
        if data_type == 'idx':
            self.data_ids = np.array(data_ids)[way_of_split:way_of_split+1] #using one

        if data_type == 'no_idx':  #using all except one
            self.data_ids = np.concatenate([np.array(data_ids)[:way_of_split-1], np.array(data_ids)[way_of_split:]])

        self.dataset_length = 0
        self.dataset_intervals = []
        self.uuids = []
        self.annots = []
        self.positive_uuids = []
        self.positive_annots = []
        self.negative_uuids = []
        self.negative_annots = []

        for data_id in self.data_ids:
            data_uuids = self.DM.get_resource_ids_by_data_id(data_id)

            valid_data_uuids = []
            for uuid in data_uuids:

                if current_dataset == 'SBIR':
                    annots = self.DM.get_annotation_by_resource_id(uuid)

                    if sbir_border[data_id][0] <= self.DM.get_resource_by_resource_id(uuid).reference_position <= \
                            sbir_border[data_id][1]:
                        # <=2 because annotations are doubled in the db
                        if '50' in [re.sub("[^0-9]", "", annot.name) for annot in annots] and \
                                len([re.sub("[^0-9]", "", annot.name) for annot in annots]) <= 2:
                            continue

                        valid_data_uuids.append(uuid)
                else:
                    valid_data_uuids.append(uuid)
                
            self.dataset_length += len(valid_data_uuids)
            self.dataset_intervals.append(self.dataset_length)
            self.uuids.append(valid_data_uuids)
            annots = []

            for data_uuid in valid_data_uuids:
                annot = self.DM.get_annotation_by_resource_id(data_uuid)
                annots.append(annot)
                if len(annot_indices) == 1:
                    if str(annot_indices[0]) in [annot_au.name for annot_au in annot]:
                        self.positive_annots.append(annot)
                        self.positive_uuids.append(data_uuid)
                    else:
                        self.negative_annots.append(annot)
                        self.negative_uuids.append(data_uuid)

            self.annots.append(annots)
        print(len(self.negative_uuids) / (len(self.positive_uuids)+0.00001))
        print(len(self.positive_uuids))

        ######################################JITTERING#################################################################
        if 'jittering' in self.list_of_data_needed.keys(): # do we want to use this every time?
            self.skew = len(self.negative_uuids) / (len(self.positive_uuids)+0.00001)
            self.real_dataset_length = self.dataset_length
            if self.skew > 1:
                self.dataset_length = self.dataset_length + round((self.skew-1)*len(self.positive_uuids))

            if self.skew < 1 and self.skew != 0:
                self.dataset_length = self.dataset_length + round((1/self.skew - 1) * len(self.negative_uuids))


    def load_data_to_memory(self, idx):
        if 'jittering' in self.list_of_data_needed.keys() and self.real_dataset_length <= idx:
            if self.skew > 1:
                jitter_index = random.randint(0, len(self.positive_uuids)-1)
                uuid = self.positive_uuids[jitter_index]
                annots = self.positive_annots[jitter_index]
            else:
                jitter_index = random.randint(0, len(self.negative_uuids)-1)
                uuid = self.negative_uuids[jitter_index]
                annots = self.negative_annots[jitter_index]

        else:
            data_idx = bisect.bisect_right(self.dataset_intervals, idx)
            # obtain the frame index
            if data_idx == 0:
                frame_idx = idx
            else:
                frame_idx = idx - self.dataset_intervals[data_idx - 1]

            # reading the frame
            uuid = self.uuids[data_idx][frame_idx]
            annots = self.annots[data_idx][frame_idx]

        data_list = []
        if self.current_dataset == 'SBIR':
            all_keypoints = pkl.loads(self.DM.get_feature_by_resource_id(uuid, 'extracted_facial_positions_2d_v1_2022_04_21_no_conf').data)
        else:
            all_keypoints = pkl.loads(self.DM.get_feature_by_resource_id(uuid, 'extracted_facial_positions_2d_v1_2022_03_26').data)

        ##Normalization
        all_keypoints = self.normalize(all_keypoints)
        
        if 'keypoints' in self.list_of_data_needed.keys():
            positions = self.list_of_data_needed['keypoints']
            if len(positions) == 0:
                chosen_keypoints = all_keypoints[:, :2]
            else:
                positions = [int(pos) for pos in positions.split(',')]
                chosen_keypoints = all_keypoints[positions, :2]

            data_list.append(chosen_keypoints)

        if 'image' in self.list_of_data_needed.keys():
            resource = self.DM.get_resource_by_resource_id(uuid)
            image = pickle.loads(resource.data)
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE).astype(np.float64)
            if 'jittering' in self.list_of_data_needed.keys() and self.real_dataset_length <= idx:
                image = jitter_points(image, (0, 0, image.shape[0], image.shape[1]), random.randint(0, 76))
            data_list[0:0] = [image]

        if 'edge_information' in self.list_of_data_needed.keys():
            current_feature = self.DM.get_feature_by_resource_id(uuid, self.list_of_data_needed['edge_information'])
            current_feature_data = pickle.loads(current_feature.data)
            G = cv2.imdecode(current_feature_data, cv2.IMREAD_GRAYSCALE).astype(np.float64)
            data_list[0:0] = [G]

        if 'p_m_diffusion_image' in self.list_of_data_needed.keys():
            current_feature = self.DM.get_feature_by_resource_id(uuid, self.list_of_data_needed['p_m_diffusion_image'])
            current_feature_data = pickle.loads(current_feature.data)
            image_blurred = cv2.imdecode(current_feature_data, cv2.IMREAD_GRAYSCALE).astype(np.float64)
            data_list[0:0] = [image_blurred]

        if 'derived_metrics' in self.list_of_data_needed.keys():
            all_derived_metrics = self.list_of_data_needed['derived_metrics'].split(',')
            for derived_metric in all_derived_metrics:
                derived_feature = self.DM.get_feature_by_resource_id(uuid, derived_metric).data
                data_list.append(pkl.loads(derived_feature))


        if 'data_type' in self.list_of_data_needed.keys():
            if self.list_of_data_needed['data_type'] == 'rgb':
                current_input = self.process_image_to_rgb(data_list, all_keypoints)
            elif self.list_of_data_needed['data_type'] == 'array':
                current_input = self.merge_input_data(data_list)
            elif self.list_of_data_needed['data_type'] == 'tuple': # currently works with 1 rgb and one or more array feature
                current_input = self.create_tuple_for_mixed_resnet(data_list)
        else:
            raise ValueError('data_type must be defined in list_of_data_needed dictionary.')

        if self.transform:
            if self.list_of_data_needed['data_type'] == 'tuple':
                current_input_first = Image.fromarray(np.uint8(current_input[0]))
                current_input_second = np.array(current_input[1].reshape(-1,), dtype=np.float32) ##shape (x,) needed
                current_input_first = self.transform(current_input_first)
                current_input_first = np.array(current_input_first, dtype=np.float32)
                current_input = (current_input_first, current_input_second)
            else:
                current_input = Image.fromarray(np.uint8(current_input))
                current_input = self.transform(current_input)
                current_input = np.array(current_input, dtype=np.float32)

        else:
            current_input = np.array(current_input, dtype=np.float32)

        # RETRIVE ANNOTATION
        # init the annots
        y = {}
        for AU in self.annot_indices:
            y[str(AU)] = 0

        y = load_annotation(annots, y)
        return current_input, y

    def process_image_to_rgb(self, current_data_list, keypoints):
        
        if len(current_data_list) == 1:
            current_input = current_data_list[0]
            if 'keypoints_image_merge' in self.list_of_data_needed.keys():
                if self.list_of_data_needed['keypoints'] == '':
                    used_keypoints = [i for i in range(len(keypoints))]
                else:
                    used_keypoints = self.list_of_data_needed['keypoints'].split(',')
                    used_keypoints = [int(keypoint) for keypoint in used_keypoints]
                for it in range(len(used_keypoints)):
                    current_input[int(keypoints[it, 1]), int(keypoints[it, 0])] = 255

            if 'crop_image' in self.list_of_data_needed.keys():
                current_input = self.crop_image(current_input, keypoints)
            current_input = self.create_3channel_image(current_input)
            
        elif len(current_data_list) == 3:
            if 'keypoints' in self.list_of_data_needed.keys():
                if self.list_of_data_needed['keypoints'] == '':
                    used_keypoints = [i for i in range(len(keypoints))]
                else:
                    used_keypoints = self.list_of_data_needed['keypoints'].split(',')
                    used_keypoints = [int(keypoint) for keypoint in used_keypoints]
                for it in range(len(current_data_list)):
                    for ij in range(len(used_keypoints)):
                        current_data_list[it][int(keypoints[ij, 1]), int(keypoints[ij, 0])] = 255
                        
            if 'crop_image' in self.list_of_data_needed.keys():
                for it in range(len(current_data_list)):
                    current_data_list[it] = self.crop_image(current_data_list[it], keypoints)
            current_input = np.concatenate(current_data_list)
        return current_input

    def merge_input_data(self, current_data_list):
        for it in range(len(current_data_list)):
            if len(current_data_list[it].shape) > 1:
                current_data_list[it] = np.concatenate(current_data_list[it])
        return np.concatenate(current_data_list, axis=0)

    def normalize(self, all_keypoints):
        # 1st normalization
        all_keypoints = all_keypoints[:, :2] - all_keypoints[13:14, :2]
        # 2nd normalization
        all_keypoints[0:5, :] = all_keypoints[0:5, :] - (
                    (all_keypoints[19:20, :] + all_keypoints[22:23, :]) / 2)  # reyebrow
        all_keypoints[5:10, :] = all_keypoints[5:10, :] - (
                    (all_keypoints[25:26, :] + all_keypoints[28:29, :]) / 2)  # leyebrow
        all_keypoints[19:25, :] = all_keypoints[19:25, :] - (
                    (all_keypoints[19:20, :] + all_keypoints[22:23, :]) / 2)  # reye
        all_keypoints[25:31, :] = all_keypoints[25:31, :] - (
                    (all_keypoints[25:26, :] + all_keypoints[28:29, :]) / 2)  # leye
        all_keypoints[32:, :] = all_keypoints[32:, :] - ((all_keypoints[32:33, :] + all_keypoints[38:39, :]) / 2)# mouth

        return all_keypoints

    def create_3channel_image(self, data):
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
        if data.shape[2] == 1:
            data = np.repeat(data, 3, 2)
        return data

    def create_tuple_for_mixed_resnet(self, data_list):
        rgb_image = self.create_3channel_image(data_list[0])
        if len(data_list) > 2:
            additional_data = self.merge_input_data(data_list[1:])
        else:
            additional_data = data_list[1]
        return (rgb_image, additional_data)

    def get_feature_data(self, uuid, feature_name):
        feature = self.DM.get_feature_by_resource_id(uuid, feature_name)
        return pkl.loads(feature)

    def crop_image(self, current_image, keypoints):
        cut_coord_positions = self.list_of_data_needed['crop_image'].split(',')
        ledge = np.round(keypoints[cut_coord_positions[0], :2])
        redge = np.round(keypoints[cut_coord_positions[1], :2])
        if ledge[0] > redge[0] or ledge[1] > redge[1]:
            box = (0, 0, current_image.shape[1], current_image.shape[0])
        else:
            box = (ledge[0] - 40, ledge[1] - 40, redge[0] + 40, redge[1] + 60)

        current_image = Image.fromarray(np.uint8(current_image))
        current_image = np.array(current_image.crop(box))
        return current_image

    def get_dataid_refpos(self, idx):
        data_id = bisect.bisect_right(self.dataset_intervals, idx)
        if data_id == 0:
            frame_idx = idx
        else:
            frame_idx = idx - self.dataset_intervals[data_id - 1]

        uuid = self.uuids[data_id][frame_idx]
        ref_pos = self.DM.get_resource_by_resource_id(uuid).reference_position
        return self.data_ids[data_id], ref_pos



class AuDataManagement():  # ApplaudDataManipulation
    def __init__(self, db_path: str, current_dataset: str, data_type: str, annot_type: str, way_of_split, jittering, annot_indices):
        self.DM = DatabaseManagement(db_path)
        data_ids = self.DM.get_ids_by_dataset_name(current_dataset)

        #####Initializing
        self.annot_indices = annot_indices
        self.annot_type = annot_type
        self.current_dataset = current_dataset

        if data_type == 'train':
            self.data_ids = [video for video in data_ids if video[0:video.find('_')] in train_subjs_ids]  #fera2015

        if data_type == 'test':
            self.data_ids = [video for video in data_ids if video[0:video.find('_')] not in train_subjs_ids]  #fera2015

        if data_type == 'train_split':
            self.data_ids = data_ids[:round(len(data_ids) * way_of_split)]  # regular split

        if data_type == 'test_split':
            self.data_ids = data_ids[round(len(data_ids) * way_of_split):]  # regular split

        if data_type == 'cv':
            self.data_ids = np.array([video for video in data_ids if video[0:video.find('_')]
                                      in np.array(all_subjs_ids)[way_of_split]])  # cross-validation split

        if data_type == 'idx':
            self.data_ids = np.array(data_ids)[way_of_split:way_of_split+1] #using one

        if data_type == 'no_idx':  #using all except one
            self.data_ids = np.concatenate([np.array(data_ids)[:way_of_split-1], np.array(data_ids)[way_of_split:]])

        self.dataset_length = 0
        self.dataset_intervals = []
        self.uuids = []
        self.annots = []
        no_positive = 0
        no_negative = 0
        self.positive_uuids = []
        self.positive_annots = []
        self.negative_uuids = []
        self.negative_annots = []

        for data_id in self.data_ids:
            data_uuids = self.DM.get_resource_ids_by_data_id(data_id)

            valid_data_uuids = []
            for uuid in data_uuids:

                if current_dataset == 'SBIR':
                    annots = self.DM.get_annotation_by_resource_id(uuid)

                    if sbir_border[data_id][0] <= self.DM.get_resource_by_resource_id(uuid).reference_position <= \
                            sbir_border[data_id][1]:
                        # <=2 because annotations are doubled in the db
                        if '50' in [re.sub("[^0-9]", "", annot.name) for annot in annots] and \
                                len([re.sub("[^0-9]", "", annot.name) for annot in annots]) <= 2:
                            continue

                        valid_data_uuids.append(uuid)
                else:
                    valid_data_uuids.append(uuid)
                
            self.dataset_length += len(valid_data_uuids)
            self.dataset_intervals.append(self.dataset_length)
            self.uuids.append(valid_data_uuids)
            annots = []

            for data_uuid in valid_data_uuids:
                annot = self.DM.get_annotation_by_resource_id(data_uuid)
                annots.append(annot)
                if len(annot_indices) == 1:
                    if str(annot_indices[0]) in [annot_au.name for annot_au in annot]:
                        no_positive += 1
                        self.positive_annots.append(annot)
                        self.positive_uuids.append(data_uuid)
                    else:
                        no_negative += 1
                        self.negative_annots.append(annot)
                        self.negative_uuids.append(data_uuid)

            self.annots.append(annots)

        ######################################JITTERING#################################################################
        self.skew = no_negative / (no_positive+0.00001)
        self.real_dataset_length = self.dataset_length
        if self.skew > 1 and (data_type == 'train' or data_type == 'train_split') and jittering:
            self.dataset_length = self.dataset_length + round((self.skew-1)*no_positive)

        if self.skew < 1 and (data_type == 'train' or data_type == 'train_split') and jittering:
            self.dataset_length = self.dataset_length + round((1/self.skew - 1) * no_negative)

        #print(self.skew)

    # get information from db
    def load_data_to_memory(self, idx):

        if self.real_dataset_length <= idx:
            if self.skew > 1:
                jitter_index = random.randint(0, len(self.positive_uuids)-1)
                uuid = self.positive_uuids[jitter_index]
                annots = self.positive_annots[jitter_index]
            else:
                jitter_index = random.randint(0, len(self.negative_uuids)-1)
                uuid = self.negative_uuids[jitter_index]
                annots = self.negative_annots[jitter_index]

            data_idx = 0 ##not significant
        else:
            data_idx = bisect.bisect_right(self.dataset_intervals, idx)
            # obtain the frame index
            if data_idx == 0:
                frame_idx = idx
            else:
                frame_idx = idx - self.dataset_intervals[data_idx - 1]

            # reading the frame
            uuid = self.uuids[data_idx][frame_idx]
            annots = self.annots[data_idx][frame_idx]

        resource = self.DM.get_resource_by_resource_id(uuid)

        # RETRIEVE THE DATA
        image = pickle.loads(resource.data)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE).astype(np.float64)

        current_feature = self.DM.get_feature_by_resource_id(uuid, 'perona_malik_diffusion_2022_04_22')
        current_feature_data = pickle.loads(current_feature.data)
        G = cv2.imdecode(current_feature_data, cv2.IMREAD_GRAYSCALE).astype(np.float64)

        current_feature = self.DM.get_feature_by_resource_id(uuid, 'perona_malik_only_diffusion_2022_04_26')
        current_feature_data = pickle.loads(current_feature.data)
        image_blurred = cv2.imdecode(current_feature_data, cv2.IMREAD_GRAYSCALE).astype(np.float64)

        if self.real_dataset_length <= idx:
            image = jitter_points(image, (0, 0, image.shape[0], image.shape[1]), random.randint(0, 76))

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            G = G[:, :, np.newaxis]
            image_blurred = image_blurred[:, :, np.newaxis]

        if image.shape[2] == 1:
            image = np.repeat(image, 3, 2)
            G = np.repeat(G, 3, 2)
            image_blurred = np.repeat(image_blurred, 3, 2)

        # facial keypoints
        if self.current_dataset == 'SBIR':
            feature = self.DM.get_feature_by_resource_id(uuid, 'extracted_facial_positions_2d_v1_2022_04_21_no_conf')
        else:
            feature = self.DM.get_feature_by_resource_id(uuid, 'extracted_facial_positions_2d_v1_2022_03_26')
        feature = pickle.loads(feature.data)

        # RETRIVE ANNOTATION
        # init the annots
        y = {}
        for AU in self.annot_indices:
            y[str(AU)] = 0

        y = load_annotation(annots, y)

        return image, G, image_blurred, feature, y, resource.reference_position, self.data_ids[data_idx]


class EmotionDataManagement():
    def __init__(self, db_path: str, current_dataset: str, data_type: str, annot_type: str, way_of_split):
        self.DM = DatabaseManagement(db_path)
        data_ids = self.DM.get_ids_by_dataset_name(current_dataset)
        
        #####Initializing
        self.annot_type = annot_type
        self.current_dataset = current_dataset

        if data_type == 'train_split':
            if self.current_dataset == 'DDCF':
                self.data_ids = [current_id for current_id in data_ids if current_id.split('_')[0] not in ddcf_valid_ids]
            else:
                self.data_ids = [current_id for current_id in data_ids if current_id.split('_')[1] not in childefes_valid_ids]

        if data_type == 'valid_split':
            if self.current_dataset == 'DDCF':
                self.data_ids = [current_id for current_id in data_ids if current_id.split('_')[0] in ddcf_valid_ids]
            else:
                self.data_ids = [current_id for current_id in data_ids if current_id.split('_')[1] in childefes_valid_ids]

        self.dataset_length = 0
        self.dataset_intervals = []
        self.uuids = []
        self.annots = []

        for data_id in self.data_ids:
            data_uuids = self.DM.get_resource_ids_by_data_id(data_id)

            valid_data_uuids = []

            for uuid in data_uuids:
                valid_data_uuids.append(uuid)

            """
                if current_dataset == 'DDCF':
                    current_annot = self.DM.get_annotation_by_resource_id(uuid)
                    
                    if current_annot[0].name in EM_INDICES:
                        valid_data_uuids.append(uuid)
                    
                if current_dataset == 'CHILDEFES':
                    #print(uuid)
                    current_annot = self.DM.get_annotation_by_resource_id(uuid)
                    
                    if current_annot[0].name in EM_INDICES:
                        valid_data_uuids.append(uuid)
            """
                
            self.dataset_length += len(valid_data_uuids)
            self.dataset_intervals.append(self.dataset_length)
            self.uuids.append(valid_data_uuids)
            annots = []
            for data_uuid in valid_data_uuids:
                annots.append(self.DM.get_annotation_by_resource_id(data_uuid))
            self.annots.append(annots)
            
            
    # get information from db
    def load_data_to_memory(self, idx):

        data_idx = bisect.bisect_right(self.dataset_intervals, idx)
        # obtain the frame index
        if data_idx == 0:
            frame_idx = idx
        else:
            frame_idx = idx - self.dataset_intervals[data_idx - 1]

        # reading the frame
        uuid = self.uuids[data_idx][frame_idx]
        resource = self.DM.get_resource_by_resource_id(uuid)

        # RETRIEVE THE DATA
        image = pickle.loads(resource.data)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE).astype(np.float64)

        current_feature = self.DM.get_feature_by_resource_id(uuid, 'perona_malik_diffusion_2022_04_22')
        current_feature_data = pickle.loads(current_feature.data)
        G = cv2.imdecode(current_feature_data, cv2.IMREAD_GRAYSCALE).astype(np.float64)

        current_feature = self.DM.get_feature_by_resource_id(uuid, 'perona_malik_only_diffusion_2022_04_26')
        current_feature_data = pickle.loads(current_feature.data)
        image_blurred = cv2.imdecode(current_feature_data, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            G = G[:, :, np.newaxis]
            image_blurred = image_blurred[:, :, np.newaxis]

        if image.shape[2] == 1:
            image = np.repeat(image, 3, 2)
            G = np.repeat(G, 3, 2)
            image_blurred = np.repeat(image_blurred, 3, 2)

        # facial keypoints
        if self.current_dataset == 'SBIR':
            feature = self.DM.get_feature_by_resource_id(uuid, 'extracted_facial_positions_2d_v1_2022_04_21_no_conf')
        else:
            feature = self.DM.get_feature_by_resource_id(uuid, 'extracted_facial_positions_2d_v1_2022_03_26')
        #print(uuid)
        feature = pickle.loads(feature.data)

        # RETRIVE ANNOTATION
        # init the annots
        y = {}

        if self.annot_type == 'emotion':
            for EM in EM_INDICES:
                y[EM] = 0
        
        annots = self.annots[data_idx][frame_idx]
        y = load_annotation(annots, y)

        return image, G, image_blurred, feature, y, resource.reference_position, self.data_ids[data_idx]

def load_annotation(annots, y):
    for annot in annots:
        if annot.type == 'AU':
            au_name = re.sub("[^0-9]", "", annot.name)  ####we do not calculate with intensities
            if au_name in y:
                if annot.data_type == 'scalar_annotation':
                    y[au_name] = 1

        if annot.type == 'emotion':
            if annot.name in y:
                if annot.data_type == 'scalar_annotation':
                    y[annot.name] = 1

    return y
