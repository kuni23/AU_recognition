import cv2
import numpy as np
import re
from database_functions import DatabaseManagement
import pickle
import bisect
import random
from jittering import jitter_points



# dict for start, stop frames
sbir_border = {
    '00ad44fd-cec2-446b-85b8-7922d172a45d': [15000, 29999],
    '2c0930ad-46c6-4a7a-9720-8736487e7b7b': [29988, 44979],
    '81b5da37-311e-4f94-a1e6-8f66486ec483': [37489, 52485],
    'b21e9c66-561a-48f8-9611-ab69c811fb35': [50965, 65965],
    '872fd1cf-a43b-4e1e-a998-db4f8e93ad9a': [41965, 56961],
    '4de55bad-84d9-4f51-a199-87410ee38a7f': [55958, 70958]
}

random.seed(0)

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
        if self.skew > 1 and jittering:
            self.dataset_length = self.dataset_length + round((self.skew-1)*no_positive)

        if self.skew < 1 and jittering:
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
