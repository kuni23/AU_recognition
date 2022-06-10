from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from research.fer.dataset_modules.dataset_functions_imbased import AuDataManagement, EmotionDataManagement, EM_INDICES
import torch


class AUDataset(Dataset):

    def __init__(self, db_path, dataset_name, data_type, annotation_type='AU', way_of_split=0.9, transform=None,
                 keypoints_eye=False, keypoints_mouth=False, crop_mouth=False, crop_eyes=False,
                 keypoints_input_eye=False, keypoints_input_mouth=False, jittering=False, annot_indices=None):

        self.name_tag = Path(db_path).stem + '-' + dataset_name
        self.DM = AuDataManagement(db_path, dataset_name, data_type, annotation_type, way_of_split, jittering, annot_indices)
        self.transform = transform
        self.annotation_type = annotation_type
        self.keypoint_eye = keypoints_eye
        self.keypoint_mouth = keypoints_mouth
        self.crop_mouth = crop_mouth
        self.crop_eyes = crop_eyes
        self.keypoint_input_eye = keypoints_input_eye
        self.keypoint_input_mouth = keypoints_input_mouth
        self.annot_indices = annot_indices

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        X, G, X_blurred, feature, y, ref_pos, data_id = self.DM.load_data_to_memory(idx)

        if self.keypoint_mouth:
            feature = feature[32:, :]
            for i in range(feature.shape[0]):
                X[int(feature[i, 1]), int(feature[i, 0]), :] = [255, 255, 255]

        if self.keypoint_eye:
            feature = np.concatenate([feature[0:5, :], feature[5:10, :], feature[19:25, :], feature[25:31, :]], axis=0)
            for i in range(feature.shape[0]):
                X[int(feature[i, 1]), int(feature[i, 0]), :] = [255, 255, 255]

        if self.crop_mouth:
            ledge = np.round(feature[32, :2])
            redge = np.round(feature[38, :2])
            if ledge[0] > redge[0] or ledge[1] > redge[1]:
                box = (0, 0, X.shape[1], X.shape[0])
            else:
                box = (ledge[0] - 40, ledge[1] - 50, redge[0] + 40, redge[1] + 30)

            X = Image.fromarray(np.uint8(X))
            X = np.array(X.crop(box))

        if self.crop_eyes:
            ledge = np.round(feature[19, :2])
            redge = np.round(feature[28, :2])
            if ledge[0] > redge[0] or ledge[1] > redge[1]:
                box = (0, 0, X.shape[1], X.shape[0])
            else:
                box = (ledge[0] - 40, ledge[1] - 40, redge[0] + 40, redge[1] + 60)

            X = Image.fromarray(np.uint8(X))
            X = np.array(X.crop(box))

        if self.transform:
            current_input = Image.fromarray(np.uint8(X))
            current_input = self.transform(current_input)
            current_input = np.array(current_input, dtype=np.float32)

        else:
            current_input = np.array(X, dtype=np.float32)

        if self.annotation_type == 'AU':
            current_AUs = [y[str(AU)] for AU in self.annot_indices]
            current_AUs = np.array(current_AUs, dtype=np.float32)
        else:
            #current_AUs = [y[EM] for EM in EM_INDICES]
            current_AUs = [y['happy'] + y['pleased']] #for AU12 or for AU6
            current_AUs = np.array(current_AUs, dtype=np.float32)

        if self.keypoint_input_mouth:
            #1st normalization
            feature = feature[:, :2] - feature[13:14, :2]
            #2nd normalization
            feature[32:, :] = feature[32:, :] - ((feature[32:33, :] + feature[38:39, :]) / 2) # mouth
            feature = np.concatenate([feature[32:, :]], axis=0)
            current_input = (current_input, np.array(feature.reshape(34, ), dtype=np.float32))

        if self.keypoint_input_eye:
            #1st normalization
            feature = feature[:, :2] - feature[13:14, :2]
            #2nd normalization
            feature[0:5, :] = feature[0:5, :] - ((feature[19:20, :] + feature[22:23, :]) / 2) #reyebrow
            feature[5:10, :] = feature[5:10, :] - ((feature[25:26, :] + feature[28:29, :]) / 2) #leyebrow
            feature[19:25, :] = feature[19:25, :] - ((feature[19:20, :] + feature[22:23, :]) / 2) #reye
            feature[25:31, :] = feature[25:31, :] - ((feature[25:26, :] + feature[28:29, :]) / 2) #leye
            feature = np.concatenate([feature[0:5, :], feature[5:10, :], feature[19:25, :], feature[25:31, :]], axis=0)
            current_input = (current_input, np.array(feature.reshape(44, ), dtype=np.float32))

        return current_input, current_AUs

    def __len__(self):
        return self.DM.dataset_length

class EmotionDataset(Dataset):
    
    def __init__(self, db_path, dataset_name, data_type, annotation_type='emotion', way_of_split=0.9, transform=None,
                 keypoints=False, crop_mouth=False, convert_to_au=False):
        self.name_tag = Path(db_path).stem + '-' + dataset_name
        self.EDM = EmotionDataManagement(db_path, dataset_name, data_type, annotation_type, way_of_split)
        self.transform = transform
        self.annotation_type = annotation_type
        self.keypoint = keypoints
        self.crop_mouth = crop_mouth
        self.convert_to_au = convert_to_au

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        X, G, X_blurred, feature, y, ref_pos, data_id = self.EDM.load_data_to_memory(idx)

        if self.keypoint:
            for i in range(feature.shape[0]):
                X[int(feature[i, 1]), int(feature[i, 0]), :] = [255, 255, 255]

        if self.crop_mouth:
            ledge = np.round(feature[32, :2])
            redge = np.round(feature[38, :2])
            if ledge[0] > redge[0] or ledge[1] > redge[1]:
                box = (0, 0, X.shape[1], X.shape[0])
            else:
                box = (ledge[0] - 40, ledge[1] - 50, redge[0] + 40, redge[1] + 30)
            X = Image.fromarray(np.uint8(X))
            X = np.array(X.crop(box))
        
        if self.transform:
            current_input = Image.fromarray(np.uint8(X))
            current_input = self.transform(current_input)
            current_input = np.array(current_input, dtype=np.float32)

        else:
            current_input = np.array(X, dtype=np.float32)
            
        current_emotions = [y[str(emotion)] for emotion in EM_INDICES]
        current_emotions = np.argmax(np.array(current_emotions, dtype=np.int_))
            
        return current_input, current_emotions

    def __len__(self):
        return self.EDM.dataset_length