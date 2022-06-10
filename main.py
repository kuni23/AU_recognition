from dataset_imbased import AUDataset
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from models import Resnet, KeypointNet, Resnet_fusion, SimpleCNN
from training_resnet import calculate_skewed_f1
from sklearn.metrics import f1_score
import argparse
from utils import *
import numpy as np


torch.manual_seed(31337)
np.random.seed(31337)

def main(test_db_path, test_dataset_name, model_path, way_split, number_split):
    ################INITIALIZATION######################################################################################
    device = 'cuda'
    bs = 16
    threshold = 0.5

    #######################TESTING######################################################################################
    create_annotation_output = False
    write_out_images = False
    y_true, y_pred = [], []
    all_predictions = {}
    annot_indices = [10]

    valid_ds = {
        'db_path': test_db_path,
        'dataset_name': test_dataset_name,
        'data_type': 'cv_sbir',
        'way_of_split': [1, 5],
        'transform': valid_transform,
        'annot_indices': annot_indices,

    }

    test_images = AUDataset(**valid_ds)
    print(len(test_images))
    model = Resnet_fusion(pretrained_task='neutral_sigmoid', n_classes=len(annot_indices), input_keypoints=34)

    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.to(device).eval()

    test_loader = DataLoader(test_images, batch_size=bs, shuffle=False)
    for i, (input, target) in enumerate(test_loader):

        data_id, ref_pos = test_images.get_dataid_refpos(range(i*bs, np.minimum(len(test_images), (i+1)*bs)))
        if data_id[0] not in all_predictions.keys():
            all_predictions[data_id[0]] = []
        with torch.no_grad():
            ##to handle list/tuple input
            if isinstance(input, list):
                input = [x_i.to(device) for x_i in input]
            else:
                input = input.to(device)

            pred = model(input)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            all_predictions[data_id[0]].append(pred.cpu().numpy())
            if write_out_images:
                write_error_images(input, target.cpu().numpy(), pred.cpu().numpy() > 0.5,
                                   data_id, ref_pos, annot_indices, transform=inv_transform)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred, dtype=float) > threshold

    # normlaized skew f1 implementation
    average_f1s = calculate_skewed_f1(y_true, y_pred)
    print('average skew normalized f1s:', average_f1s)
    print('average of the average skew normalized f1s:', np.mean(average_f1s))

    # debug - compare original f1 to skew normalized f1
    if len(annot_indices) == 1:
        f1s = np.array(f1_score(y_true=y_true, y_pred=y_pred))
        print("f1 score", f1s)
    else:
        f1s = np.array(f1_score(y_true=y_true, y_pred=y_pred, average=None))
        print("f1 score", f1s)
        print("average of f1 scores", np.mean(f1s))

    for i in range(y_true.shape[1]):
        print(confusion_matrix(y_true[:, i], y_pred[:, i]))

    if create_annotation_output:
        threshold_for_cutoff = 0.5
        for key in all_predictions.keys():
            convert_to_annotation(all_predictions[key], threshold_for_cutoff, key)

if __name__ == '__main__':
    param_parser = argparse.ArgumentParser(description='base paths')

    param_parser.add_argument('--dbPath_test',
                                metavar='path2',
                                default = '',
                                type=str,
                                help='root path for test images to process')

    param_parser.add_argument('--datasetName_test',
                                metavar='mod2',
                                default = 'SBIR',
                                type=str,
                                help='dataset used for testing')

    param_parser.add_argument('--pathModel',
                                metavar='mod6',
                                default='',
                                type=str,
                                help='path of the model to save or to test')

    param_parser.add_argument('--waySplit',
                                metavar='mod7',
                                default='test_split',
                                type=str,
                                help='the way to evaluate')

    param_parser.add_argument('--numberSplit',
                                metavar='mod8',
                                default=0,
                                type=int,
                                help='the number of the split')

    args = param_parser.parse_args()

    main(args.dbPath_test, args.datasetName_test, args.pathModel, args.waySplit, args.numberSplit)
