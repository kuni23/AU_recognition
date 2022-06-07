
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, F1Score

from argusutil.ml.trainer import Trainer
from sklearn.model_selection import KFold
from argusutil.ml.metrics import SkewedF1Score
from research.fer.models.models import Resnet, Resnet_fusion
from research.fer.utils import train_transform, valid_transform, keypoints_indices
from research.fer.dataset_modules.dataset_imbased import AUDatasetPrototype


torch.manual_seed(823)
np.random.seed(823)


AU_INDICES = [10]
list_of_data_needed_train = {'data_type': 'tuple', 'keypoints': keypoints_indices['mouth'], 'image': ''}
list_of_data_needed_test = {'data_type': 'tuple', 'keypoints': keypoints_indices['mouth'], 'image': ''}

splits = [[0, 2, 3, 4], [1, 5]]
f1_preds = []
for fold, (valid_ids) in enumerate(splits):
    config = {
        'AU': AU_INDICES,
        'epochs': 100,
        'batch_size': 16,
        'lr': 0.001,
        'train_ds': {
            'db_path': '/data/emotion_data/normalized_face_database_v12.db',
            'dataset_name': 'BP-4DSFE',
            'data_type': 'train_split',
            'way_of_split': 1,
            'transform': train_transform,
            'annot_indices': AU_INDICES,
            'list_of_data_needed': list_of_data_needed_train

        },
        'valid_ds': [
            {
                'db_path': '/data/emotion_data/rush_emotion_data_reviewed_v7.db',
                'dataset_name': 'SBIR',
                'data_type': 'cv_sbir',
                'way_of_split': valid_ids,
                'transform': valid_transform,
                'annot_indices': AU_INDICES,
                'list_of_data_needed': list_of_data_needed_test
            }

        ],
    }

    train_ds = AUDatasetPrototype(**config['train_ds'])
    valid_ds = [AUDatasetPrototype(**val_config) for val_config in config['valid_ds']]
    print(len(train_ds))
    print(len(valid_ds[0]))

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], num_workers=4, shuffle=True)
    valid_loader = [DataLoader(ds, batch_size=config['batch_size'], num_workers=4, shuffle=False)
                    for ds in valid_ds]
    date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # model = Resnet(pretrained=True, n_classes=len(config['AU']))
    model = Resnet_fusion(pretrained_task='neutral_sigmoid', n_classes=len(config['AU']), input_keypoints=34)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      lr_scheduler=StepLR(optimizer, step_size=30, gamma=0.1),
                      loss_fn=torch.nn.BCELoss(),
                      metrics=[Precision(), Recall(), F1Score(), SkewedF1Score()],
                      train_dl=train_loader,
                      val_dl=valid_loader,
                      epochs=config['epochs'],
                      logs_root='/data/runs',
                      experiment_id=f'2foldcv_finaleval_{"_".join(map(str, config["AU"]))}/{date}',
                      monitor=(f'Validation[{valid_ds[0].name_tag}]/F1Score', 'max'),
                      n_checkpoints=3,
                      early_stop_patience=5)
    trainer.fit()
