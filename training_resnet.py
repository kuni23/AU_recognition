from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
from emotion import AU_INDICES
import time


def calculate_metrics(pred, target):
    return {
        'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
        'macro/prec': precision_score(y_true=target, y_pred=pred, average='macro'),
        'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro')}


def calculate_skewed_f1(y_true, y_pred):

    average_f1s = []
    for it in range(y_true.shape[1]):
        current_y_true = y_true[:, it]
        current_y_pred = y_pred[:, it]
        positive_positions = [i for i, x in enumerate(current_y_true) if x == 1]
        negative_positions = [i for i, x in enumerate(current_y_true) if x == 0]
        print('no of positives:', len(positive_positions), 'no of negatives:', len(negative_positions))
        number_of_positives = len(positive_positions)
        number_of_negatives = len(negative_positions)
        rounded_skew = int(np.ceil(number_of_negatives / number_of_positives))
        print('current skew:', rounded_skew)
        if rounded_skew > 1:
            current_f1s = []
            for ij in range(rounded_skew):
                # take sample and calculate f1
                current_negatives = np.random.choice(negative_positions, size=int(number_of_positives), replace=False)
                random_y_true = np.concatenate((current_y_true[positive_positions], current_y_true[current_negatives]),
                                               axis=0)
                random_y_pred = np.concatenate((current_y_pred[positive_positions], current_y_pred[current_negatives]),
                                               axis=0)
                current_f1 = np.array(f1_score(y_true=random_y_true, y_pred=random_y_pred))
                current_f1s.append(current_f1)

            average_f1s.append(np.mean(current_f1))

    return average_f1s


class Trainer_resnet():
    def __init__(self, train_loader, valid_loader, device, model, criterion):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion.to(device)

    def train(self, optimizer, epochs, early_stopping=False):
        cudnn.benchmark = True
        stop_it = 0
        best_value = 0
        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch, epochs)

            # train for one epoch
            train_for_epoch(self.train_loader, self.model, self.criterion, optimizer, epoch, self.device)

            # evaluate on validation set
            f1_model, y_true, probs_visual = validate(self.valid_loader, self.model, self.criterion, self.device)
            y_pred_visual = probs_visual > 0.5
            metrics_visual = calculate_metrics(y_pred_visual, y_true)
            #average_f1s = calculate_skewed_f1(y_true, y_pred_visual)
            #f1_skew_norm = np.mean(average_f1s)
            f1_skew_norm = 0

            if len(AU_INDICES) == 1:
                #binary f1
                useful_metric = f1_score(y_true=y_true, y_pred=y_pred_visual)
            else:
                #average f1
                useful_metric = metrics_visual['macro/f1']

            print('Current f1 score {f1_current} '
                  'Current f1 norm skewed {f1_skewed} '
                  'Best f1 score {f1_best} '.format(f1_current=useful_metric,
                                                    f1_skewed=f1_skew_norm, f1_best=best_value))
            if len(AU_INDICES) == 1:
                print(confusion_matrix(y_true, y_pred_visual))
            else:
                print(f1_score(y_true=y_true, y_pred=y_pred_visual, average=None))

            if best_value < useful_metric:
                best_value = useful_metric
                stop_it = 0
                #torch.save({
                #    'epoch': epoch,
                #    'model_state_dict': self.model.state_dict(),
                #    'optimizer_state_dict': optimizer.state_dict(),
                #}, 'checkpoints/best_skewed' + str(epoch) + '.pt')

            if early_stopping:
                if best_value > useful_metric:
                    stop_it += 1
                    if stop_it == 5:
                        break

        return best_value


def train_for_epoch(train_loader, model, criterion, optimizer, epoch, device):
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for input, target in train_loader:
        target = target.type(torch.FloatTensor).to(device)
        if isinstance(input, list):
            input = [x_i.to(device) for x_i in input]
        else:
            input = input.to(device)
        pred_score = model(input)

        loss = criterion(pred_score, target)

        # measure accuracy and record loss
        if isinstance(input, list):
            losses.update(loss.item(), input[0].size(0))
        else:
            losses.update(loss.item(), input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch: {0} '
          'Training loss ({loss.avg}) '.format(epoch, loss=losses))


def validate(val_loader, model, criterion, device):
    losses = AverageMeter()
    y_true, y_pred = [], []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for input, target in val_loader:
            target = target.type(torch.FloatTensor).to(device)
            if isinstance(input, list):
                input = [x_i.to(device) for x_i in input]
            else:
                input = input.to(device)

            pred_score = model(input)

            loss = criterion(pred_score, target)

            # measure accuracy and record loss
            if isinstance(input, list):
                losses.update(loss.item(), input[0].size(0))
            else:
                losses.update(loss.item(), input.size(0))

            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred_score.cpu().numpy())

    #print('Valid loss: {loss.avg} '.format(loss=losses))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred, dtype=float)

    return f1_score(y_true=y_true, y_pred=y_pred > 0.5, average=None), y_true, y_pred


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, epochs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    if epoch in [int(epochs * 0.3), int(epochs * 0.5), int(epochs * 0.8)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
