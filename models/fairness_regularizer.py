import torch
from torch import nn


class DemographicParityLoss_FNNC(nn.Module):
    def __init__(self):
        super(DemographicParityLoss_FNNC, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, X, out, sensitive, y=None):
        sensitive = sensitive.view(out.shape)
        out = torch.sigmoid(out)

        idx_true = (sensitive == 1)
        # If nan value is here, just use the overall average
        mean_group0 = out[~idx_true].mean()
        mean_group1 = out[idx_true].mean()
        if torch.sum(idx_true) == 0:
            mean_group1 = mean_group0
        elif torch.sum(idx_true) == len(idx_true):
            mean_group0 = mean_group1
        cons = torch.abs(mean_group1 - mean_group0)
        return cons


class DemographicParityLoss_squared(nn.Module):
    def __init__(self):
        super(DemographicParityLoss_squared, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, X, out, sensitive, y=None):
        sensitive = sensitive.view(out.shape)
        out = torch.sigmoid(out)

        idx_true = (sensitive == 1)
        # If nan value is here, just use the overall average
        mean_group0 = out[~idx_true].mean()
        mean_group1 = out[idx_true].mean()
        if torch.sum(idx_true) == 0:
            mean_group1 = mean_group0
        elif torch.sum(idx_true) == len(idx_true):
            mean_group0 = mean_group1
        cons = torch.square(mean_group1 - mean_group0)
        return cons


def get_second_head_criterion(second_head_criterion='MSE'):

    if second_head_criterion == 'MSE':
        criterion = torch.nn.MSELoss()
    elif second_head_criterion == 'BCE':
        return torch.nn.BCEWithLogitsLoss()

    def my_loss(outputs, targets):
        loss = criterion(torch.sigmoid(outputs), targets)
        return loss

    return my_loss


def get_fairness_criterion(fairness_notion='DDP'):

    if fairness_notion == 'DDP_abs':
        return DemographicParityLoss_FNNC()
    elif fairness_notion == 'DDP_squared':
        return DemographicParityLoss_squared()

    return None