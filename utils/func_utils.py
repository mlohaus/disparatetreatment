import os
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.special import logit


def make_dir(pathname):
    if not os.path.isdir(pathname):
        os.makedirs(pathname)


def get_model_path(dataset: str = 'celebA',
                   protected_attribute: str = 'Male',
                   target_attribute: str = 'Young',
                   model_name: str = 'resnet50',
                   tuning_method: str = 'full_pass',
                   seed: int = 0,
                   optimizer_settings: dict = None,
                   fairness: str = 'unconst',
                   fairness_parameter: float = 1.0,
                   pretrained_on: str = 'imagenet',
                   pretrained_seed: int = 0,
                   two_headed: bool = False,
                   fair_backbone: str = 'unconst',
                   fair_backbone_parameter: float = 1.0,
                   backbone_stratified: bool = False
                   ):

    optimizer_settings_copy = optimizer_settings.copy()
    if 'stratified' in optimizer_settings_copy and not optimizer_settings_copy['stratified']:
        optimizer_settings_copy.pop('stratified')

    save_folder = os.path.join('logs',dataset,
                               'target_'+target_attribute+'_prot_' + protected_attribute)

    if pretrained_on != 'imagenet':
        if fair_backbone == 'unconst':
            save_folder = os.path.join(save_folder,
                                       model_name+'_pretrained_' + '_on_' + pretrained_on + backbone_stratified * 'stratified_'
                                       + '_fairness_' + fair_backbone,
                                       'previous_seed_' + str(pretrained_seed) + '_this_seed_' + str(seed))
        else:
            save_folder = os.path.join(save_folder,
                                       model_name+'_pretrained_' + '_on_' + pretrained_on + backbone_stratified * 'stratified_'
                                       + '_fairness_' + fair_backbone + '_lambda_' + str(fair_backbone_parameter),
                                       'previous_seed_' + str(pretrained_seed) + '_this_seed_' + str(seed))
        save_folder = os.path.join(save_folder,
                                   '_'.join([key + '_' + str(optimizer_settings_copy[key])
                                             for key in optimizer_settings_copy.keys()]))
    else:

        save_folder = os.path.join(save_folder,
                                   model_name+'_pretrained_on_imagenet',
                                   tuning_method)

        if fairness == 'unconst':
            save_folder = os.path.join(save_folder,
                                       'fairness_' + fairness + bool(two_headed) * '_twoheaded',
                                       'seed_' + str(seed) + '_' + '_'.join([key + '_' + str(optimizer_settings_copy[key])
                                                 for key in optimizer_settings_copy.keys()]))
        else:
            save_folder = os.path.join(save_folder,
                                       'fairness_' + fairness
                                       + '_lambda_' + str(fairness_parameter) + bool(two_headed) * '_twoheaded',
                                       'seed_' + str(seed) + '_' + '_'.join([key + '_' + str(optimizer_settings_copy[key])
                                                 for key in optimizer_settings_copy.keys()]))
    return save_folder


def get_acc(y_score, y_true, threshold=0.5):
    y_pred = np.where(y_score > threshold, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def get_constant_accuracy(y_true):
    constant_accuracy = max(accuracy_score(y_true, np.ones_like(y_true)),
                            accuracy_score(y_true, np.zeros_like(y_true)))
    return constant_accuracy


def get_DDP(y_pred, sens_label):
    return np.average(y_pred[sens_label == 1]) - np.average(y_pred[sens_label == 0])

def invert_sigmoid(y_score):
    return np.clip(logit(y_score), -25, 25)


if __name__ == '__main__':

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    sens_attr = np.array([1, 0, 1, 0])