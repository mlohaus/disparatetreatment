import argparse
from utils import func_utils
from utils import load_data
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', dest='dataset', default='celebA', choices=['celebA', 'fairface'])
    parser.add_argument('--attribute', type=int, default=31)
    parser.add_argument('--protected_attribute', type=int, default=20)
    parser.add_argument('--model_name', dest='model', default='mobilenet', choices=['resnet50', 'mobilenet'])
    parser.add_argument('--fairness', dest='fairness_notion', default='unconst',
                        choices=['unconst', 'DDP_abs', 'DDP_squared']) # Determine if fairness should be considered during training
    parser.add_argument('--fairness_parameter', type=float, default=1.0)
    parser.add_argument('--two_headed', dest='two_headed', action='store_true')
    parser.add_argument('--second_head_weight', dest='second_head_weight', type=float, default=0.5)

    # optimizer settings
    parser.add_argument('--total_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--stratified', dest='stratified', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--number_of_samples', type=int, default=0)

    # settings for a pretrained model if used
    # Default is the pytorch version pretrained on Imagenet
    parser.add_argument('--pretrained_on', type=int, default=-1)  # number of attribute that the network was pretrained on
    parser.add_argument('--seed_of_pretrained', type=int, default=0)  # id aka seed of the pretrained model
    # determine the fairness notion the pretrained model was trained with
    parser.add_argument('--fair_backbone', dest='fair_backbone', default='unconst',
                        choices=['unconst', 'DDP_abs', 'DDP_squared'])
    parser.add_argument('--fair_backbone_parameter', type=float, default=1.0)
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--backbone_stratified', dest='backbone_stratified', action='store_true')

    # more parameters for the training
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)

    exp_configs = vars(parser.parse_args())
    exp_configs['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_configs['dtype'] = torch.float32

    if exp_configs['dataset'] == 'celebA':
        attr_list = load_data.get_all_celeba_attributes()
    elif exp_configs['dataset'] == 'fairface':
        attr_list = load_data.get_all_fairface_attributes()

    optimizer_settings = {
        'learning_rate': exp_configs['learning_rate'],
        'stratified':  exp_configs['stratified']
    }
    exp_configs['optimizer_settings'] = optimizer_settings

    # saving directories
    if exp_configs['pretrained_on'] > -1:
        save_folder = func_utils.get_model_path(dataset=exp_configs['dataset'],
                                           protected_attribute=attr_list[exp_configs['protected_attribute']],
                                           target_attribute=attr_list[exp_configs['attribute']],
                                           model_name=exp_configs['model'],
                                           tuning_method='last_layer',
                                           seed=exp_configs['seed'],
                                           optimizer_settings=exp_configs['optimizer_settings'].copy(),
                                           fairness=exp_configs['fairness_notion'],
                                           fairness_parameter=exp_configs['fairness_parameter'],
                                           pretrained_on=attr_list[exp_configs['pretrained_on']],
                                           pretrained_seed=exp_configs['seed_of_pretrained'],
                                           fair_backbone=exp_configs['fair_backbone'],
                                           fair_backbone_parameter=exp_configs['fair_backbone_parameter'],
                                           backbone_stratified=exp_configs['backbone_stratified'])
        pre_optimizer_settings = {
            'learning_rate': exp_configs['backbone_lr'],
            'stratified': exp_configs['backbone_stratified']
        }
        pretrained_model_folder = func_utils.get_model_path(dataset=exp_configs['dataset'],
                                                       protected_attribute=attr_list[exp_configs['protected_attribute']],
                                                       target_attribute=attr_list[exp_configs['pretrained_on']],
                                                       model_name=exp_configs['model'],
                                                       tuning_method='full_pass',
                                                       seed=exp_configs['seed_of_pretrained'],
                                                       optimizer_settings=pre_optimizer_settings.copy(),
                                                       fairness=exp_configs['fair_backbone'],
                                                       fairness_parameter=exp_configs['fair_backbone_parameter'],
                                                       pretrained_on='imagenet')
        exp_configs['pretrained_folder'] = pretrained_model_folder + '/'
    else:
        save_folder = func_utils.get_model_path(dataset=exp_configs['dataset'],
                                           protected_attribute=attr_list[exp_configs['protected_attribute']],
                                           target_attribute=attr_list[exp_configs['attribute']],
                                           model_name=exp_configs['model'],
                                           tuning_method='full_pass',
                                           seed=exp_configs['seed'],
                                           optimizer_settings=exp_configs['optimizer_settings'].copy(),
                                           fairness=exp_configs['fairness_notion'],
                                           fairness_parameter=exp_configs['fairness_parameter'],
                                           two_headed=exp_configs['two_headed'],
                                           pretrained_on='imagenet')
        exp_configs['pretrained_folder'] = None

    func_utils.make_dir(save_folder)
    exp_configs['save_folder'] = save_folder + '/'

    params_train = {'batch_size': exp_configs['batch_size'],
                    'shuffle': True,
                    'num_workers': 0}

    params_val = {'batch_size': exp_configs['batch_size'],
                  'shuffle': False,
                  'num_workers': 0}

    data_settings = {
        'params_train': params_train,
        'params_val': params_val,
        'augment': True
    }
    exp_configs['data_settings'] = data_settings
    return exp_configs