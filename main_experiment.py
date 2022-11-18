import os
from os import path
import numpy as np
from utils import load_data
from Models.attr_classifier import BinaryAttributeClassifier
from Models.attr_classifier import FairBinaryAttributeClassifier
import argparse
import pickle
import utils.utils as utils
import torch
import logging
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
import random


def collect_args_main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', dest='dataset', default='celebA')
    parser.add_argument('--model_name', dest='model', default='resnet50')
    # Determine if fairness should be considered during training
    parser.add_argument('--fairness', dest='fairness_notion', default='unconst',
                        choices=['unconst', 'DDP', 'DDP_FNNC', 'DDP_matkle', 'DEO', 'DEO_squared'])
    parser.add_argument('--fairness_parameter', type=float, default=1.0)
    parser.add_argument('--add_awareness_head', dest='awareness', default=None)
    parser.add_argument('--awareness_head_weight', dest='awareness_head_weight', type=float, default=1.0)

    # determine which parameters should be trained
    parser.add_argument('--tuning', dest='tuning', default='full_pass', choices=['last_layer', 'full_pass'])

    # If at all, determine which pretrained model should be chosen.
    # Default is the pytorch version pretrained on Imagenet
    parser.add_argument('--pretrained_on', type=int, default=-1)
    parser.add_argument('--seed_of_pretrained', type=int, default=0)  # id aka seed of the pretrained model
    # determine the fairness notion the pretrained model was trained with
    parser.add_argument('--fair_backbone', dest='fair_backbone', default='unconst',
                        choices=['unconst', 'DDP', 'DDP_FNNC', 'DDP_matkle', 'DEO', 'DEO_squared'])
    parser.add_argument('--fair_backbone_parameter', type=float, default=100)
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--backbone_stratified', dest='backbone_stratified', action='store_true')

    # optimizer settings
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--stratified', dest='stratified', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    # more parameters for the training
    parser.add_argument('--attribute', type=int, default=15)
    parser.add_argument('--protected_attribute', type=int, default=20)
    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--total_epochs', type=int, default=20)
    parser.add_argument('--number_of_samples', type=int, default=0)
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')  # if option is used, no cuda is used
    parser.add_argument('--no_log', dest='logging', action='store_false')
    parser.add_argument('--print_freq', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)

    exp_configs = vars(parser.parse_args())
    exp_configs['device'] = torch.device('cuda' if exp_configs['cuda'] else 'cpu')
    exp_configs['dtype'] = torch.float32

    if exp_configs['dataset'] == 'celebA':
        attr_list = load_data.get_all_celeba_attributes()
    elif exp_configs['dataset'] == 'LFWA':
        attr_list = load_data.get_all_LFWA_attributes()
    elif exp_configs['dataset'] == 'fairface':
        attr_list = load_data.get_all_fairface_attributes()

    optimizer_settings = {
        'optimizer': 'adam',
        'learning_rate': exp_configs['learning_rate'],
        'weight_decay': 0,
        'stratified':  exp_configs['stratified']
    }
    # make this different for the kind of finetuning, fairness notion, etc.?
    exp_configs['optimizer_settings'] = optimizer_settings

    # saving directories

    if exp_configs['pretrained_on'] > -1:
        save_folder = utils.get_model_path(dataset=exp_configs['dataset'],
                                           protected_attribute=attr_list[exp_configs['protected_attribute']],
                                           target_attribute=attr_list[exp_configs['attribute']],
                                           model_name=exp_configs['model'],
                                           tuning_method=exp_configs['tuning'],
                                           seed=exp_configs['seed'],
                                           optimizer_settings=exp_configs['optimizer_settings'].copy(),
                                           fairness=exp_configs['fairness_notion'],
                                           fairness_parameter=exp_configs['fairness_parameter'],
                                           pretrained_on=attr_list[exp_configs['pretrained_on']],
                                           pretrained_tuning_method='full_pass',
                                           pretrained_seed=exp_configs['seed_of_pretrained'],
                                           fair_backbone=exp_configs['fair_backbone'],
                                           fair_backbone_parameter=exp_configs['fair_backbone_parameter'],
                                           backbone_stratified=exp_configs['backbone_stratified'])
        pre_optimizer_settings = {
            'optimizer': 'adam',
            'learning_rate': exp_configs['backbone_lr'],
            'weight_decay': 0,
            'stratified': exp_configs['backbone_stratified']
        }
        pretrained_model_folder = utils.get_model_path(dataset=exp_configs['dataset'],
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
        save_folder = utils.get_model_path(dataset=exp_configs['dataset'],
                                           protected_attribute=attr_list[exp_configs['protected_attribute']],
                                           target_attribute=attr_list[exp_configs['attribute']],
                                           model_name=exp_configs['model'],
                                           tuning_method=exp_configs['tuning'],
                                           seed=exp_configs['seed'],
                                           optimizer_settings=exp_configs['optimizer_settings'].copy(),
                                           fairness=exp_configs['fairness_notion'],
                                           fairness_parameter=exp_configs['fairness_parameter'],
                                           awareness=exp_configs['awareness'],
                                           awareness_weight=exp_configs['awareness_head_weight'],
                                           pretrained_on='imagenet')
        exp_configs['pretrained_folder'] = None

    utils.make_dir(save_folder)
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


def main(experiment_configs):
    logging.basicConfig(filename=experiment_configs['save_folder'] + 'logging.log', filemode='w', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(experiment_configs)

    #  set random seed
    random.seed(experiment_configs['seed'])
    torch.manual_seed(experiment_configs['seed'])
    np.random.seed(experiment_configs['seed'])

    writer = SummaryWriter(experiment_configs['save_folder'])
    if experiment_configs['dataset'] == 'celebA':
        attr_list = load_data.get_all_celeba_attributes()
    elif experiment_configs['dataset'] == 'LFWA':
        attr_list = load_data.get_all_LFWA_attributes()
    elif experiment_configs['dataset'] == 'fairface':
        attr_list = load_data.get_all_fairface_attributes()

    if experiment_configs['model'] == 'inceptionv3':
        input_size = 299
    else:
        input_size = 224

    train_dataloader = load_data.get_dataloader(
        experiment_configs['attribute'],
        experiment_configs['protected_attribute'],
        experiment_configs['data_settings']['params_train'],
        dataset=experiment_configs['dataset'],
        input_size=input_size,
        augment=experiment_configs['data_settings']['augment'],
        number_of_samples=experiment_configs['number_of_samples'],
        stratified=experiment_configs['optimizer_settings']['stratified'])

    val_dataloader = load_data.get_dataloader(
        experiment_configs['attribute'],
        experiment_configs['protected_attribute'],
        experiment_configs['data_settings']['params_val'],
        dataset=experiment_configs['dataset'],
        input_size=input_size,
        augment=False,
        split='valid')

    test_dataloader = load_data.get_dataloader(
        experiment_configs['attribute'],
        experiment_configs['protected_attribute'],
        experiment_configs['data_settings']['params_val'],
        dataset=experiment_configs['dataset'],
        input_size=input_size,
        augment=False,
        split='test')

    # Train the attribute classifier
    save_path_best = experiment_configs['save_folder'] + 'best.pth'
    save_path_current = experiment_configs['save_folder'] + 'current.pth'
    save_path_history = experiment_configs['save_folder'] + 'history.pkl'

    if not experiment_configs['test_mode']:
        logging.info('Starting to train model...')
        model_path = None
        use_pretrained_model = False
        if path.exists(save_path_current):
            logging.info('Model exists, resuming training')
            model_path = save_path_current
            if path.exists(save_path_history):
                logging.info('Appending Training History')
                with open(save_path_history, 'rb') as f:
                    evaluation_history = pickle.load(f)
        else:
            logging.info('Creating Training History')
            evaluation_history = {'average_precision': [], 'accuracy': [], 'DEO': [], 'DTNR': [], 'DDP': [], 'DAcc': []}
            if experiment_configs['pretrained_folder']:
                logging.info('Loading one of my pretrained models')
                model_path = experiment_configs['pretrained_folder'] + 'best.pth'
                use_pretrained_model = True

        if experiment_configs['fairness_notion'] == 'unconst':
            attribute_classifier = BinaryAttributeClassifier(experiment_configs['model'],
                                                             device=experiment_configs['device'],
                                                             tuning=experiment_configs['tuning'],
                                                             optimizer_settings=experiment_configs[
                                                                 'optimizer_settings'],
                                                             dtype=experiment_configs['dtype'],
                                                             model_path=model_path,
                                                             writer=writer,
                                                             print_freq=experiment_configs['print_freq'],
                                                             only_warmstart=use_pretrained_model,
                                                             add_awareness_head=experiment_configs['awareness'],
                                                             awareness_head_weight=experiment_configs['awareness_head_weight'])
        else:
            attribute_classifier = FairBinaryAttributeClassifier(experiment_configs['model'],
                                                                 fairness_notion=experiment_configs[
                                                                     'fairness_notion'],
                                                                 fairness_parameter=experiment_configs[
                                                                     'fairness_parameter'],
                                                                 tuning=experiment_configs['tuning'],
                                                                 device=experiment_configs['device'],
                                                                 optimizer_settings=experiment_configs[
                                                                     'optimizer_settings'],
                                                                 dtype=experiment_configs['dtype'],
                                                                 model_path=model_path,
                                                                 writer=writer,
                                                                 print_freq=experiment_configs['print_freq'],
                                                                 only_warmstart=use_pretrained_model)

        for i in range(attribute_classifier.epoch, experiment_configs['total_epochs']):
            attribute_classifier.train(train_dataloader)

            # Evaluate on Validation Set
            y_true_val, y_score_val, val_loss = attribute_classifier.get_true_label_and_score_prediction(val_dataloader)

            # TODO Check if this is helpful. And maybe rewrite the attribute classifier and add a validate() function
            attribute_classifier.scheduler.step(val_loss)

            sens_label = y_true_val[:, 1]
            target_label = y_true_val[:, 0]
            if experiment_configs['awareness']:
                target_score = y_score_val[:, 0]
                sens_attr_pred = np.where(y_score_val[:, 1] > 0.5, 1, 0)
                awareness_accuracy = accuracy_score(sens_label, sens_attr_pred)
                logging.info('Val Awareness Accuracy = {}'.format(awareness_accuracy))
                writer.add_scalar('val_awareness_acc_' + attr_list[experiment_configs['protected_attribute']],
                                  awareness_accuracy, i)
            else:
                target_score = y_score_val

            average_precision = average_precision_score(target_label, target_score)
            threshold = 0.5
            y_pred_val = np.where(target_score > threshold, 1, 0)
            accuracy = accuracy_score(target_label, y_pred_val)

            logging.info('Val Target Accuracy = {}'.format(accuracy))
            logging.info('Decision Threshold = {}'.format(threshold))
            logging.info('Val Avg precision = {}'.format(average_precision))
            evaluation_history['average_precision'].append(average_precision)
            evaluation_history['accuracy'].append(accuracy)
            writer.add_scalar('val_loss', val_loss, i)
            writer.add_scalar('val_acc', accuracy, i)
            writer.add_scalar('val_average_precision', average_precision, i)

            fairness_evaluation = utils.evaluate_fairness_measures(y_pred_val, target_label, sens_label)
            logging.info('Val DTPR = {}'.format(fairness_evaluation['DTPR']))
            logging.info('Val DTNR = {}'.format(fairness_evaluation['DTNR']))
            logging.info('Val DDP = {}'.format(fairness_evaluation['DDP']))
            logging.info('Val DAcc = {}'.format(fairness_evaluation['DAcc']))

            evaluation_history['DEO'].append(fairness_evaluation['DTPR'])
            evaluation_history['DTNR'].append(fairness_evaluation['DTNR'])
            evaluation_history['DDP'].append(fairness_evaluation['DDP'])
            evaluation_history['DAcc'].append(fairness_evaluation['DAcc'])

            writer.add_scalar('val_dtpr', fairness_evaluation['DTPR'], i)
            writer.add_scalar('val_dtnr', fairness_evaluation['DTNR'], i)
            writer.add_scalar('val_ddp', fairness_evaluation['DDP'], i)
            writer.add_scalar('val_dacc', fairness_evaluation['DAcc'], i)
            writer.add_scalar('val_acc_group0', fairness_evaluation[0]['acc'], i)
            writer.add_scalar('val_acc_group1', fairness_evaluation[1]['acc'], i)
            writer.add_scalar('val_pr_group0', fairness_evaluation[0]['pr'], i)
            writer.add_scalar('val_pr_group1', fairness_evaluation[1]['pr'], i)
            writer.add_scalar('val_tpr_group0', fairness_evaluation[0]['tpr'], i)
            writer.add_scalar('val_tpr_group1', fairness_evaluation[1]['tpr'], i)
            writer.add_scalar('val_tnr_group0', fairness_evaluation[0]['tnr'], i)
            writer.add_scalar('val_tnr_group1', fairness_evaluation[1]['tnr'], i)

            if experiment_configs['logging']:
                with open(save_path_history, 'wb+') as handle:
                    pickle.dump(evaluation_history, handle)
                if average_precision > attribute_classifier.best_eval_measure:
                    attribute_classifier.best_eval_measure = average_precision
                    attribute_classifier.save_model(save_path_best)
                attribute_classifier.save_model(save_path_current)

    attribute_classifier = BinaryAttributeClassifier(experiment_configs['model'],
                                                     optimizer_settings=None,
                                                     tuning='full_pass',
                                                     model_path=save_path_best,
                                                     add_awareness_head=experiment_configs['awareness'],
                                                     awareness_head_weight=experiment_configs['awareness_head_weight'],
                                                     dtype=experiment_configs['dtype'],
                                                     only_warmstart=True,
                                                     device=experiment_configs['device'])
    train_dataloader = load_data.get_dataloader(
                                experiment_configs['attribute'],
                                experiment_configs['protected_attribute'],
                                experiment_configs['data_settings']['params_val'],
                                dataset=experiment_configs['dataset'],
                                input_size=input_size,
                                augment=False,
                                stratified=False,
                                split='train')
    y_true_train, y_score_train, _ = attribute_classifier.get_true_label_and_score_prediction(train_dataloader)
    y_true_val, y_score_val, _ = attribute_classifier.get_true_label_and_score_prediction(val_dataloader)
    y_true_test, y_score_test, _ = attribute_classifier.get_true_label_and_score_prediction(test_dataloader)

    if experiment_configs['logging']:
        with open(experiment_configs['save_folder'] + 'y_score_train.pkl', 'wb+') as handle:
            pickle.dump(y_score_train, handle)
        with open(experiment_configs['save_folder'] + 'y_true_train.pkl', 'wb+') as handle:
            pickle.dump(y_true_train, handle)
        with open(experiment_configs['save_folder'] + 'y_score_val.pkl', 'wb+') as handle:
            pickle.dump(y_score_val, handle)
        with open(experiment_configs['save_folder'] + 'y_true_val.pkl', 'wb+') as handle:
            pickle.dump(y_true_val, handle)
        with open(experiment_configs['save_folder'] + 'y_score_test.pkl', 'wb+') as handle:
            pickle.dump(y_score_test, handle)
        with open(experiment_configs['save_folder'] + 'y_true_test.pkl', 'wb+') as handle:
            pickle.dump(y_true_test, handle)

    cal_thresh = 0.5
    y_pred_train = np.where(y_score_train > cal_thresh, 1, 0)
    y_pred_val = np.where(y_score_val > cal_thresh, 1, 0)
    y_pred_test = np.where(y_score_test > cal_thresh, 1, 0)

    awareness_accuracy_val = None
    awareness_accuracy_test = None
    awareness_accuracy_train = None
    if experiment_configs['awareness']:
        s_pred_train = y_pred_train[:, 1]
        s_pred_val = y_pred_val[:, 1]
        s_pred_test = y_pred_test[:, 1]
        y_pred_train = y_pred_train[:, 0]
        y_pred_val = y_pred_val[:, 0]
        y_pred_test = y_pred_test[:, 0]
        y_score_train = y_score_train[:, 0]
        y_score_val = y_score_val[:, 0]
        y_score_test = y_score_test[:, 0]
        awareness_accuracy_val = accuracy_score(y_true_val[:, 1], s_pred_val)
        awareness_accuracy_test = accuracy_score(y_true_test[:, 1], s_pred_test)
        awareness_accuracy_train = accuracy_score(y_true_train[:, 1], s_pred_train)

    average_precision = average_precision_score(y_true_val[:, 0], y_score_val)
    accuracy = accuracy_score(y_true_val[:, 0], y_pred_val)

    fairness_results = utils.evaluate_fairness_measures(y_pred_val, y_true_val[:, 0], y_true_val[:, 1])

    val_results = {
        # 'f1_thresh': f1_thresh,
        'cal_thresh': cal_thresh,
        'opt': experiment_configs,
        'average_precision': average_precision,
        'accuracy': accuracy,
        'awareness_accuracy': awareness_accuracy_val,
        'DEO': fairness_results['DTPR'],
        'DTNR': fairness_results['DTNR'],
        'DDP': fairness_results['DDP'],
        'DAcc': fairness_results['DAcc'],
        'fairness_results': fairness_results
    }

    logging.info('Validation results: ')
    logging.info(val_results)

    average_precision = average_precision_score(y_true_test[:, 0], y_score_test)
    accuracy = accuracy_score(y_true_test[:, 0], y_pred_test)
    fairness_results = utils.evaluate_fairness_measures(y_pred_test, y_true_test[:, 0], y_true_test[:, 1])

    test_results = {
        #'f1_thresh': f1_thresh,
        'cal_thresh': cal_thresh,
        'opt': experiment_configs,
        'average_precision': average_precision,
        'accuracy': accuracy,
        'awareness_accuracy': awareness_accuracy_test,
        'DEO': fairness_results['DTPR'],
        'DTNR': fairness_results['DTNR'],
        'DDP': fairness_results['DDP'],
        'DAcc': fairness_results['DAcc'],
        'fairness_results': fairness_results
    }

    logging.info('Test results: ')
    logging.info(test_results)

    average_precision = average_precision_score(y_true_train[:, 0], y_score_train)
    accuracy = accuracy_score(y_true_train[:, 0], y_pred_train)

    if experiment_configs['attribute'] != experiment_configs['protected_attribute']:
        fairness_results = utils.evaluate_fairness_measures(y_pred_train, y_true_train[:, 0], y_true_train[:, 1])
    else:
        fairness_results = {'DDP': [], 'DTPR': [], 'DTNR': [], 'DAcc': []}

    train_results = {
        # 'f1_thresh': f1_thresh,
        'cal_thresh': cal_thresh,
        'opt': experiment_configs,
        'average_precision': average_precision,
        'accuracy': accuracy,
        'awareness_accuracy': awareness_accuracy_train,
        'DEO': fairness_results['DTPR'],
        'DTNR': fairness_results['DTNR'],
        'DDP': fairness_results['DDP'],
        'DAcc': fairness_results['DAcc'],
        'fairness_results': fairness_results
    }

    logging.info('Train results: ')
    logging.info(train_results)

    if experiment_configs['logging']:
        with open(experiment_configs['save_folder'] + 'train_results.pkl', 'wb+') as handle:
            pickle.dump(train_results, handle)
        with open(experiment_configs['save_folder'] + 'val_results.pkl', 'wb+') as handle:
            pickle.dump(val_results, handle)
        with open(experiment_configs['save_folder'] + 'test_results.pkl', 'wb+') as handle:
            pickle.dump(test_results, handle)


if __name__ == "__main__":
    configs = collect_args_main()
    main(configs)
