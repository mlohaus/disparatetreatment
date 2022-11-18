from os import path
import numpy as np
import pickle
import utils.func_utils as ut
import utils.parse_experiment_args as parse_experiment_args
from utils import load_data
import torch
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import random
import logging

from models.attr_classifier import BinaryAttributeClassifier
from models.attr_classifier import FairBinaryAttributeClassifier


def main(experiment_configs):
    logging.basicConfig(filename=experiment_configs['save_folder'] + 'logging.log', filemode='w', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(experiment_configs)

    #  set random seed
    if experiment_configs['seed']:
        random.seed(experiment_configs['seed'])
        torch.manual_seed(experiment_configs['seed'])
        np.random.seed(experiment_configs['seed'])

    train_dataloader = load_data.get_dataloader(
        experiment_configs['attribute'],
        experiment_configs['protected_attribute'],
        experiment_configs['data_settings']['params_train'],
        dataset=experiment_configs['dataset'],
        augment=experiment_configs['data_settings']['augment'],
        number_of_samples=experiment_configs['number_of_samples'],
        stratified=experiment_configs['optimizer_settings']['stratified'])

    val_dataloader = load_data.get_dataloader(
        experiment_configs['attribute'],
        experiment_configs['protected_attribute'],
        experiment_configs['data_settings']['params_val'],
        dataset=experiment_configs['dataset'],
        number_of_samples=experiment_configs['number_of_samples'],
        augment=False,
        split='valid')

    test_dataloader = load_data.get_dataloader(
        experiment_configs['attribute'],
        experiment_configs['protected_attribute'],
        experiment_configs['data_settings']['params_val'],
        dataset=experiment_configs['dataset'],
        number_of_samples=experiment_configs['number_of_samples'],
        augment=False,
        split='test')

    save_path_best = experiment_configs['save_folder'] + 'best.pth'
    save_path_current = experiment_configs['save_folder'] + 'current.pth'
    save_path_history = experiment_configs['save_folder'] + 'history.pkl'

    logging.info('Starting to train model...')
    model_path = None
    use_pretrained_model = False
    tuning='full_pass'
    if path.exists(save_path_current):
        logging.info('Model already exists, resuming training')
        model_path = save_path_current
        with open(save_path_history, 'rb') as f:
            evaluation_history = pickle.load(f)
    else:
        evaluation_history = {'average_precision': [], 'accuracy': [], 'DDP': []}
        if experiment_configs['pretrained_folder']:
            # Here, we are training using a pretrained model. Hence, we only tune the last layer.
            logging.info('Loading pretrained model')
            model_path = experiment_configs['pretrained_folder'] + 'best.pth'
            use_pretrained_model = True
            tuning='last_layer'

    if experiment_configs['fairness_notion'] == 'unconst':
        attribute_classifier = BinaryAttributeClassifier(experiment_configs['model'],
                                                         device=experiment_configs['device'],
                                                         tuning=tuning,
                                                         optimizer_settings=experiment_configs[
                                                             'optimizer_settings'],
                                                         dtype=experiment_configs['dtype'],
                                                         model_path=model_path,
                                                         print_freq=experiment_configs['print_freq'],
                                                         only_warmstart=use_pretrained_model,
                                                         two_headed=experiment_configs['two_headed'],
                                                         second_head_weight=experiment_configs['second_head_weight'])
    else:
        attribute_classifier = FairBinaryAttributeClassifier(experiment_configs['model'],
                                                             fairness_notion=experiment_configs[
                                                                 'fairness_notion'],
                                                             fairness_parameter=experiment_configs[
                                                                 'fairness_parameter'],
                                                             tuning=tuning,
                                                             device=experiment_configs['device'],
                                                             optimizer_settings=experiment_configs[
                                                                 'optimizer_settings'],
                                                             dtype=experiment_configs['dtype'],
                                                             model_path=model_path,
                                                             print_freq=experiment_configs['print_freq'],
                                                             only_warmstart=use_pretrained_model)

    # Train the attribute classifier
    for i in range(attribute_classifier.epoch, experiment_configs['total_epochs']):
        attribute_classifier.train(train_dataloader)

        # Evaluate on Validation Set
        y_true_val, y_score_val, val_loss = attribute_classifier.get_true_label_and_score_prediction(val_dataloader)
        attribute_classifier.scheduler.step(val_loss)

        prot_attr = y_true_val[:, 1]
        target_label = y_true_val[:, 0]
        if experiment_configs['two_headed']:
            target_score = y_score_val[:, 0]
            prot_attr_pred = np.where(y_score_val[:, 1] > 0.5, 1, 0)
            prot_attr_accuracy = accuracy_score(prot_attr, prot_attr_pred)
            logging.info('Val Protected Attribute Accuracy = {}'.format(prot_attr_accuracy))
        else:
            target_score = y_score_val

        y_pred_val = np.where(target_score > 0.5, 1, 0)
        average_precision = average_precision_score(target_label, target_score)
        accuracy = accuracy_score(target_label, y_pred_val)

        logging.info('Val Target Accuracy = {}'.format(accuracy))
        evaluation_history['accuracy'].append(accuracy)
        logging.info('Val Avg precision = {}'.format(average_precision))
        evaluation_history['average_precision'].append(average_precision)

        DDP_fairness = ut.get_DDP(y_pred_val, prot_attr)
        logging.info('Val DDP = {}'.format(DDP_fairness))
        evaluation_history['DDP'].append(DDP_fairness)

        with open(save_path_history, 'wb+') as handle:
            pickle.dump(evaluation_history, handle)
        if average_precision > attribute_classifier.best_eval_measure:
            attribute_classifier.best_eval_measure = average_precision
            attribute_classifier.save_model(save_path_best)
        attribute_classifier.save_model(save_path_current)


    # Evaluate attribute classifier on all splits
    attribute_classifier = BinaryAttributeClassifier(experiment_configs['model'],
                                                     optimizer_settings=None,
                                                     tuning='full_pass',
                                                     model_path=save_path_best,
                                                     two_headed=experiment_configs['two_headed'],
                                                     second_head_weight=experiment_configs['second_head_weight'],
                                                     dtype=experiment_configs['dtype'],
                                                     only_warmstart=True,
                                                     device=experiment_configs['device'])
    train_dataloader = load_data.get_dataloader(
                                experiment_configs['attribute'],
                                experiment_configs['protected_attribute'],
                                experiment_configs['data_settings']['params_val'],
                                dataset=experiment_configs['dataset'],
                                number_of_samples=experiment_configs['number_of_samples'],
                                augment=False,
                                stratified=False,
                                split='train')
    y_true_train, y_score_train, _ = attribute_classifier.get_true_label_and_score_prediction(train_dataloader)
    y_true_val, y_score_val, _ = attribute_classifier.get_true_label_and_score_prediction(val_dataloader)
    y_true_test, y_score_test, _ = attribute_classifier.get_true_label_and_score_prediction(test_dataloader)

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

    y_pred_train = np.where(y_score_train > 0.5, 1, 0)
    y_pred_val = np.where(y_score_val > 0.5, 1, 0)
    y_pred_test = np.where(y_score_test > 0.5, 1, 0)

    awareness_accuracy_val = None
    awareness_accuracy_test = None
    awareness_accuracy_train = None
    if experiment_configs['two_headed']:
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
    DDP_fairness = ut.get_DDP(y_pred_val, y_true_val[:, 1])

    val_results = {
        'opt': experiment_configs,
        'average_precision': average_precision,
        'accuracy': accuracy,
        'prot_attr_accuracy': awareness_accuracy_val,
        'DDP': DDP_fairness
    }

    logging.info('Validation results: ')
    logging.info(val_results)

    average_precision = average_precision_score(y_true_test[:, 0], y_score_test)
    accuracy = accuracy_score(y_true_test[:, 0], y_pred_test)
    DDP_fairness = ut.get_DDP(y_pred_test, y_true_test[:, 1])

    test_results = {
        'opt': experiment_configs,
        'average_precision': average_precision,
        'accuracy': accuracy,
        'prot_attr_accuracy': awareness_accuracy_test,
        'DDP': DDP_fairness
    }

    logging.info('Test results: ')
    logging.info(test_results)

    average_precision = average_precision_score(y_true_train[:, 0], y_score_train)
    accuracy = accuracy_score(y_true_train[:, 0], y_pred_train)
    DDP_fairness = ut.get_DDP(y_pred_train, y_true_train[:, 1])

    train_results = {
        'opt': experiment_configs,
        'average_precision': average_precision,
        'accuracy': accuracy,
        'prot_attr_accuracy': awareness_accuracy_train,
        'DDP': DDP_fairness,
    }

    logging.info('Train results: ')
    logging.info(train_results)

    with open(experiment_configs['save_folder'] + 'train_results.pkl', 'wb+') as handle:
        pickle.dump(train_results, handle)
    with open(experiment_configs['save_folder'] + 'val_results.pkl', 'wb+') as handle:
        pickle.dump(val_results, handle)
    with open(experiment_configs['save_folder'] + 'test_results.pkl', 'wb+') as handle:
        pickle.dump(test_results, handle)


if __name__ == "__main__":
    configs = parse_experiment_args.parse_arguments()
    main(configs)
