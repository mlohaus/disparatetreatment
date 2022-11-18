from os import path
import numpy as np
from utils import load_data
from utils.func_utils import get_model_path, get_DDP, invert_sigmoid
import pickle
import argparse
import logging
from sklearn.metrics import accuracy_score
from explicit_approaches.lipton_fair_thresholding import learn_fair_threshold_classifier
from explicit_approaches.grid_search_two_heads import learn_fair_classifier_via_grid_search_RECURSIVE


def collect_args_main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', dest='model', default='mobilenet')
    parser.add_argument('--dataset', dest='dataset', default='celebA')
    parser.add_argument('--stratified', dest='stratified', action='store_true')
    parser.add_argument('--attribute', type=int, default=31)
    parser.add_argument('--protected_attribute', type=int, default=20)
    exp_configs = vars(parser.parse_args())

    return exp_configs


def get_pvalue(y_pred, sens_label):
    if np.average(np.where(y_pred[sens_label == 0] == 1, 1, 0)) > np.average(np.where(y_pred[sens_label == 1] == 1, 1, 0)) > 0:
        return np.average(np.where(y_pred[sens_label == 1] == 1, 1, 0)) / np.average(np.where(y_pred[sens_label == 0] == 1, 1, 0))
    else:
        return np.average(np.where(y_pred[sens_label == 0] == 1, 1, 0)) / np.average(np.where(y_pred[sens_label == 1] == 1, 1, 0))


def get_two_headed_model_scores(target_attr_nmb=2,
                                protected_attr_nmb=20,
                                dataset='celebA',
                                stratified=True,
                                model='mobilenet',
                                split='test',
                                seed=0):
    if dataset == 'celebA':
        attr_list = load_data.get_all_celeba_attributes()
        lr = 1e-4
    elif dataset == 'fairface':
        attr_list = load_data.get_all_fairface_attributes()
        lr = 1e-5

    protected_attribute = attr_list[protected_attr_nmb]
    target_attribute = attr_list[target_attr_nmb]

    optimizer_settings = {
        'learning_rate': lr,
        'stratified':  stratified
    }

    model_path = get_model_path(target_attribute=target_attribute,
                                protected_attribute=protected_attribute,
                                dataset=dataset,
                                model_name=model,
                                tuning_method='full_pass',
                                seed=seed,
                                optimizer_settings=optimizer_settings,
                                fairness='unconst',
                                two_headed=True,
                                pretrained_on='imagenet') + '/'
    if path.exists(model_path + 'y_score_' + split + '.pkl'):
        with open(model_path + 'y_score_' + split + '.pkl', 'rb') as f:
            y_score = pickle.load(f)
        with open(model_path + 'y_true_' + split + '.pkl', 'rb') as f:
            y_true = pickle.load(f)
        return y_true, y_score, model_path
    else:
        logging.info('File does not exist:')
        logging.info(model_path + 'y_score_' + split + '.pkl')
        return None, None, None


def get_unconst_model_scores(target_attr_nmb=2,
                             protected_attr_nmb=20,
                             dataset='celebA',
                             stratified=True,
                             model='mobilenet',
                             split='test',
                             seed=0):
    if dataset == 'celebA':
        attr_list = load_data.get_all_celeba_attributes()
        lr = 1e-4
    elif dataset == 'fairface':
        attr_list = load_data.get_all_fairface_attributes()
        lr = 1e-5

    protected_attribute = attr_list[protected_attr_nmb]
    target_attribute = attr_list[target_attr_nmb]

    optimizer_settings = {
        'learning_rate': lr,
        'stratified': stratified
    }

    model_path = get_model_path(target_attribute=target_attribute,
                                protected_attribute=protected_attribute,
                                dataset=dataset,
                                model_name=model,
                                tuning_method='full_pass',
                                seed=seed,
                                optimizer_settings=optimizer_settings,
                                fairness='unconst',
                                pretrained_on='imagenet') + '/'
    if path.exists(model_path + 'y_score_' + split + '.pkl'):
        with open(model_path + 'y_score_' + split + '.pkl', 'rb') as f:
            y_score = pickle.load(f)
        with open(model_path + 'y_true_' + split + '.pkl', 'rb') as f:
            y_true = pickle.load(f)
        return y_true, y_score
    else:
        logging.info('File does not exist:')
        logging.info(model_path + 'y_score_' + split + '.pkl')
        return None, None


def learn_and_save_new_lipton_thresholds(target_attr_nmb=2,
                                         protected_attr_nmb=20,
                                         dataset='celebA',
                                         stratified=True,
                                         grid_size_for_fairness_slack=10,
                                         model='mobilenet'
                                         ):

    y_true_train, y_score_train, model_path = get_two_headed_model_scores(target_attr_nmb=target_attr_nmb,
                                                                          protected_attr_nmb=protected_attr_nmb,
                                                                          dataset=dataset,
                                                                          stratified=stratified,
                                                                          model=model,
                                                                          split='val')
    y_true_unconst, y_score_unconst = get_unconst_model_scores(target_attr_nmb=target_attr_nmb,
                                                               protected_attr_nmb=protected_attr_nmb,
                                                               dataset=dataset,
                                                               model=model,
                                                               split='test')

    if model_path:

        unconstrained_fairness_value = get_pvalue(np.where(y_score_unconst > 0.5, 1, 0), y_true_unconst[:, 1])

        results_per_fairness_slack = {}
        for fairness_slack in np.linspace(unconstrained_fairness_value, 1, num=grid_size_for_fairness_slack, endpoint=True):

            new_thresholds = learn_fair_threshold_classifier(invert_sigmoid(y_score_train), y_true_train, fairness_slack)

            results_overall = {}
            results_overall['lipton_thresholds'] = new_thresholds
            for split in ['val', 'test']:
                y_true, y_score, _ = get_two_headed_model_scores(target_attr_nmb=target_attr_nmb,
                                                                 protected_attr_nmb=protected_attr_nmb,
                                                                 dataset=dataset,
                                                                 stratified=stratified,
                                                                 model=model,
                                                                 split=split)

                y_score = invert_sigmoid(y_score)
                s_pred = np.where(y_score[:, 1] > 0.5, 1, 0)
                y_pred_lipton = np.where(np.logical_or(np.logical_and(s_pred == 0, y_score[:, 0] > new_thresholds[0]),
                                                       np.logical_and(s_pred == 1, y_score[:, 0] > new_thresholds[1])), 1, 0)

                accuracy_lipton = accuracy_score(y_true[:, 0], y_pred_lipton)
                fairness_lipton = get_DDP(y_pred_lipton, y_true[:, 1])
                results_overall[split] = {'DDP': fairness_lipton, 'accuracy': accuracy_lipton}

            results_per_fairness_slack[fairness_slack] = results_overall

        with open(model_path + 'lipton_results.pkl', 'wb+') as handle:
            pickle.dump(results_per_fairness_slack, handle)
            logging.info('Dumped files')
            logging.info(model_path + 'lipton_results.pkl')


def learn_and_save_best_weighted_model(target_attr_nmb=2,
                                       protected_attr_nmb=20,
                                       dataset='celebA',
                                       stratified=True,
                                       grid_size_for_fairness_slack=11,
                                       model='mobilenet'
                                       ):

    y_true_train, y_score_train, model_path = get_two_headed_model_scores(target_attr_nmb=target_attr_nmb,
                                                                          protected_attr_nmb=protected_attr_nmb,
                                                                          dataset=dataset,
                                                                          stratified=stratified,
                                                                          model=model,
                                                                          split='test')
    y_true_unconst, y_score_unconst = get_unconst_model_scores(target_attr_nmb=target_attr_nmb,
                                                               protected_attr_nmb=protected_attr_nmb,
                                                               dataset=dataset,
                                                               model=model,
                                                               split='val')

    if model_path:
        unconstrained_fairness_value = np.abs(get_DDP(np.where(y_score_unconst > 0.5, 1, 0), y_true_unconst[:, 1]))

        results_per_fairness_slack = {}
        for fairness_slack in np.linspace(0.001, unconstrained_fairness_value, num=grid_size_for_fairness_slack, endpoint=False):

            coef_target, coef_protected, intercept = learn_fair_classifier_via_grid_search_RECURSIVE(invert_sigmoid(y_score_train),
                                                                                                     y_true_train,
                                                                                                     fairness_slack)
            if coef_target is None:
                print('ERROR')
                coef_target = 1.0
                coef_protected = 0.0
                intercept = 0.0

            results_overall = {'coef_': [coef_target, coef_protected], 'intercept_': intercept}
            for split in ['val', 'test']:
                y_true, y_score, _ = get_two_headed_model_scores(target_attr_nmb=target_attr_nmb,
                                                                 protected_attr_nmb=protected_attr_nmb,
                                                                 dataset=dataset,
                                                                 stratified=stratified,
                                                                 model=model,
                                                                 split=split)
                y_score = invert_sigmoid(y_score)
                y_pred_new = np.where(coef_target * y_score[:, 0] + coef_protected * y_score[:, 1] + intercept > 0, 1, 0)
                accuracy_new = accuracy_score(y_true[:, 0], y_pred_new)
                fairness_DDP = get_DDP(y_pred_new, y_true[:, 1])

                results_overall[split] = {'DDP': fairness_DDP, 'accuracy': accuracy_new}
                logging.info(fairness_slack)
                logging.info(split + ' --- DDP: ' + str(fairness_DDP) + ', Acc: ' + str(accuracy_new))
            results_per_fairness_slack[fairness_slack] = results_overall

        with open(model_path + 'grid_search_weighted_heads.pkl', 'wb+') as handle:
            pickle.dump(results_per_fairness_slack, handle)
            logging.info('Dumped files')
            logging.info(model_path + 'grid_search_weighted_heads.pkl')


def run_experiment(configs):
    model = configs['model']
    protected_attr_nmb = configs['protected_attribute']
    target_attr_nmb = configs['attribute']
    dataset = configs['dataset']
    stratified = configs['stratified']

    logging.basicConfig(filename='logs/explicit_approaches_' + model + '.log', filemode='w', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    learn_and_save_best_weighted_model(dataset=dataset,
                                       target_attr_nmb=target_attr_nmb,
                                       protected_attr_nmb=protected_attr_nmb,
                                       stratified=stratified,
                                       grid_size_for_fairness_slack=21,
                                       model=model)
    learn_and_save_new_lipton_thresholds(dataset=dataset,
                                         target_attr_nmb=target_attr_nmb,
                                         protected_attr_nmb=protected_attr_nmb,
                                         stratified=stratified,
                                         grid_size_for_fairness_slack=21,
                                         model=model)


if __name__ == '__main__':

    configs = collect_args_main()
    run_experiment(configs)
