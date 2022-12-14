import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from os import path
import numpy as np
from utils import load_data
from utils.func_utils import make_dir
from utils.func_utils import get_model_path,get_DDP, get_constant_accuracy, invert_sigmoid, get_acc
import pickle
import matplotlib.pyplot as plt

plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# SMALL_SIZE = 28
# MEDIUM_SIZE = 30
# BIGGER_SIZE = 32
#
MARKER_SIZE = 30
#
# plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
# plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

attr_list = load_data.get_all_celeba_attributes()

markers_per_notion = {'DDP_squared': 's', 'DDP_abs': 'o'}
legend_name_per_notion = {'DDP_squared': r'Regularizer $\widehat{\mathcal{R}}_{\mathrm{DP}}$',
                          'DDP_abs': r'Regularizer $\widehat{\mathcal{R}}^{\mathrm{abs}}_{\mathrm{DP}}$'}


def load_explicit_approach_results(results_path,
                                   explicit_approach='two_headed'):
    # or explicit_approach = 'lipton'

    if explicit_approach == 'lipton':
        results_path += 'lipton_results.pkl'
    else:
        results_path += 'grid_search_weighted_heads.pkl'
    if path.exists(results_path):
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        return results
    else:
        print('path missing')
        print(results_path)
        return None


def get_fair_models_results_data(model='mobilenet',
                                 dataset='celebA',
                                 target_attr_nmb=2,
                                 protected_attr_nmb=20,
                                 regularizers=None,
                                 fairness_parameters=None,
                                 split='test',
                                 verbose=False):
    if regularizers is None:
        regularizers = ['DDP_squared']
    if fairness_parameters is None:
        fairness_parameters = [0.0]

    if dataset == 'celebA':
        attr_list = load_data.get_all_celeba_attributes()
        lr = 1e-4
    elif dataset == 'fairface':
        attr_list = load_data.get_all_fairface_attributes()
        lr = 1e-5

    seed_of_pretrained = 0
    seed_of_finetuned = 0

    protected_attribute = attr_list[protected_attr_nmb]
    target_attribute = attr_list[target_attr_nmb]

    optimizer_settings = {
        'learning_rate': lr,
        'stratified': True
    }
    prot_attr_clf_optimizer_settings = {
        'learning_rate': 1e-4,
        'stratified': False
    }

    fair_pretrained_models = {}
    for regularizer in regularizers:
        # Fairness Parameter, Acc on prot. attr., target acc, target fairness
        data_results_fair_models = np.zeros(shape=(len(fairness_parameters), 4))
        data_results_fair_models[:] = None
        for fair_iter, fairness_param in enumerate(fairness_parameters):
            if fairness_param == 0.0:
                tmp_regularizer = 'unconst'
            else:
                tmp_regularizer = regularizer
            prot_attr_clf_path = get_model_path(dataset=dataset,
                                                protected_attribute=protected_attribute,
                                                target_attribute=protected_attribute,
                                                model_name=model,
                                                tuning_method='last_layer',
                                                seed=seed_of_finetuned,
                                                optimizer_settings=prot_attr_clf_optimizer_settings.copy(),
                                                fairness='unconst',
                                                pretrained_on=target_attribute,
                                                pretrained_seed=seed_of_pretrained,
                                                fair_backbone=tmp_regularizer,
                                                fair_backbone_parameter=fairness_param,
                                                backbone_stratified=optimizer_settings['stratified']) + '/'
            prot_attr_clf_scores_path = prot_attr_clf_path + 'y_score_' + split + '.pkl'
            prot_attr_clf_true_path = prot_attr_clf_path + 'y_true_' + split + '.pkl'
            model_path = get_model_path(dataset=dataset,
                                        protected_attribute=protected_attribute,
                                        target_attribute=target_attribute,
                                        model_name=model,
                                        tuning_method='full_pass',
                                        seed=seed_of_pretrained,
                                        optimizer_settings=optimizer_settings.copy(),
                                        fairness=tmp_regularizer,
                                        fairness_parameter=fairness_param,
                                        pretrained_on='imagenet') + '/'
            model_scores_path = model_path + 'y_score_' + split + '.pkl'
            model_true_path = model_path + 'y_true_' + split + '.pkl'

            if path.exists(model_scores_path):
                data_results_fair_models[fair_iter, 0] = fairness_param
                if path.exists(prot_attr_clf_scores_path):
                    with open(prot_attr_clf_scores_path, 'rb') as f:
                        prot_attr_score = pickle.load(f)
                    with open(prot_attr_clf_true_path, 'rb') as f:
                        prot_attr_true = pickle.load(f)
                    data_results_fair_models[fair_iter, 1] = get_acc(prot_attr_score, prot_attr_true[:, 0])
                elif verbose:
                    print(
                        f'Protected Attr Classifier Models missing for parameter {fairness_param}, prot attr {protected_attribute}, target {target_attribute}')
                    print(prot_attr_clf_scores_path)

                with open(model_scores_path, 'rb') as f:
                    y_score = pickle.load(f)
                with open(model_true_path, 'rb') as f:
                    y_true = pickle.load(f)
                data_results_fair_models[fair_iter, 2] = get_acc(y_score, y_true[:, 0])
                data_results_fair_models[fair_iter, 3] = get_DDP(np.where(y_score > 0.5, 1, 0), y_true[:, 1])
            else:
                print(f'Fair Models missing for parameter {fairness_param}, prot attr {protected_attribute}, target {target_attribute}')
                print(model_scores_path)

        fair_pretrained_models[regularizer] = data_results_fair_models

    constant_accuracy = get_constant_accuracy(y_true[:, 0])
    ground_truth_DDP = get_DDP(y_true[:, 0], y_true[:, 1])

    return fair_pretrained_models, constant_accuracy, ground_truth_DDP


def get_explicit_approaches_results_data(model='mobilenet',
                                         dataset='celebA',
                                         target_attr_nmb=2,
                                         protected_attr_nmb=20,
                                         explicit_approach='two_headed',
                                         split='test',
                                         verbose=False):
    """

    :param model:
    :param dataset:
    :param target_attr_nmb:
    :param protected_attr_nmb:
    :param lipton:
    :param split:
    :param verbose:
    :return:
    """

    if dataset == 'celebA':
        attr_list = load_data.get_all_celeba_attributes()
        lr = 1e-4
    elif dataset == 'fairface':
        attr_list = load_data.get_all_fairface_attributes()
        lr = 1e-5
    seed = 0

    protected_attribute = attr_list[protected_attr_nmb]
    target_attribute = attr_list[target_attr_nmb]

    optimizer_settings = {
        'learning_rate': lr,
        'stratified': False
    }
    # Todo: clarify if two headed model is stratified
    # if model == 'resnet50':
    #     optimizer_settings['stratified'] = False

    # Here, I am only using unconstrained models as the base
    model_path = get_model_path(target_attribute=target_attribute,
                                dataset=dataset,
                                protected_attribute=protected_attribute,
                                model_name=model,
                                tuning_method='full_pass',
                                seed=seed,
                                optimizer_settings=optimizer_settings.copy(),
                                fairness='unconst',
                                two_headed=True,
                                pretrained_on='imagenet') + '/'
    aware_model_scores_path = model_path + 'y_score_' + split + '.pkl'
    aware_model_true_path = model_path + 'y_true_' + split + '.pkl'
    best_weighted_average_model_results = load_explicit_approach_results(results_path=model_path,
                                                                           explicit_approach=explicit_approach)
    if not best_weighted_average_model_results is None:

        fairness_slack_range = best_weighted_average_model_results.keys()
        # Fairness Slack, Acc on prot. attr., target acc, target fairness, FairLoss, target acc, and target fairness of best average model
        results_average_models = np.zeros(shape=(len(fairness_slack_range), 6))
        results_average_models[:] = None
        for fair_iter, fairness_slack in enumerate(fairness_slack_range):
            if path.exists(aware_model_scores_path):
                results_average_models[fair_iter, 0] = fairness_slack
                with open(aware_model_scores_path, 'rb') as f:
                    y_score = pickle.load(f)
                with open(aware_model_true_path, 'rb') as f:
                    y_true = pickle.load(f)
                y_score[:, 1] = invert_sigmoid(y_score[:, 1])


                results_average_models[fair_iter, 1] = get_acc(y_score[:, 1], y_true[:, 1])  # reconstruction of sens attr
                results_average_models[fair_iter, 2] = get_acc(y_score[:, 0], y_true[:, 0])  # target_acc of aware model (only target head)
                results_average_models[fair_iter, 3] = get_DDP(np.where(y_score[:, 0] > 0.5, 1, 0), y_true[:, 1])

                if explicit_approach == 'lipton':
                    lipton_thresholds = best_weighted_average_model_results[fairness_slack]['lipton_thresholds']

                    y_score[:, 0] = invert_sigmoid(y_score[:, 0])
                    # s_pred = np.where(y_score[:, 1] > 0.5, 1, 0)
                    y_pred_lipton = np.where(np.logical_or(np.logical_and(y_true[:, 1] == 0, y_score[:, 0] > lipton_thresholds[0]),
                                                           np.logical_and(y_true[:, 1] == 1, y_score[:, 0] > lipton_thresholds[1])), 1, 0)
                    results_average_models[fair_iter, 4] = get_acc(y_true[:, 0], y_pred_lipton)
                    results_average_models[fair_iter, 5] = get_DDP(y_pred_lipton, y_true[:, 1])
                else:
                    # target_acc of average model (using also second head)
                    results_average_models[fair_iter, 4] = best_weighted_average_model_results[fairness_slack][split]['accuracy']
                    results_average_models[fair_iter, 5] = best_weighted_average_model_results[fairness_slack][split]['DDP']
            else:
                if verbose:
                    print(f'Aware Models missing for unconstrained model, prot attr {protected_attribute}, target {target_attribute}')
                    print(aware_model_scores_path)

        constant_accuracy = get_constant_accuracy(y_true[:, 0])
        ground_truth_DDP = get_DDP(y_true[:, 0], y_true[:, 1])
    else:
        results_average_models = np.zeros(shape=(1, 7))
        results_average_models[:] = None
        constant_accuracy = None
        ground_truth_DDP = None

    return results_average_models, constant_accuracy, ground_truth_DDP

def scatter_plot_fairness_vs_awareness(dataset='celebA',
                                       target_attr_nmb=2,
                                       protected_attr_nmb=20,
                                       regularizers=None,
                                       fairness_parameters=None,
                                       split='test',
                                       model='mobilenet',
                                       verbose=False,
                                       fig_kwargs={"figsize": (5, 5)}):
    if dataset == 'celebA':
        attr_list = load_data.get_all_celeba_attributes()
    elif dataset == 'fairface':
        attr_list = load_data.get_all_fairface_attributes()
    protected_attribute = attr_list[protected_attr_nmb]
    target_attribute = attr_list[target_attr_nmb]
    save_folder = 'plots/exp_' + dataset + '/fairness_group_accuracy/prot_' + protected_attribute + '_target_' + target_attribute + '/'

    fair_models_results, constant_accuracy, ground_truth_DDP = get_fair_models_results_data(dataset=dataset,
                                                                                            target_attr_nmb=target_attr_nmb,
                                                                                            protected_attr_nmb=protected_attr_nmb,
                                                                                            regularizers=regularizers,
                                                                                            fairness_parameters=fairness_parameters,
                                                                                            split=split,
                                                                                            model=model,
                                                                                            verbose=verbose)

    fig, ax = plt.subplots(1, 1, **fig_kwargs)

    fig.suptitle('Classification Task ' + target_attribute + ' \n Protected Attribute ' + protected_attribute)

    ax.set_xlabel('demographic disparity (DDP)')
    ax.set_ylabel('protected attribute accuracy')
    for regularizer in regularizers:
        # Fairness Parameter, Acc on prot. attr., target acc, target fairness, FairLoss
        ax.scatter(fair_models_results[regularizer][1:, 3],
                   fair_models_results[regularizer][1:, 1], marker=markers_per_notion[regularizer],
                   s=MARKER_SIZE, label='regularized model')
        ax.scatter(fair_models_results[regularizer][0, 3],
                   fair_models_results[regularizer][0, 1], marker='*', color='orange',
                   s=MARKER_SIZE*4, label='unconstrained model')
        for xpos, ypos, fair_param in zip(fair_models_results[regularizer][1:5, 3],
                                          fair_models_results[regularizer][1:5, 1],
                                          fair_models_results[regularizer][1:5, 0]):
            ax.annotate(r"   $\lambda = {}$".format(str(fair_param)), (xpos, ypos),
                        fontsize='medium', va='top', ha='left')

    for pos in ['right', 'top']:
        ax.spines[pos].set_visible(False)

    ax.set_ylim(min(fair_models_results[regularizer][:, 1]) - 0.01, max(fair_models_results[regularizer][:, 1]) + 0.01)
    ax.legend(frameon=False, bbox_to_anchor=(-0.05, 1.0), loc="upper left")
    make_dir(save_folder)
    plt.savefig(save_folder + '/' + model + '_reg_' + '_'.join(regularizers) + '.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def scatter_plot_fairness_vs_accuracy(dataset='celebA',
                                      target_attr_nmb=2,
                                      protected_attr_nmb=20,
                                      regularizers=None,
                                      fairness_parameters=None,
                                      explicit_approaches=None,
                                      model='mobilenet',
                                      split='test',
                                      verbose=False,
                                      fig_kwargs={"figsize": (5, 5)}):
    if dataset == 'celebA':
        attr_list = load_data.get_all_celeba_attributes()
    elif dataset == 'fairface':
        attr_list = load_data.get_all_fairface_attributes()
    if explicit_approaches is None:
        explicit_approaches = []
    if regularizers is None:
        regularizers = []
    protected_attribute = attr_list[protected_attr_nmb]
    target_attribute = attr_list[target_attr_nmb]
    save_folder = 'plots/exp_' + dataset + '/fairness_accuracy/prot_' + protected_attribute + '_target_' + target_attribute + '/'

    if regularizers:
        fair_models_results, constant_accuracy, ground_truth_DDP = get_fair_models_results_data(dataset=dataset,
                                                                                                target_attr_nmb=target_attr_nmb,
                                                                                                protected_attr_nmb=protected_attr_nmb,
                                                                                                regularizers=regularizers,
                                                                                                fairness_parameters=fairness_parameters,
                                                                                                split=split,
                                                                                                model=model,
                                                                                                verbose=verbose)
    if 'two_headed' in explicit_approaches:
        aware_models_results, constant_accuracy, ground_truth_DDP = get_explicit_approaches_results_data(dataset=dataset,
                                                                                                         target_attr_nmb=target_attr_nmb,
                                                                                                         protected_attr_nmb=protected_attr_nmb,
                                                                                                         explicit_approach='two_headed',
                                                                                                         split=split,
                                                                                                         model=model,
                                                                                                         verbose=verbose)
    if 'lipton' in explicit_approaches:
        lipton_models_results, constant_accuracy, ground_truth_DDP = get_explicit_approaches_results_data(dataset=dataset,
                                                                                                          target_attr_nmb=target_attr_nmb,
                                                                                                          protected_attr_nmb=protected_attr_nmb,
                                                                                                          explicit_approach='lipton',
                                                                                                          split=split,
                                                                                                          model=model,
                                                                                                          verbose=verbose)

    fig, ax = plt.subplots(1, 1, **fig_kwargs)

    fig.suptitle('Classification Task ' + target_attribute + '\n with Protected Attribute ' + protected_attribute)

    ax.set_xlabel('Demographic Disparity (DDP)')
    ax.set_ylabel('Classification Accuracy for ' + target_attribute)

    if regularizers:
        for regularizer in regularizers:
            # Fairness Parameter, Acc on prot. attr., target acc, target fairness, FairLoss
            ax.scatter(fair_models_results[regularizer][1:, 3],
                       fair_models_results[regularizer][1:, 2], marker=markers_per_notion[regularizer],
                       s=MARKER_SIZE, label=legend_name_per_notion[regularizer])
        ax.scatter(fair_models_results[regularizers[0]][0, 3],
                   fair_models_results[regularizers[0]][0, 2], marker='*',
                   s=MARKER_SIZE*3, label='Unconstrained')

    if 'two_headed' in explicit_approaches:
        ax.scatter(aware_models_results[:, 5],
                   aware_models_results[:, 4], marker='d', s=MARKER_SIZE*2, label='Our Approach', color='r')
    if 'lipton' in explicit_approaches:
        ax.scatter(lipton_models_results[:, 5],
                   lipton_models_results[:, 4], marker='X', s=MARKER_SIZE, label='Lipton et al.')

    ax.scatter(0, constant_accuracy, color='k', marker='o', s=MARKER_SIZE, label='Constant')
    ax.axvline(x=ground_truth_DDP, color='k', linestyle=':', label='True Label DDP')

    for pos in ['right', 'top']:
        ax.spines[pos].set_visible(False)

    plt.subplots_adjust(wspace=0.3)
    ax.legend(frameon=False)  # loc='upper left'
    make_dir(save_folder)
    plt.savefig(save_folder + '/' + model + '_reg_' + '_'.join(regularizers) + '.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

if __name__ == '__main__':


    dataset = 'celebA'
    model = 'resnet50'
    protected_attr_numbers = [20]  # 20, 2, 39, 31, 8
    target_attr_numbers = [20, 2, 31, 8, 3, 5, 15, 19, 39]  # 20, 2, 39, 31, 8, 3, 5, 15, 19

    for target_attr_nmb in target_attr_numbers:
        for protected_attr_nmb in protected_attr_numbers:
            if target_attr_nmb != protected_attr_nmb:
                scatter_plot_fairness_vs_accuracy(model=model, dataset=dataset, target_attr_nmb=target_attr_nmb, protected_attr_nmb=protected_attr_nmb,
                                                  explicit_approaches=['ours', 'lipton'],
                                                  regularizers=['DDP_matkle'], fairness_notion='DDP', split='test', verbose=True)
                scatter_plot_fairness_vs_awareness(model=model, dataset=dataset, target_attr_nmb=target_attr_nmb, protected_attr_nmb=protected_attr_nmb,
                                                   fairness_notion='DDP', regularizers=['DDP_matkle'], split='test',
                                                   verbose=True)
