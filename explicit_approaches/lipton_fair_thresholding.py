import numpy as np
from utils.func_utils import get_DDP

def sigmoid(X):
   return 1/(1+np.exp(-X))


def learn_fair_threshold_classifier(y_score, y_true, fairness_parameter):

    #assumes binary classification with values in {0,+1} and +1 to be the preferred outcome
    #assumes y_score to comprise x_1 and x_2 where an (unfair) target classifier would predict +1 iff x_1>0 
    # and x_2 is used to predict the sensitive attribute
    #assumes fairness_parameter in [0,1]

    #RETURNS: tuple (Threshoold of advantaged group, Threshold of disadvantaged group); thresholds are to be applied
    #         before applying sigmoid function

    labels = y_true[:, 0]
    sensitive_features = y_true[:, 1]
    sensitive_feature_values = np.unique(sensitive_features)
    n = len(labels)
    score = y_score[:, 0]    #this is x_1
    unfair_pred = np.zeros(score.shape)
    unfair_pred[score > 0] = 1

    if len(np.unique(unfair_pred)) == 1:
        print('Given predictor is constant')
        return 0, 0

    if np.mean(unfair_pred[sensitive_features == sensitive_feature_values[0]] == 1) > np.mean(
            unfair_pred[sensitive_features == sensitive_feature_values[1]] == 1):
        disadvantaged_group = 1
        advantaged_group = 0
    else:
        disadvantaged_group = 0
        advantaged_group = 1

    indices_adv_group = np.arange(n)[sensitive_features == sensitive_feature_values[advantaged_group]]
    indices_disadv_group = np.arange(n)[sensitive_features == sensitive_feature_values[disadvantaged_group]]
    n_a = len(indices_adv_group)
    n_b = len(indices_disadv_group)

    indices_adv_group_and_unfair_pred_equals_1 = indices_adv_group[unfair_pred[indices_adv_group] == 1]
    indices_disadv_group_and_unfair_pred_equals_0 = indices_disadv_group[unfair_pred[indices_disadv_group] == 0]

    c_adv_group = fairness_parameter/(n_a*(2*sigmoid(score[indices_adv_group_and_unfair_pred_equals_1])-1))
    c_disadv_group = 1/(n_b * (1-2 * sigmoid(score[indices_disadv_group_and_unfair_pred_equals_0])))

    predictions = unfair_pred.copy()
    p_gap = fairness_parameter*np.mean(predictions[indices_adv_group])-np.mean(predictions[indices_disadv_group])

    new_thresholds = [0, 0]

    while p_gap >= 0:
        max_c_adv = np.amax(c_adv_group)
        max_c_disadv = np.amax(c_disadv_group)

        if max_c_adv > max_c_disadv:
            arg_max = np.argmax(c_adv_group)
            c_adv_group[arg_max] = -np.inf
            predictions[indices_adv_group_and_unfair_pred_equals_1[arg_max]]=0
            arg_max_2 = np.argmax(c_adv_group)
            if c_adv_group[arg_max_2] > (-np.inf):
                new_thresholds[advantaged_group] = (score[indices_adv_group_and_unfair_pred_equals_1[arg_max]] +
                                 score[indices_adv_group_and_unfair_pred_equals_1[arg_max_2]])/2
            else:
                new_thresholds[advantaged_group] = np.inf

        else:
            arg_max = np.argmax(c_disadv_group)
            c_disadv_group[arg_max] = -np.inf
            predictions[indices_disadv_group_and_unfair_pred_equals_0[arg_max]] = 1
            arg_max_2 = np.argmax(c_disadv_group)
            if c_disadv_group[arg_max_2] > (-np.inf):
                new_thresholds[disadvantaged_group] = (score[indices_disadv_group_and_unfair_pred_equals_0[arg_max]] +
                                       score[indices_disadv_group_and_unfair_pred_equals_0[arg_max_2]]) / 2
            else:
                new_thresholds[disadvantaged_group] = -np.inf

        p_gap = fairness_parameter * np.mean(predictions[indices_adv_group]) - np.mean(
            predictions[indices_disadv_group])

    return new_thresholds


if __name__ == "__main__":

    n = np.random.randint(100, 500)
    y_true = np.random.randint(0, 2, size=(n, 2))
    y_score = np.random.normal(size=(n, 2))
    y_score[y_true[:, 1] == 0, 0] += 0.1
    y_score[y_true[:, 1] == 1, 0] -= 0.5
    fairness_parameter = 1

    print(n, fairness_parameter)
    new_thresholds = learn_fair_threshold_classifier(y_score, y_true, fairness_parameter)
    print(new_thresholds)
    print(get_DDP(y_score[:, 0] > 0, y_true[:, 1]))
    print(get_DDP(np.where(np.logical_or(np.logical_and(y_true[:, 1] == 0, y_score[:, 0] > new_thresholds[0]),
                                         np.logical_and(y_true[:, 1] == 1, y_score[:, 0] > new_thresholds[1])), 1, 0),
                                         y_true[:, 1]))