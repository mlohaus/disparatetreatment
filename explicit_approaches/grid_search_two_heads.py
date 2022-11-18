from utils.func_utils import get_DDP
from sklearn.metrics import accuracy_score
import numpy as np


def learn_fair_classifier_via_grid_search(y_score,
                                          y_true,
                                          fairness_parameter,
                                          grid_range_b=(-10, 10),
                                          grid_size_b=101,
                                          grid_range_c=(-10, 10),
                                          grid_size_c=101):

    # assumes binary classification with values in {0,+1} and +1 to be the preferred outcome
    # assumes y_score to comprise x_1 and x_2 where an (unfair) target classifier would predict +1 iff x_1>0
    # and x_2 is used to predict the sensitive attribute
    # assumes fairness_parameter in [0,1]

    # RETURNS: coefficients (a=1,b,c)

    labels = y_true[:, 0]
    sensitive_features = y_true[:, 1]
    sensitive_feature_values = np.unique(sensitive_features)

    best_b = np.nan
    best_c = np.nan
    best_acc = 0
    found_at_least_one = False

    for cand_b in np.linspace(grid_range_b[0], grid_range_b[1], grid_size_b):
        for cand_c in np.linspace(grid_range_c[0], grid_range_c[1], grid_size_c):

            pred = np.where(y_score[:, 0] + cand_b * y_score[:, 1] + cand_c > 0, 1, 0)

            fairness_violation = np.abs(np.mean(pred[sensitive_features == sensitive_feature_values[0]]) -
                                 np.mean(pred[sensitive_features == sensitive_feature_values[1]]))

            if fairness_violation <= fairness_parameter:
                found_at_least_one = True
                acc = np.mean(labels == pred)
                if acc > best_acc:
                    best_b = cand_b
                    best_c = cand_c
                    best_acc = acc

    if not found_at_least_one:
        print('No classifier in range of consideration satisfies fairness constraint')
        return None
    else:
        return 1, best_b, best_c


def learn_fair_classifier_via_grid_search_RECURSIVE(y_score, y_true,
                                                    fairness_parameter,
                                                    grid_range_b=(-15, 15),
                                                    grid_range_c=(-15, 15),
                                                    grid_size=200,
                                                    nr_of_recursions=4):
    g_range_b = grid_range_b
    step_size_b = (grid_range_b[1] - grid_range_b[0]) / (grid_size - 1)
    g_range_c = grid_range_c
    step_size_c = (grid_range_c[1] - grid_range_c[0]) / (grid_size - 1)

    ret = None
    for ell in range(nr_of_recursions):
        ret = learn_fair_classifier_via_grid_search(y_score, y_true, fairness_parameter, g_range_b, grid_size, g_range_c, grid_size)
        if ret is None:
            print('No classifier in range of consideration satisfies fairness constraint')
            return None, None, None
        else:
            g_range_b = (ret[1] - step_size_b, ret[1] + step_size_b)
            g_range_c = (ret[2] - step_size_c, ret[1] + step_size_c)

    return ret


def test_routine():
    n = np.random.randint(1000, 5000)
    y_true = np.random.randint(0, 2, size=(n, 2))
    y_score = np.random.normal(size=(n, 2))
    y_score[y_true[:, 1] == 0, 0] += 0.1
    y_score[y_true[:, 1] == 1, 0] -= 0.5
    fairness_parameter = 0.23
    y_pred = np.where(y_score[:, 0] > 0, 1, 0)
    print(n, fairness_parameter)
    print('DDP before', get_DDP(y_pred, y_true[:, 1]))
    print(accuracy_score(y_score[:, 0] > 0, y_true[:, 0]))

    a, b, c = learn_fair_classifier_via_grid_search_RECURSIVE(y_score, y_true, fairness_parameter)
    y_pred_new = np.where(a * y_score[:, 0] + b * y_score[:, 1] + c > 0, 1, 0)
    print('DDP after', get_DDP(y_pred_new, y_true[:, 1]))
    print(accuracy_score(np.where(a * y_score[:, 0] + b * y_score[:, 1] + c > 0, 1, 0), y_true[:, 0]))
    print(learn_fair_classifier_via_grid_search_RECURSIVE(y_score, y_true, fairness_parameter))
    print(learn_fair_classifier_via_grid_search_RECURSIVE(y_score, y_true, fairness_parameter, nr_of_recursions=2))


if __name__ == "__main__":
    test_routine()
