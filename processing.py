import numpy as np
from scipy.stats import itemfreq
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC as SVM


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def impute(data, imputer, imp_method, params_dict):
    imp_data = None

    if imp_method == 'RandomReplace':
        imp_data = imputer.replace(data, params_dict['miss_data_cond'])
    elif imp_method == 'Drop':
        imp_data = imputer.drop(data, params_dict['miss_data_cond'])
    elif imp_method == 'Summary':
        imp_data = imputer.summarize(data,
                                     params_dict['summary_func'],
                                     params_dict['miss_data_cond'])
    elif imp_method == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=100, criterion='gini')
        imp_data = imputer.predict(data,
                                   params_dict['cat_cols'],
                                   params_dict['miss_data_cond'],
                                   clf)

    elif imp_method == 'SVM':
        clf = SVM()
        imp_data = imputer.predict(data,
                                   params_dict['cat_cols'],
                                   params_dict['miss_data_cond'],
                                   clf)
    elif imp_method == 'LogisticRegression':
        clf = LogisticRegression()
        imp_data = imputer.predict(data,
                                   params_dict['cat_cols'],
                                   params_dict['miss_data_cond'],
                                   clf)
    elif imp_method == 'PCA':
        imp_data = imputer.factor_analysis(data,
                                           params_dict['cat_cols'],
                                           params_dict['miss_data_cond'],
                                           technique='PCA')
    elif imp_method == 'KNN':
        imp_data = imputer.knn(data,
                               params_dict['n_neighbors'],
                               params_dict['knn_summary_func'],
                               params_dict['miss_data_cond'],
                               params_dict['cat_cols'])
    elif imp_method == 'Identity':
        imp_data = data
    else:
        raise Exception("Imputation method {} is not valid".format(imp_method))
    return imp_data


def perturbate_data(x, cols, ratio, monotone, missing_data_symbol, mnar=None,
                    in_place=False):
    """Perturbs data by substituting existing values with missing data symbol
    such that each feature has a minimum missing data ratio


    Parameters
    ----------
    x : np.ndarray
        Matrix with categorical data, where rows are observations and
        columns are features
    cols : int tuple
        index of columns that are categorical
    ratio : float [0, 1]
        Ratio of observations in data to have missing data
    missing_data_symbol : str
        String that represents missing data in data
    monotone: boolean
        Non-monotone: Any observation and feature can present a missing
            value. Restrict the number of missing values in a observations
            to not more than half of the features.
        Monotone: set to missing all the values of 30% of randomly selected
            features with categorical variables
    mnar: tuple
    	Will perturb only items in the x matrix that matches items in the tuple
    	MNAR will suppress monotone
    """

    def zero():
        return 0

    if in_place:
        data = x
    else:
        data = np.copy(x)

    n_perturbations = int(len(x) * ratio)
    if mnar is not None:
    	mask = []
    	[mask.extend(np.argwhere(data == item).tolist()) for item in mnar]
    	mask = np.array(mask)
    	n_perturbations = int(len(mask) * ratio)
    	if n_perturbations < 1:
    		raise Exception('Number of perturbations is smaller than 1.')
    	else:
	    	mask_rows = np.random.choice(mask.shape[0], 
	    		max(int(len(mask) * ratio), 1), replace=False)
	    	coords = mask[mask_rows]
	    	data[coords[:, 0], coords[:, 1]] = missing_data_symbol
	    miss_dict = defaultdict(list)
	    for i in coords:
	    	miss_dict[i[1]].extend(i[0])
    elif monotone:
        missing_mask = np.random.choice((0, 1), data[:, cols].shape, True,
                                        (1-ratio, ratio)).astype(bool)

        miss_dict = defaultdict(list)
        for i in xrange(len(cols)):
            rows = np.where(missing_mask[:, i])[0]
            data[rows, cols[i]] = missing_data_symbol
            miss_dict[cols[i]] = rows
        """
        cols = np.random.choice(cols, int(len(cols) * monotone))
        rows = np.random.randint(0, len(data), n_perturbations)
        cols = np.random.choice(cols, n_perturbations)

        data[rows, cols] = missing_data_symbol
        miss_dict = defaultdict(list)
        for (row, col) in np.dstack((rows, cols))[0]:
            miss_dict[col].append(row)
        """
    else:
        # slow
        row_col_miss = defaultdict(zero)
        miss_dict = defaultdict(list)
        i = 0
        while i < n_perturbations:
            row = np.random.randint(0, len(data))
            col = np.random.choice(cols)

            # proceed if less than half the features are missing
            if row_col_miss[row] < len(cols) * 0.5 \
                    and data[row, col] != missing_data_symbol:
                data[row, col] = missing_data_symbol
                row_col_miss[row] += 1
                miss_dict[col].append(row)
                i += 1

    return data, miss_dict


def compute_histogram(data, labels):
    histogram = dict(itemfreq(data))
    for label in labels:
        if label not in histogram:
            histogram[label] = .0
    return histogram


def compute_error_rate(y, y_hat, feat_imp_ids):
    error_rate = {}
    for col, ids in feat_imp_ids.items():
        errors = sum(y[ids, col] != y_hat[ids, col])
        error_rate[col] = errors / float(len(ids))

    return error_rate
