import numpy as np
import seaborn as sbn
from scipy.stats import mode, itemfreq
from scipy import delete
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from missing_data_imputation import Imputer

plt.rcParams.update({'figure.autolayout': True})




# declare csv headers
x = np.genfromtxt('adult-train-raw', delimiter=', ', dtype=object)

# remove redundant education-number feature
x = delete(x, (4, 14), 1)

# enumerate parameters and instantiate Imputer
imp = Imputer()
missing_data_symbol = '?'
miss_data_cond = lambda x: x == missing_data_symbol
cat_cols = (1, 3, 4, 5, 6, 7, 8, 12)
n_neighbors = 5
miss_data_cols = np.unique(np.where(miss_data_cond(x))[1])
ratios = np.arange(10, 100, 10)


def perturb_data(x, columns, ratio, miss_data_cond, missing_data_symbol,
                 in_place=False):
    """Perturbs data by substituting existing values with missing data symbol
    such that each feature has a minimum missing data ratio
    """

    if in_place:
        data = x
    else:
        data = np.copy(x)

    n_miss_req = int(data.shape[0] * ratio)
    indices = {}
    for col in columns:
        miss_ids_binary = miss_data_cond(data[:, col])
        n_miss = sum(miss_ids_binary)
        remaining = n_miss_req - n_miss

        # if not enough missing data, perturb more
        if remaining > 0:
            perturb_ids = np.random.choice(np.where(~miss_ids_binary)[0],
                                           remaining, replace=False)
            data[perturb_ids, col] = missing_data_symbol
            indices[col] = perturb_ids

    return data, indices

for ratio in ratios:
    print 'Experiments on {}% missing data'.format(ratio)
    data, feat_imp_ids = perturb_data(x, miss_data_cols, 0.01*ratio,
                                        miss_data_cond, missing_data_symbol)

    # drop observations with missing variables
    print 'imputing with drop'
    data_drop = imp.drop(data, miss_data_cond)

    # replace missing values with random existing values
    print 'imputing with random replacement'
    data_rep = imp.replace(data, miss_data_cond)

    # replace missing values with feature summary
    print 'imputing with feature summarization (mode)'
    summ_func = lambda x: mode(x)[0]
    data_mode = imp.summarize(data, summ_func, miss_data_cond)

    # replace categorical features with one summer row
    print 'imputing with one-summer'
    data_onehot = imp.binarize_data(data, cat_cols)

    # replace missing data with predictions using random forest
    print 'imputing with predicted values'
    data_rf = imp.predict(data, cat_cols, miss_data_cond)

    # replace missing data with values obtained after factor analysis
    print 'imputing with factor analysis'
    data_pca = imp.factor_analysis(data, cat_cols, miss_data_cond)

    # replace missing data with knn
    """
    print 'imputing with K-Nearest Neighbors'
    data_knn = imp.knn(data, n_neighbors, np.mean, miss_data_cond, cat_cols)
    """

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

    # helper function to plot histograms
    def plot_histogram(freq_data, labels, axes, axis, width, title,
                       color_mapping):
        n_methods = len(freq_data.keys())
        labels = sorted(freq_data.values()[0].keys())
        bins = np.arange(len(labels))

        for i in xrange(n_methods):
            key = sorted(freq_data.keys())[i]
            offset = i*2*width/float(n_methods)
            values = [freq_data[key][label] for label in labels]
            axes.ravel()[axis].bar(bins+offset, values,
                                   width, label=key,
                                   color=plt.cm.YlGnBu(color_mapping[key]),
                                   align='center')

        axes.ravel()[axis].set_xlim(bins[0]-0.5, bins[-1]+width+0.5)
        axes.ravel()[axis].set_title(title)
        axes.ravel()[axis].set_xticks(bins + width)
        axes.ravel()[axis].set_xticklabels(labels, rotation=90,
                                           fontsize='small')
        axes.ravel()[axis].legend(loc='best', prop={'size': 8},
                                  shadow=True, fancybox=True)

    # compute error rate for each feature with missing data
    error_rates = {}
    error_rates['Mode'] = compute_error_rate(x, data_mode, feat_imp_ids)
    error_rates['RandomReplace'] = compute_error_rate(x, data_rep, feat_imp_ids)
    error_rates['RandomForest'] = compute_error_rate(x, data_rf, feat_imp_ids)
    error_rates['PCA'] = compute_error_rate(x, data_pca, feat_imp_ids)
    error_rates['KNN'] = compute_error_rate(x, data_knn, feat_imp_ids)

    # plot results
    methods = ['RawData', 'Drop', 'RandomReplace', 'Mode', 'RandomForest',
               'PCA', 'KNN']
    color_mapping = {}
    for i in xrange(len(methods)):
        color_mapping[methods[i]] = (i+1) / float(len(methods))

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    width = .25

    # compute and plot histograms for each feature with missing data
    for i in xrange(len(miss_data_cols)):
        col = miss_data_cols[i]
        labels = np.unique(x[:, col])
        freq_data = {}
        freq_data['RawData'] = compute_histogram(x[:, col], labels)
        freq_data['RandomReplace'] = compute_histogram(data_rep[:, col], labels)
        freq_data['Mode'] = compute_histogram(data_mode[:, col], labels)
        freq_data['Drop'] = compute_histogram(data_drop[:, col], labels)
        freq_data['RandomForest'] = compute_histogram(data_rf[:, col], labels)
        freq_data['PCA'] = compute_histogram(data_pca[:, col], labels)
        freq_data['KNN'] = compute_histogram(data_knn[:,1], labels)
        plot_histogram(freq_data, labels, axes, i, width, col, color_mapping)

    # plot error rate bars given feature and imputation technique
    n_methods = len(error_rates.keys())
    bins = np.arange(len(feat_imp_ids))
    width = .25

    for i in xrange(n_methods):
        key = sorted(error_rates.keys())[i]
        offset = i*width/float(n_methods)
        values = [error_rates[key][feat] for feat in sorted(error_rates[key])]
        axes.ravel()[-1].bar(bins+offset, values, width, label=key,
                             color=plt.cm.YlGnBu(color_mapping[key]),
                             align='center')
    axes.ravel()[-1].set_xlim(bins[0]-0.5, bins[-1]+width+0.5)
    axes.ravel()[-1].set_xticks(bins + width)
    axes.ravel()[-1].set_xticklabels(sorted(feat_imp_ids.keys()))
    axes.ravel()[-1].legend(loc='best', prop={'size': 8},
                            shadow=True, fancybox=True)
    axes.ravel()[-1].set_title('Error rates')

    plt.title('Imputation with {}% missing data'.format(ratio))
    plt.savefig('miss_ratio_{}.png'.format(ratio), dpi=300)