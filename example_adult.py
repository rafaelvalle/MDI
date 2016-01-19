import numpy as np
from scipy.stats import mode, itemfreq
from scipy import delete
import matplotlib.pylab as plt

from missing_data_imputation import Imputer


# declare csv headers
x = np.genfromtxt('adult-train-raw', delimiter=', ', dtype=object)

# remove redundant education-number feature
x = delete(x, (4, 14), 1)

# enumerate parameters and instantiate Imputer
imp = Imputer()
missing_data_cond = lambda x: x == '?'
cat_cols = (1, 3, 4, 5, 6, 7, 8, 12)
n_neighbors = 5

# drop observations with missing variables
print 'imputing with drop'
data_drop = imp.drop(x, missing_data_cond)

# replace missing values with random existing values
print 'imputing with random replacement'
data_replace = imp.replace(x, missing_data_cond)

# replace missing values with feature summary
print 'imputing with feature summarization (mode)'
summ_func = lambda x: mode(x)[0]
data_mode = imp.summarize(x, summ_func, missing_data_cond)

# replace categorical features with one hot row
print 'imputing with one-hot'
data_onehot = imp.binarize_data(x, cat_cols)

# replace missing data with predictions using random forest
print 'imputing with predicted values'
data_predicted = imp.predict(x, cat_cols, missing_data_cond)

# replace missing data with values obtained after factor analysis
print 'imputing with factor analysis'
data_facanal = imp.factor_analysis(x, cat_cols, missing_data_cond)

# replace missing data with knn
print 'imputing with K-Nearest Neighbors'
data_knn = imp.knn(x, n_neighbors, np.mean, missing_data_cond, cat_cols)

def compute_histogram(data, labels):
    histogram = itemfreq(sorted(data))
    for label in labels:
        if label not in histogram[:,0]:
            histogram = np.vstack((histogram,
                                   np.array([[label, 0]], dtype=object)))
    histogram = histogram[histogram[:,0].argsort()]
    return histogram

# compute histograms
labels = np.unique(x[:,1])
freq_data = {}
freq_data['Raw Data'] = compute_histogram(x[:,1], labels)
freq_data['Mode Replacement'] = compute_histogram(data_mode[:,1], labels)
freq_data['Drop'] = compute_histogram(data_drop[:,1], labels)
freq_data['RF prediction'] = compute_histogram(data_predicted[:,1], labels)
freq_data['PCA Imputation'] = compute_histogram(data_facanal[:,1], labels)
freq_data['KNN Imputation'] = compute_histogram(data_knn[:,1], labels)

# plot histograms given feature with missing data
n_methods = len(freq_data.keys())
bins = np.arange(len(labels))
width = .25
fig, ax = plt.subplots(figsize=(12,8))

for i in xrange(n_methods):
    key = sorted(freq_data.keys())[i]
    offset = i*2*width/float(n_methods)
    ax.bar(bins+offset, freq_data[key][:,1].astype(int), width, label=key,
           color=plt.cm.hot(i/float(n_methods)), align='center')

ax.set_xticks(bins + width)
ax.set_xticklabels(labels, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
