import numpy as np
from scipy.stats import mode, itemfreq
from scipy import delete
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC as SVM

from missing_data_imputation import Imputer


# declare csv headers
x = np.genfromtxt('data/house-votes-84.data', delimiter=',', dtype=object)

# enumerate parameters and instantiate Imputer
imp = Imputer()
missing_data_cond = lambda x: x == '?'
cat_cols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)
n_neighbors = 3 # lower for votes

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
print 'imputing with predicted values from random forest'
clf = RandomForestClassifier(n_estimators=100, criterion='gini')
data_rf = imp.predict(x, cat_cols, missing_data_cond, clf)

# replace missing data with predictions using SVM
print 'imputing with predicted values usng SVM'
clf = clf = SVM(
    penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', 
    fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, 
    random_state=None, max_iter=1000)
data_svm = imp.predict(x, cat_cols, missing_data_cond, clf)

# replace missing data with predictions using logistic regression
print 'imputing with predicted values usng logistic regression'
clf = LogisticRegression(
            penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
            intercept_scaling=1)
data_logistic = imp.predict(x, cat_cols, missing_data_cond, clf)

# replace missing data with values obtained after factor analysis
# print 'imputing with factor analysis'
# data_facanal = imp.factor_analysis(x, cat_cols, missing_data_cond)

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
freq_data['Raw data'] = compute_histogram(x[:,1], labels)
freq_data['Drop missing'] = compute_histogram(data_drop[:,1], labels)
freq_data['Mode replacement'] = compute_histogram(data_mode[:,1], labels)
freq_data['Random replacement'] = compute_histogram(data_replace[:,1], labels)
freq_data['RF prediction'] = compute_histogram(data_rf[:,1], labels)
freq_data['SVM prediction'] = compute_histogram(data_svm[:,1], labels)
freq_data['Logistic prediction'] = compute_histogram(data_svm[:,1], labels)
# freq_data['PCA imputation'] = compute_histogram(data_facanal[:,1], labels)
freq_data['KNN imputation'] = compute_histogram(data_knn[:,1], labels)

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

#fig.suptitle('Congressional vote records dataset', fontsize=15, fontweight='bold')
ax.set_xlabel('Vote categories', size=15)
ax.set_ylabel('Count', size=15)
ax.set_title('Congressional vote records dataset (N=435)', size=15, fontweight='bold')
ax.set_xticks(bins + width)
#ax.set_xticklabels(labels, rotation=45)
ax.set_xticklabels(('Missing','Yea','Nay'), rotation=45)
plt.legend(loc=2)
plt.tight_layout()
plt.show()
