import numpy as np
import cPickle as pickle
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from missing_data_imputation import Imputer

#declare csv headers
x = np.genfromtxt('adult-train-raw', delimiter=', ', dtype=object)

# binarize labels
labels = (np.array(x[:,-1]) == '>50K').astype(int)

# remove weight factor and label column
x = x[:,:-1]

# remove redundant education-number feature
relevant_cols = [i for i in xrange(x.shape[1]) if i != 4]
x = x[:, relevant_cols]

# store valid information for generating data
rows, cols = np.where(x == '?')
full_obs = [i for i in xrange(x.shape[0]) if i not in rows]

# enumerate parameters and instantiate Imputer
imp = Imputer()
missing_data_cond = lambda x : x == '?'
cat_cols = (1, 3, 4, 5, 6, 7, 8, 12)
n_neighbors = 5

# drop missing variables, binarize and save complete observations and labels
print 'imputing with drop'
data_drop = imp.drop(x, missing_data_cond)
data_drop_bin = imp.binarize_data(data_drop, cat_cols).astype(float)
scaler = StandardScaler().fit(data_drop_bin)
data_drop_bin_scaled = scaler.transform(data_drop_bin)

# replace missing values with random existing values
print 'imputing with replace'
data_replace = imp.replace(x, missing_data_cond)
data_replace_bin = imp.binarize_data(data_replace, cat_cols).astype(float)
scaler = StandardScaler().fit(data_replace_bin)
data_replace_bin_scaled = scaler.transform(data_replace_bin)
data_replace_bin_scaled.dump('../adult-dataset/data_replace_bin_scaled.np')

# replace missing values with feature mode
print 'imputing with mode'
data_mode = imp.summarize(x, mode, missing_data_cond)
data_mode_bin = imp.binarize_data(data_mode, cat_cols).astype(float)
scaler = StandardScaler().fit(data_mode_bin)
data_mode_bin_scaled = scaler.transform(data_mode_bin)

# replace categorical features with one hot row
print 'imputing with onehot'
data_onehot = imp.binarize_data(x, cat_cols).astype(float)
scaler = StandardScaler().fit(data_onehot)
data_onehot_scaled = scaler.transform(data_onehot)

# replace missing data with predictions
print 'imputing with predicted'
data_predicted = imp.predict(x, cat_cols, missing_data_cond)
data_predicted_bin = imp.binarize_data(data_predicted, cat_cols).astype(float)
scaler = StandardScaler().fit(data_predicted_bin)
data_predicted_bin_scaled = scaler.transform(data_predicted_bin)

# replace missing data with values obtained after factor analysis
print 'imputing with factor analysis'
data_facanal = imp.factor_analysis(x, cat_cols, missing_data_cond)
data_facanal_bin = imp.binarize_data(data_facanal, cat_cols).astype(float)
scaler = StandardScaler().fit(data_facanal_bin)
data_facanal_bin_scaled = scaler.transform(data_facanal_bin)

# replace missing data with knn
data_knn = imp.knn(x, n_neighbors, np.mean, missing_data_cond, cat_cols)
data_knn_bin = imp.binarize_data(data_knn, cat_cols).astype(float)
scaler = StandardScaler().fit(data_knn_bin)
data_knn_bin_scaled = scaler.transform(data_knn_bin)
