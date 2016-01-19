import numpy as np
from scipy.stats import mode
from scipy import delete
from sklearn.preprocessing import StandardScaler
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