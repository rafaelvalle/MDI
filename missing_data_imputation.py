import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse.linalg import svds
from scipy.stats import mode


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


class Imputer(object):
    def __init__(self):
        """
        Attributes
        ----------

        """

    def drop(self, x, missing_data_cond):
        # drop observations with missing values
        return x[np.sum(missing_data_cond(x), axis=1) == 0]


    def replace(self, x, missing_data_cond, in_place=False):
        """ Replace missing data with a random observation with data

        """
        if in_place:
            data = x
        else:
            data = np.copy(x)


        for col in xrange(x.shape[1]):
            nan_ids = missing_data_cond(x[:,col])
            val_ids = np.random.choice(np.where(nan_ids == False)[0],
                                       np.sum(nan_ids == True))
            data[nan_ids, col] = data[val_ids, col]
        return data


    def summarize(self, x, summary_func, missing_data_cond, in_place=False):
        """ Substitutes missing values with a statistical summary of each
        feature vector

        Parameters
        ----------
        x : numpy.array
            Assumes that each feature column is of single type. Converts
            digit string features to float.
        summary_func : function
            Summarization function to be used for imputation
            (mean, median, mode, max, min...)
        data_types: tuple
            Tuple with data types of each column
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        # replace missing values with the summarization function
        for col in xrange(x.shape[1]):
            nan_ids = missing_data_cond(x[:, col])
            if True in nan_ids:
                val = summary_func(x[~nan_ids, col])
                data[nan_ids, col] = val

        return data


    def one_hot(self, x, missing_data_cond, in_place=False):
        """Create a one-hot row for each observation

        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data, where rows are observations and
            columns are features

        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.

        Returns
        -------
        data : np.ndarray
            Matrix with categorical data replaced with one-hot rows
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        # find rows and columns with missing data
        _, miss_cols = np.where(missing_data_cond(data))
        miss_cols_uniq = np.unique(miss_cols)

        for miss_col in miss_cols_uniq:
            uniq_vals, indices = np.unique(data[:,miss_col],
                                          return_inverse=True)

            data = np.column_stack((data, np.eye(uniq_vals.shape[0],
                                                 dtype=int)[indices]))

        # remove categorical columns with missing data
        data = np.delete(data, miss_cols, 1)
        # val_cols = [n for n in xrange(data.shape[1]) if n not in miss_cols_uniq]
        # data = data[:, val_cols]
        return data


    def knn(self, x, k, summary_func, missing_data_cond, cat_cols,
            in_place=False, verbose=False):
        """ Replace missing values with the summary function of K-Nearest
        Neighbors

        Parameters
        ----------
        k : int
            Number of nearest neighbors to be used

        """

        def row_col_from_condensed_idx(n_obs, row):
            b = 1 -2 * n_obs
            x = np.floor((-b - np.sqrt(b**2 - 8*row))/2).astype(int)
            y = row + x*(b + x + 2)/2 + 1
            return (x, y)


        def condensed_idx_from_row_col(row, col, n_rows):
            if row > col:
                row, col = col, row

            return row*n_rows + col - row*(row+1)/2 - row - 1


        if in_place:
            data = x
        else:
            data = np.copy(x)

        imp = Imputer()

        # first transform features with categorical missing data into one hot
        data_complete = imp.one_hot(data, missing_data_cond)

        # binarize complete categorical variables and convert to int
        col = 0
        cat_ids_comp = []
        while col < max(cat_cols):
            if isinstance(data_complete[0, col], basestring) and not data_complete[0, col].isdigit():
                cat_ids_comp.append(col)
            col += 1

        data_complete = imp.binarize_data(data_complete, cat_ids_comp).astype(float)

        # normalize features
        scaler = StandardScaler().fit(data_complete)
        data_complete = scaler.transform(data_complete)

        # get indices of observations with nan
        miss_rows = np.unique(np.where(missing_data_cond(data))[0])
        n_obs = data_complete.shape[0]

        # compute distance matrix with nan values set to 0.0
        print 'Computing distance matrix'
        dist_cond = pdist(data_complete, metric='euclidean')

        print 'Substituting missing values'
        # substitute missing values with mode of knn
        # this code must be optimized for speed!!!
        for j in xrange(len(miss_rows)):
            miss_row_idx = miss_rows[j]

            # get indices of distances in condensed form
            ids_cond = [condensed_idx_from_row_col(miss_row_idx, idx, n_obs)
                         for idx in xrange(n_obs) if idx not in miss_rows]
            ids_cond = np.array(ids_cond, dtype=int)

            # compute k-nearest neighbors
            knn_ids_cond = ids_cond[np.argsort(dist_cond[ids_cond])[:k]]
            rows, cols = row_col_from_condensed_idx(n_obs, knn_ids_cond)

            # swap if necessary
            good_obs_ids = np.array([a for a in cols if a != miss_row_idx] +
                                    [b for b in rows if b != miss_row_idx],
                                    dtype=int)

            # cols with missing data
            obs_nan_cols = np.where(missing_data_cond(x[miss_row_idx]))[0]

            # get feature mode value given knn
            knn_mean_vals, _ = mode(data[:,obs_nan_cols][good_obs_ids])
            if verbose:
                print 'Substituting {}-th of {} total \n Value {}'.format(j,
                    len(miss_rows), knn_mean_vals)
            data[miss_row_idx, obs_nan_cols] = knn_mean_vals.flatten()
        return data


    def predict(self, x, cat_cols, missing_data_cond, in_place=False):
        """ Uses random forest for predicting missing values

        Parameters
        ----------
        cat_cols : int tuple
            Index of columns that are categorical

        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        # find rows and columns with missing data
        miss_rows, miss_cols = np.where(missing_data_cond(data))

        miss_cols_uniq = np.unique(miss_cols)
        valid_cols = [n for n in xrange(data.shape[1])
                      if n not in miss_cols_uniq]

        # factorize valid cols
        data_factorized = np.copy(data)

        # factorize categorical variables and store transformation
        factor_labels = {}
        for cat_col in cat_cols:
            factors, labels = pd.factorize(data[:, cat_col])
            factor_labels[cat_col] = labels
            data_factorized[:,cat_col] = factors

        # values are integers, convert accordingly
        data_factorized = data_factorized.astype(int)

        # update each column with missing features
        for miss_col in miss_cols_uniq:
            # edatatract valid observations given current column missing data
            valid_obs = [n for n in xrange(data.shape[0])
                         if data[n, miss_col] != '?']

            # prepare independent and dependent variables, valid obs only
            data_train = data_factorized[:, valid_cols][valid_obs]
            y_train = data_factorized[valid_obs, miss_col]

            # train random forest classifier
            rf_clf = RandomForestClassifier(n_estimators=100)
            rf_clf.fit(data_train, y_train)

            # given current feature, find obs with missing vals
            miss_obs_iddata = miss_rows[miss_cols == miss_col]

            # predict missing values
            y_hat = rf_clf.predict(data_factorized[:, valid_cols][miss_obs_iddata])

            # replace missing data with prediction
            data_factorized[miss_obs_iddata, miss_col] = y_hat

        # replace values on original data data
        for col in factor_labels.keys():
            data[:, col] = factor_labels[col][data_factorized[:, col]]

        return data


    def factor_analysis(self, x, cat_cols, missing_data_cond, threshold=0.9,
                        in_place = False):
        """ Performs principal component analisis and replaces missing data with
        values obtained from the data projected onto N principal components

        threshold : float
            Variance threshold that must be explained by eigen values.

        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        # factorize valid cols
        data_factorized = np.copy(data)

        # factorize categorical variables and store encoding
        factor_labels = {}
        for cat_col in cat_cols:
            factors, labels = pd.factorize(x[:, cat_col])
            factor_labels[cat_col] = labels
            data_factorized[:, cat_col] = factors

        data_factorized = data_factorized.astype(float)

        # questionable whether high variance = high importance.
        u, s, vt = svds(data_factorized, data_factorized.shape[1] - 1,
                        which = 'LM')

        # find number of eigenvalues that explain 90% of variance
        n_pcomps = 1
        while sum(s[-n_pcomps:]) / sum(s) < threshold:
            n_pcomps += 1

        # compute data projected onto principal components space
        data_factor_proj = np.dot(u[:,-n_pcomps:],
                   np.dot(np.diag(s[-n_pcomps:]), vt[-n_pcomps:,]))

        # get missing data indices
        nans = np.argwhere(missing_data_cond(x))

        # update data given projection
        for col in np.unique(nans[:, 1]):
            obs_ids = nans[nans[:, 1] == col, 0]
            proj_cats = np.clip(data_factor_proj[obs_ids, col],
                                0, len(factor_labels[col])-1).astype(int)
            data[obs_ids, col] = factor_labels[col][proj_cats]

        return data


    def factorize_data(self, x, cols, in_place=False):
        """Replace column in cols with one-hot representation of cols

        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data

        cols: tuple <int>
            Index of columns with categorical data

        Returns
        -------
        d : np.ndarray
            Matrix with categorical data replaced with one-hot rows
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        factors_labels = {}
        for col in cols:
            factors, labels = pd.factorize(data[:,col])
            factors_labels[col] = (factors_labels)
            data[:,col] = factors

        return data, factor_labels


    def binarize_data(self, x, cols, one_minus_one=True, in_place=False):
        """Replace column in cols with one-hot representation of cols

        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data, where rows are observations and
            columns are features

        cols: tuple <int>
            Index of columns with categorical data

        Returns
        -------
        d : np.ndarray
            Matrix with categorical data replaced with one-hot rows
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        for col in cols:
            uniq_vals, indices = np.unique(data[:,col],
                                          return_inverse=True)

            if one_minus_one:
                data = np.column_stack((data,
                    (np.eye(uniq_vals.shape[0], dtype=int)[indices] * 2) - 1))
            else:
                data = np.column_stack((data, np.eye(uniq_vals.shape[0],
                                                     dtype=int)[indices]))

        # remove columns with categorical variables
        val_cols = [n for n in xrange(data.shape[1]) if n not in cols]
        data = data[:, val_cols]
        return data
