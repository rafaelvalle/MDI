# MDI
Missing Data Imputation Python Library (v. 0.1)

This repository offers techniques for handling missing data and encoding categorical data such that it is appropriate to neural network classifiers and other tasks. We provide six different imputation strategies and include examples using the ADULT dataset.

## Techniques for handling missing data
We categorize proposed imputation methods into six groups listed below [1]:

Case substitution
:One observation with missing data is replaced with another non-sampled obser- vation.

Summary statistic
: Replace the missing data with the mean, median, or mode of the feature vec- tor. Using a numerical approach directly is not appropriate for nonordinal categorical data.

One-hot
: Create a binary variable to indicate whether or not a specific feature is missing.

Hot deck and cold deck
: Compute the K-Nearest Neighbors of the observation with missing data and assign the mode of the K-neighbors to the missing data. algorithm.
Prediction Model
: Train a prediction model (e.g., random forests) to predict the missing value.

Factor analysis
: Perform factor analysis (e.g., principal component analysis (PCA)) on the design matrix, project the design matrix onto the first N eigenvectors and replace the missing values by the values that might be given by the projected design matrix.