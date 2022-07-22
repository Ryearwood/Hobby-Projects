# Important Information Regarding Use of PyCaret Library for Time-Series Forecasting

Dataset Acquired from Kaggle <a href="https://www.kaggle.com/chirag19/air-passengers" target="_blank">Here</a>

PyCaret’s Regression module default settings are not ideal for time series data - Default data preparation steps aren't suitable for sequential/ordered data and must be adjusted.

An example is splitting data into Train/test Splits - this is normally done with random shuffling, however for time series data specifically - we don't want recent dates included in the training set and historical dates as part of the test set respectively.

`We can either set data_split_shuffle = False in the setup function to avoid shuffling the dataset or manually split the data on specified date windows`

Cross-Validation also needs to be adjusted with regards to date ordering. PyCaret's regression module's default settings uses k-fold random cross-validation and this is not suitable for time-series data. 

`We designate PyCaret's validation setting to 'timeseries' in order to correctly adjust.`