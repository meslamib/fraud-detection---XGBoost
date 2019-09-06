# Financial Fraud and Anomalous Event Detection
## IEEE-CIS Fraud Detection from Kaggle Competition

# Introduction
## Data Description 
In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target isFraud.
The data is broken into two files identity and transaction, which are joined by TransactionID. Not all transactions have corresponding identity information.
Categorical Features - Transaction
ProductCD, card1 - card6, addr1, addr2, P_emaildomain, R_emaildomain, M1 - M9
Categorical Features - Identity
DeviceType, DeviceInfo, id_12 - id_38
The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).
Include more about the data from the Discussion page of competition host.
Data Preprocessing Phase
•	Feature Extraction
•	Data Type Portability
o	Categorical to numerical (One-Hot Encoding), numeric to categorical?, Text to numeric? , etc.
•	Data Cleaning is a process of preparing data for analysis. We diagnosed data for problems that require cleaning. 
o	Handling missing values (needs to be identified and addressed) and wrong data types, inconsistent/incorrect column names (capitalization, bad character)?, Outliers?, Duplicate rows (can bias analysis and needs to be found and dropped)?, untidy?, column types can signal unexpected data values, scaling and normalization
train_identity.shape : (144233, 41)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 144233 entries, 0 to 144232
Columns: 41 entries, TransactionID to DeviceInfo
dtypes: float64(23), int64(1), object(17)
memory usage: 45.1+ MB


train_transaction.shape : (590540, 394)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 590540 entries, 0 to 590539
Columns: 394 entries, TransactionID to V339
dtypes: float64(376), int64(4), object(14)
memory usage: 1.7+ GB

stored as object: object type is a generic type in pandas that stored as string. Should be int if there is no decimals or float otherwise. 

Exploratory data analysis: 
Frequency counts: df.column_name.value_counts(drapna=False)
We calculated summary statistics on numeric columns to spot outliers in the data. We used .describe() method. It returns the summary statistics only on the numeric columns and not missing values. 
Data visualization: great way to spot outliers and obvious errors. We used bar plots for discrete data counts and histograms for continuous data counts. Box plots are also good way to visualize all the basic summary statistics into a single figure that we can use to quickly compare across multiple categories. We can spot outliers, the min/max of our data,  and the 25, 50, and 75 percent quartiles of our data. 
Tidy data (Hadley Wickham): formalize the way we describe the shape of data. Gives us a goal when formatting our data. “standard way to organize data values within a dataset” . Three principles of tidy data:
1.	Columns represent separate variables
2.	Rows represent individual observations
3.	Each form of observational unit forms a table
Data format in tidy form are better for analysis and not necessarily for reporting. Tidy data makes it easier to fix common data problems. 
Problem that should be fixed: Columns containing values instead of variables. Solution: pd.melt(frame=df, id_vars=’col_name’, value_vars=[‘value_1’, ‘value_2’], var_name = ‘new_var_name’, value_name=’new_value_name’)

Maybe pd.pivot_table() is needed?
Do we need merging to tables?
Converting data types 
# object dtype is a general representation that is typically encoded as strings
# to string: df['col'] = df['col'].astype(str)
# to category: .astype('category') : make df smaller in memory, untilized by other python libraries for analysis
# numeric columns can be strings or vice-verca
# Is there any numeric data loaded as string/object because of missing value? pd.to_numeric(df['col'], errors = 'coerce')
String manipulation: using regular expressions to clean strings

Duplicate and missing data:
# delete duplicates with drop_duplicates()
# delete NaNs .dropna() or fill it with .fillna() with mean, median (if there is outliers), constant, or a string
# test your data with asserts
•	Data Reduction and Transformation
o	Sampling, Random Under-Sampling and Oversampling, Feature selection,  Dimensionality reduction (PCA, UMAP, Autoencoder, SVD, t-SNE)
Analytical Phase 
•	Outlier Detection
o	Extreme value analysis, Probabilistic models, Clustering for outlier detection, distance-based methods, density-based methods.
•	Data Classification
o	Rare Class Learning, Example reweighting, sampling methods, Synthetic Oversampling (SMOTE)
o	Ensemble Methods, XGBoost
Evaluation Phase 
•	
Notes from Previous Mistakes on Imbalanced Datasets:
•	Never test on the oversampled or undersampled dataset.
•	If we want to implement cross validation, remember to oversample or undersample your training data during cross-validation, not before!
•	Don't use accuracy score as a metric with imbalanced datasets (will be usually high and misleading), instead use f1-score, precision/recall score or confusion matrix
References:
